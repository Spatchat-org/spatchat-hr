# =============================================
# File: estimators/locoh.py
# =============================================
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import math
import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon, MultiPoint, mapping
from shapely.ops import unary_union
from scipy.spatial import cKDTree
from pyproj import Transformer

# --- Public API --------------------------------------------------------------
@dataclass
class LoCoHParams:
    method: str = "k"  # one of {"k", "a", "r"}
    k: int = 10         # used when method == "k"
    a: Optional[float] = None  # distance adaptive threshold (meters)
    r: Optional[float] = None  # fixed radius (meters)
    isopleths: Tuple[int, ...] = (50, 95)  # % isopleths to return


def compute_locoh(
    df: pd.DataFrame,
    id_col: str,
    x_col: str,
    y_col: str,
    params: Optional[LoCoHParams] = None,
) -> Dict:
    """
    Compute Local Convex Hull (LoCoH) home ranges, per animal, returning polygons for
    requested isopleths and summary areas (sq km).

    Expects coordinates in WGS84 (lon/lat). Internally projects to World Equal Area
    (EPSG:6933) for distance/area calculations, returning GeoJSON in WGS84.
    """
    if params is None:
        params = LoCoHParams()

    if params.method not in {"k", "a", "r"}:
        raise ValueError("LoCoH method must be one of 'k', 'a', or 'r'")

    # Clean & sort for stable results
    gdf = df[[id_col, x_col, y_col]].dropna().copy()
    gdf.columns = ["animal_id", "lon", "lat"]

    # Build transformers
    to_eq = Transformer.from_crs(4326, 6933, always_xy=True)
    to_wgs = Transformer.from_crs(6933, 4326, always_xy=True)

    # Project to meters for KDTree & area
    xs, ys = to_eq.transform(gdf["lon"].values, gdf["lat"].values)
    gdf["x_m"] = xs
    gdf["y_m"] = ys

    out = {"method": params.method, "isopleths": list(params.isopleths), "animals": {}}

    for animal, sub in gdf.groupby("animal_id"):
        sub = sub.reset_index(drop=True)
        coords_m = np.column_stack([sub["x_m"].values, sub["y_m"].values])
        tree = cKDTree(coords_m)

        # Build local hulls per point
        hulls_m: List[Polygon] = []
        for i, p in enumerate(coords_m):
            if params.method == "k":
                k = max(3, min(params.k, len(coords_m)))
                dists, idxs = tree.query(p, k)
                # cKDTree returns scalar for k==1; ensure array
                idxs = np.atleast_1d(idxs)
                pts = coords_m[idxs]
            elif params.method == "r":
                if not params.r:
                    raise ValueError("r-LoCoH requires params.r in meters")
                idxs = tree.query_ball_point(p, params.r)
                if len(idxs) < 3:
                    # fallback to nearest 3 to form a hull
                    _, idxs = tree.query(p, 3)
                    idxs = np.atleast_1d(idxs)
                pts = coords_m[idxs]
            else:  # adaptive a-LoCoH
                if not params.a:
                    raise ValueError("a-LoCoH requires params.a in meters")
                # find neighbors whose cumulative distance (sorted) ≤ a
                dists, idxs = tree.query(p, k=min(20, len(coords_m)))
                dists = np.atleast_1d(dists)
                idxs = np.atleast_1d(idxs)
                order = np.argsort(dists)
                cumdist = 0.0
                kept = []
                for j in order:
                    nd = float(dists[j])
                    cumdist += nd
                    kept.append(int(idxs[j]))
                    if cumdist >= float(params.a):
                        break
                if len(kept) < 3:
                    # ensure at least triangle
                    extra_k = max(3, min(5, len(coords_m)))
                    _, extra = tree.query(p, extra_k)
                    kept = list(np.unique(np.atleast_1d(extra)))
                pts = coords_m[kept]

            if len(pts) >= 3:
                hull = MultiPoint(pts).convex_hull
                if isinstance(hull, Polygon):
                    hulls_m.append(hull)

        if not hulls_m:
            continue

        # Sort hulls by area ascending per LoCoH definition
        hulls_m.sort(key=lambda h: h.area)

        # Build cumulative unions and record polygons at desired isopleths
        total_area = sum(h.area for h in hulls_m)
        targets = sorted(set(params.isopleths))
        polys_by_iso: Dict[int, Polygon] = {}

        cum_union = None
        cum_area = 0.0
        t_idx = 0
        target_fracs = [t / 100.0 for t in targets]

        for h in hulls_m:
            cum_area += h.area
            cum_union = h if cum_union is None else cum_union.union(h)
            while t_idx < len(target_fracs) and (cum_area / total_area) >= target_fracs[t_idx]:
                polys_by_iso[targets[t_idx]] = cum_union
                t_idx += 1
            if t_idx >= len(target_fracs):
                break

        # Ensure we have the max isopleth (e.g., 95) even if not exact
        if targets[-1] not in polys_by_iso:
            polys_by_iso[targets[-1]] = cum_union if cum_union is not None else unary_union(hulls_m)

        # Reproject to WGS84 and compute areas (sq km), preserving MultiPolygons & holes
        from shapely.ops import transform as shp_transform
        
        def reproj_geom(geom):
            # transform expects a function (x, y) -> (x', y')
            return shp_transform(lambda x, y, z=None: to_wgs.transform(x, y), geom)
        
        results = []
        for iso, geom_m in polys_by_iso.items():
            area_sqkm = geom_m.area / 1e6  # m^2 -> km^2
            geom_wgs = reproj_geom(geom_m)  # preserves MultiPolygon + interiors
            results.append({
                "isopleth": int(iso),
                "area_sq_km": float(area_sqkm),
                "geometry": mapping(geom_wgs),
            })

        out["animals"][str(animal)] = {
            "n_points": int(len(sub)),
            "isopleths": results,
        }

    return out

