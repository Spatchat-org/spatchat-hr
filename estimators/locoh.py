# =============================================
# File: estimators/locoh.py
# =============================================
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from shapely.geometry import MultiPoint, Polygon, mapping
from shapely.ops import unary_union, transform as shp_transform
from scipy.spatial import cKDTree
from pyproj import Transformer


# ---------- Public API ---------------------------------------------------------

@dataclass
class LoCoHParams:
    """
    Parameters for Local Convex Hull (LoCoH).

    method:
      - "k":   k-nearest neighbours (default)
      - "a":   adaptive; include neighbours until cumulative distance <= a (meters)
      - "r":   fixed radius in meters

    isopleths:
      Percent envelopes to return, e.g., (50, 95). Per-hull facets are
      always returned separately for visualization.
    """
    method: str = "k"                         # one of {"k","a","r"}
    k: int = 10                               # used when method == "k"
    a: Optional[float] = None                 # meters, when method == "a"
    r: Optional[float] = None                 # meters, when method == "r"
    isopleths: Tuple[int, ...] = (50, 95)     # percent envelopes


def compute_locoh(
    df: pd.DataFrame,
    id_col: str,
    x_col: str,
    y_col: str,
    params: Optional[LoCoHParams] = None,
) -> Dict:
    """
    Compute Local Convex Hull (LoCoH) per animal.

    Inputs
    ------
    df : DataFrame with at least [id_col, x_col, y_col] in WGS84 (lon/lat).
    id_col : animal/individual identifier column.
    x_col, y_col : longitude, latitude column names (WGS84).
    params : LoCoHParams.

    Returns
    -------
    dict with:
      {
        "method": "k"|"a"|"r",
        "isopleths": [50,95,...],
        "animals": {
          "<animal_id>": {
            "n_points": int,
            "isopleths": [
              {"isopleth": 50, "area_sq_km": float, "geometry": <GeoJSON>},
              ...
            ],
            "facets": [
              {"cum_frac": 0.23, "cum_percent": 23, "area_sq_km": 0.012, "geometry": <GeoJSON>},
              ...
            ]
          }, ...
        }
      }

    Notes
    -----
    - Distances/areas are computed in EPSG:6933 (World Equidistant Cylindrical; meters).
    - Geometries returned in WGS84; MultiPolygons and holes are preserved.
    - Isopleths are formed by sorting local hulls by area (ascending) and
      cumulatively unioning them until a target fraction of the *sum of hull areas*
      is reached (classic LoCoH convention).
    """
    if params is None:
        params = LoCoHParams()

    method = params.method.lower()
    if method not in {"k", "a", "r"}:
        raise ValueError("LoCoH method must be one of {'k','a','r'}")

    # Basic validation
    if method == "a" and (params.a is None or float(params.a) <= 0):
        raise ValueError("a-LoCoH requires a positive 'a' (meters).")
    if method == "r" and (params.r is None or float(params.r) <= 0):
        raise ValueError("r-LoCoH requires a positive 'r' (meters).")

    # Shallow copy & standardize column names
    gdf = df[[id_col, x_col, y_col]].dropna().copy()
    gdf.columns = ["animal_id", "lon", "lat"]

    # Project to EPSG:6933 for distances/areas; keep WGS84 for output
    to_eq = Transformer.from_crs(4326, 6933, always_xy=True)   # lon/lat -> meters
    to_wgs = Transformer.from_crs(6933, 4326, always_xy=True)  # meters  -> lon/lat

    # Helper: shapely geometry reprojection preserving MultiPolygons/holes
    def reproj_geom(geom):
        return shp_transform(lambda x, y, z=None: to_wgs.transform(x, y), geom)

    # Precompute projected coordinates
    xs, ys = to_eq.transform(gdf["lon"].values, gdf["lat"].values)
    gdf["x_m"] = xs
    gdf["y_m"] = ys

    out: Dict = {"method": method, "isopleths": list(params.isopleths), "animals": {}}

    for animal, sub in gdf.groupby("animal_id"):
        sub = sub.reset_index(drop=True)
        if len(sub) < 3:
            # Not enough points for any hulls
            out["animals"][str(animal)] = {"n_points": int(len(sub)), "isopleths": [], "facets": []}
            continue

        coords_m = np.column_stack([sub["x_m"].values, sub["y_m"].values])
        tree = cKDTree(coords_m)

        # Build local hulls for each point
        hulls: List[Polygon] = []
        n = len(coords_m)
        # Defensive cap for k to keep complexity bounded
        k_cap = max(3, min(params.k if method == "k" else 10, n, 50))

        for i, p in enumerate(coords_m):
            if method == "k":
                k = max(3, min(params.k, n, 50))
                dists, idxs = tree.query(p, k)
                idxs = np.atleast_1d(idxs)
                pts = coords_m[idxs]
            elif method == "r":
                idxs = tree.query_ball_point(p, float(params.r))
                if len(idxs) < 3:
                    _, idxs = tree.query(p, 3)
                    idxs = np.atleast_1d(idxs)
                pts = coords_m[idxs]
            else:  # method == "a"
                # Take a small neighborhood, then keep sorted-by-distance until cumdist >= a
                dists, idxs = tree.query(p, k=min(max(10, min(2 * k_cap, n)), n))
                dists = np.atleast_1d(dists)
                idxs = np.atleast_1d(idxs)
                order = np.argsort(dists)
                cumdist = 0.0
                kept: List[int] = []
                for j in order:
                    nd = float(dists[j])
                    cumdist += nd
                    kept.append(int(idxs[j]))
                    if cumdist >= float(params.a):  # meters
                        break
                if len(kept) < 3:
                    _, extra = tree.query(p, 3)
                    kept = list(np.unique(np.atleast_1d(extra)))
                pts = coords_m[kept]

            # Build convex hull if possible
            # MultiPoint.convex_hull may return LineString/Point if <3 unique pts
            if len(pts) >= 3:
                hull = MultiPoint(pts).convex_hull
                # Only keep polygonal hulls (ignore degenerate)
                if isinstance(hull, Polygon):
                    hulls.append(hull)

        if not hulls:
            out["animals"][str(animal)] = {"n_points": int(len(sub)), "isopleths": [], "facets": []}
            continue

        # Sort local hulls by area ascending (LoCoH definition)
        hulls.sort(key=lambda h: h.area)

        # ----- Per-hull facets for visualization (cumulative rank) -----
        total_area = sum(h.area for h in hulls) or 1.0
        facets = []
        cum = 0.0
        for h in hulls:
            cum += h.area
            frac = min(1.0, cum / total_area)
            pct = int(round(frac * 100))
            facets.append({
                "cum_frac": float(frac),
                "cum_percent": int(pct),
                "area_sq_km": float(h.area / 1e6),
                "geometry": mapping(reproj_geom(h)),
            })

        # ----- Cumulative unions at requested isopleths -----
        targets = sorted({int(t) for t in params.isopleths if 1 <= int(t) <= 100})
        polys_by_iso: Dict[int, Polygon] = {}

        cum_union = None
        cum_area = 0.0
        t_idx = 0
        target_fracs = [t / 100.0 for t in targets]

        for h in hulls:
            cum_area += h.area
            cum_union = h if cum_union is None else cum_union.union(h)
            while t_idx < len(target_fracs) and (cum_area / total_area) >= target_fracs[t_idx]:
                polys_by_iso[targets[t_idx]] = cum_union
                t_idx += 1
            if t_idx >= len(target_fracs):
                break

        if targets and targets[-1] not in polys_by_iso:
            polys_by_iso[targets[-1]] = cum_union if cum_union is not None else unary_union(hulls)

        # Reproject unions, compute areas (km²)
        envelopes = []
        for iso, geom_m in polys_by_iso.items():
            geom_wgs = reproj_geom(geom_m)
            envelopes.append({
                "isopleth": int(iso),
                "area_sq_km": float(geom_m.area / 1e6),  # area in projected CRS
                "geometry": mapping(geom_wgs),
            })

        out["animals"][str(animal)] = {
            "n_points": int(len(sub)),
            "isopleths": envelopes,
            "facets": facets,
        }

    return out
