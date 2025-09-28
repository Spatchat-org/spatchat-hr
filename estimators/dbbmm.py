# =============================================
# File: estimators/dbbmm.py
# =============================================
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import math
import os
import numpy as np
import pandas as pd
from affine import Affine
import rasterio
from rasterio.features import shapes as rio_shapes
from shapely.geometry import shape as shp_shape, mapping as shp_mapping, MultiPolygon, Polygon
from shapely.ops import unary_union, transform as shp_transform
from pyproj import Transformer


# -------------------------------------------------------------
# Parameters & result container
# -------------------------------------------------------------

@dataclass
class DBBMMParams:
    # Observation location error (1-sigma, meters)
    location_error_m: float = 30.0
    # Odd-sized window used only for a smoothed speed fallback (not critical now)
    window_size: int = 31
    # Not used in this simplified variant, kept for future parity
    margin: int = 11
    # Raster resolution (meters) in EPSG:3857
    raster_resolution_m: float = 50.0
    # Extra buffer around track extent for raster (meters)
    buffer_m: float = 1000.0
    # Number of sub-steps to integrate per segment
    n_substeps: int = 40
    # Isopleths to extract from the UD (percent)
    isopleths: Tuple[int, ...] = (50, 95)
    # Scale factor on diffusion (lets us tune contrast if needed)
    sigma2_scale: float = 1.0  # 1.0 works well with the speed-based diffusion below


@dataclass
class DBBMMResult:
    geotiff: str                    # path to GeoTIFF in EPSG:3857
    isopleths: List[Dict]           # [{percent, area_sq_km, geometry (WGS84 GeoJSON)}]


# -------------------------------------------------------------
# Core
# -------------------------------------------------------------

def compute_dbbmm(
    df: pd.DataFrame,
    id_col: str,
    x_col: str,
    y_col: str,
    ts_col: str,
    params: Optional[DBBMMParams] = None,
    outputs_dir: str = "outputs",
) -> Dict[str, DBBMMResult]:
    """
    Compute dynamic Brownian Bridge UDs per animal.

    Inputs
    ------
    df : DataFrame with [id_col, x_col (lon), y_col (lat), ts_col] in WGS84.
    outputs_dir : where GeoTIFFs are written (one per animal)

    Returns
    -------
    { animal_id: DBBMMResult(geotiff=<webmerc tif>, isopleths=[...]) }
    """
    if params is None:
        params = DBBMMParams()

    # Use Web Mercator (EPSG:3857) for a globally valid meter grid that Folium can overlay easily.
    to_eq  = Transformer.from_crs(4326, 3857, always_xy=True)  # lon/lat -> Web Mercator meters
    to_wgs = Transformer.from_crs(3857, 4326, always_xy=True)  # meters -> lon/lat

    def reproj_geom_to_wgs(geom):
        return shp_transform(lambda x, y, z=None: to_wgs.transform(x, y), geom)

    # Basic cleaning
    df0 = df[[id_col, x_col, y_col, ts_col]].dropna().copy()
    df0.columns = ["animal_id", "lon", "lat", "timestamp"]
    df0["timestamp"] = pd.to_datetime(df0["timestamp"], errors="coerce")
    df0 = df0.dropna(subset=["timestamp"])

    results: Dict[str, DBBMMResult] = {}
    os.makedirs(outputs_dir, exist_ok=True)

    for animal, sub in df0.groupby("animal_id"):
        sub = sub.sort_values("timestamp").reset_index(drop=True)
        if len(sub) < 2:
            continue

        # Project to metric CRS for modelling
        xs, ys = to_eq.transform(sub["lon"].values, sub["lat"].values)

        # Seconds since epoch
        ts = sub["timestamp"].astype("int64").to_numpy() / 1e9  # pandas-safe
        dt = np.diff(ts)
        # Drop zero/negative time intervals (duplicate/improper ordering)
        valid = dt > 0
        if not np.all(valid):
            keep_idx = np.insert(valid, 0, True)
            xs, ys, ts = xs[keep_idx], ys[keep_idx], ts[keep_idx]
            if len(xs) < 2:
                continue
            dt = np.diff(ts)

        coords = np.column_stack([xs, ys])
        steps = np.diff(coords, axis=0)
        d = np.hypot(steps[:, 0], steps[:, 1])      # meters
        v = d / np.maximum(dt, 1e-6)                # m/s (segment speed)

        # --- Diffusion parameter (m^2/s) -----------------------------
        # Simple, dimensionally-consistent proxy: proportional to speed times a
        # characteristic length scale (location error or half a pixel), then
        # scaled by a user knob.
        length_scale = float(max(params.location_error_m, params.raster_resolution_m * 0.5))
        sigma2 = np.maximum(v, 0.1) * length_scale * float(params.sigma2_scale)  # m^2/s
        # --------------------------------------------------------------

        # --- Raster grid in EPSG:3857 ---
        res = float(params.raster_resolution_m)
        buf = float(params.buffer_m)
        minx, maxx = float(np.min(xs) - buf), float(np.max(xs) + buf)
        miny, maxy = float(np.min(ys) - buf), float(np.max(ys) + buf)
        width = int(max(1, math.ceil((maxx - minx) / res)))
        height = int(max(1, math.ceil((maxy - miny) / res)))
        transform = Affine.translation(minx, maxy) * Affine.scale(res, -res)

        UD = np.zeros((height, width), dtype=np.float64)
        cell_area = res * res  # m^2 per pixel

        # --- Integrate Gaussian kernels along each segment (Brownian bridge) ---
        nseg = len(steps)
        n_sub = int(max(5, params.n_substeps))
        loc_err2 = float(params.location_error_m) ** 2  # m^2

        for i in range(nseg):
            x0, y0 = coords[i]
            x1, y1 = coords[i + 1]
            Ti = max(dt[i], 1e-3)               # seconds
            sig2 = max(sigma2[i], 1e-3)         # m^2/s (floor to avoid collapse)

            # Bounding window ~ 3 sigma around the segment to limit work
            seg_len = max(1.0, np.hypot(x1 - x0, y1 - y0))
            sigma_max = math.sqrt(loc_err2 + sig2 * (0.25 * Ti))  # s*(1-s) max at 0.5
            radius = 3.0 * (sigma_max + 0.5 * seg_len)
            minx_w = min(x0, x1) - radius
            maxx_w = max(x0, x1) + radius
            miny_w = min(y0, y1) - radius
            maxy_w = max(y0, y1) + radius

            c0, r0 = ~transform * (minx_w, miny_w)
            c1, r1 = ~transform * (maxx_w, maxy_w)
            r0 = int(max(0, math.floor(r0)))
            r1 = int(min(height - 1, math.ceil(r1)))
            c0 = int(max(0, math.floor(c0)))
            c1 = int(min(width - 1, math.ceil(c1)))
            if r1 < r0 or c1 < c0:
                continue

            # Substep positions along the segment
            ss = np.linspace(0.0, 1.0, n_sub, endpoint=True)
            xs_sub = x0 + ss * (x1 - x0)
            ys_sub = y0 + ss * (y1 - y0)

            rows = np.arange(r0, r1 + 1)
            cols = np.arange(c0, c1 + 1)
            xx = minx + (cols + 0.5) * res
            yy = maxy - (rows + 0.5) * res
            XX, YY = np.meshgrid(xx, yy)

            for x_s, y_s, s in zip(xs_sub, ys_sub, ss):
                # Bridge variance at fraction s of segment
                var_s = loc_err2 + sig2 * (s * (1.0 - s) * Ti)  # m^2
                if var_s <= 0:
                    continue
                dx = XX - x_s
                dy = YY - y_s
                inv_two_var = 1.0 / (2.0 * var_s)
                kernel = np.exp(-(dx * dx + dy * dy) * inv_two_var) / (2.0 * math.pi * var_s)
                UD[rows[:, None], cols[None, :]] += kernel

        # Normalize UD to integrate to 1 over space
        total = UD.sum() * cell_area
        if total > 0:
            UD /= total

        # --- DEBUG: print UD stats so we can see if it has contrast ---
        nz = UD[UD > 0]
        print(
            f"[dBBMM] animal={animal} UD stats: "
            f"min={UD.min():.3e}, max={UD.max():.3e}, sum={UD.sum():.3e}, "
            f"nonzero={nz.size}/{UD.size}"
        )
        if nz.size:
            qs = np.percentile(nz, [50, 90, 99, 99.9])
            print(
                f"[dBBMM] animal={animal} UD nz percentiles: "
                f"50%={qs[0]:.3e}, 90%={qs[1]:.3e}, 99%={qs[2]:.3e}, 99.9%={qs[3]:.3e}"
            )

        # Write GeoTIFF (EPSG:3857)
        tif_path = os.path.join(outputs_dir, f"dbbmm_{str(animal).replace(' ', '_')}.tif")
        with rasterio.open(
            tif_path,
            "w",
            driver="GTiff",
            height=UD.shape[0],
            width=UD.shape[1],
            count=1,
            dtype=rasterio.float32,
            crs="EPSG:3857",
            transform=transform,
            compress="lzw",
        ) as dst:
            dst.write(UD.astype(np.float32), 1)

        # --- Extract isopleth polygons (threshold by cumulative mass) ---
        flat = UD.ravel()
        if flat.size == 0 or total <= 0:
            results[str(animal)] = DBBMMResult(geotiff=tif_path, isopleths=[])
            continue

        order = np.argsort(flat)[::-1]
        flat_sorted = flat[order]
        mass = np.cumsum(flat_sorted * cell_area)
        total_mass = mass[-1]

        env_list: List[Dict] = []
        for p in sorted(set(int(x) for x in params.isopleths if 1 <= int(x) <= 100)):
            target = (p / 100.0) * total_mass
            idx = int(np.searchsorted(mass, target, side="left"))
            thr = float(flat_sorted[min(idx, len(flat_sorted) - 1)])

            # Binary mask where UD >= threshold
            mask = (UD >= thr).astype(np.uint8)

            polygons: List[Polygon] = []
            for geom, val in rio_shapes(mask, mask=mask.astype(bool), transform=transform):
                if val == 1:
                    poly = shp_shape(geom)  # EPSG:3857
                    if not poly.is_empty and poly.area > 0:
                        polygons.append(poly)

            if not polygons:
                continue

            mp = unary_union(polygons)
            if isinstance(mp, (Polygon, MultiPolygon)):
                area_sq_km = float(mp.area / 1e6)
                mp_wgs = reproj_geom_to_wgs(mp)
                env_list.append({
                    "percent": int(p),
                    "area_sq_km": area_sq_km,
                    "geometry": shp_mapping(mp_wgs),  # WGS84 GeoJSON
                })

        results[str(animal)] = DBBMMResult(geotiff=tif_path, isopleths=env_list)

    return results
