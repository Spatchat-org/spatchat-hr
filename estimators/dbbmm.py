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
# Public API
# -------------------------------------------------------------

@dataclass
class DBBMMParams:
    """Parameters for dynamic Brownian Bridge Movement Model (dBBMM)."""
    # Observation location error (1-sigma, meters)
    location_error_m: float = 30.0
    # Odd-sized window of steps used to estimate local variance of velocity
    window_size: int = 31
    # Margin steps excluded at both ends when estimating local variance
    margin: int = 11
    # Raster resolution (meters)
    raster_resolution_m: float = 50.0
    # Extra buffer around track extent for raster (meters)
    buffer_m: float = 1000.0
    # Number of sub-steps to integrate per segment
    n_substeps: int = 40
    # Isopleths to extract from the UD (percent)
    isopleths: Tuple[int, ...] = (50, 95)


@dataclass
class DBBMMResult:
    geotiff: str
    isopleths: List[Dict]


# -------------------------------------------------------------
# Helpers
# -------------------------------------------------------------

def _cdf_threshold(prob_arr: np.ndarray, target: float) -> float | None:
    """Return threshold t such that pixels >= t cover `target` fraction of mass."""
    a = np.asarray(prob_arr, dtype=float)
    a = a[np.isfinite(a)]
    a = a[a > 0]
    if a.size == 0:
        return None
    a.sort()          # ascending
    a = a[::-1]       # descending
    cdf = np.cumsum(a) / float(a.sum())
    idx = int(np.searchsorted(cdf, target, side="left"))
    idx = min(max(idx, 0), a.size - 1)
    return float(a[idx])


def _reproj_geom_to_wgs(geom, src_crs: str | int = "EPSG:6933"):
    tr = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
    return shp_transform(lambda x, y, z=None: tr.transform(x, y), geom)


def _isopleths_from_ud(
    prob_arr: np.ndarray,
    transform,
    src_crs: str | int,
    levels: Tuple[int, ...],
) -> List[Dict]:
    """
    From a UD raster (probabilities ~ sum to 1), build isopleth polygons at `levels`.
    Geometry is returned in WGS84 (GeoJSON), area in km².
    """
    if not np.isfinite(prob_arr).any() or float(np.nanmax(prob_arr)) <= 0:
        return []

    # For constant-resolution rasters, pixel area is constant so we can skip area weighting
    out: List[Dict] = []
    src_crs_str = str(src_crs or "EPSG:6933")

    for p in sorted({int(l) for l in levels if 1 <= int(l) <= 100}):
        thr = _cdf_threshold(prob_arr, p / 100.0)
        if thr is None:
            continue
        # Tiny epsilon to avoid dropping plateaus due to float equality
        mask = np.nan_to_num(prob_arr, nan=0.0) >= (thr - 1e-12)

        # Vectorize to polygons (in src_crs)
        polygons: List[Polygon] = []
        for geom, val in rio_shapes(mask.astype(np.uint8), mask=mask, transform=transform):
            if val == 1:
                poly = shp_shape(geom)
                if not poly.is_empty and poly.area > 0:
                    polygons.append(poly)
        if not polygons:
            continue

        mp = unary_union(polygons)  # src_crs
        # area in m² -> km² (assumes src_crs meters; our UD is in EPSG:6933)
        area_km2 = float(mp.area) / 1e6
        mp_wgs = _reproj_geom_to_wgs(mp, src_crs_str)
        out.append({
            "percent": int(p),
            "area_sq_km": area_km2,
            "geometry": shp_mapping(mp_wgs),
        })

    return out


# -------------------------------------------------------------
# Main
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
    Compute dBBMM per animal.

    Inputs
    ------
    df : DataFrame with [id_col, x_col (lon), y_col (lat), ts_col] (WGS84 lon/lat).
    params : DBBMMParams
    outputs_dir : where GeoTIFFs are written (one per animal)

    Returns
    -------
    dict: { animal_id: DBBMMResult(geotiff=path, isopleths=[{percent, area_sq_km, geometry}, ...]) }
    """
    if params is None:
        params = DBBMMParams()

    # Prepare transformers
    to_eq = Transformer.from_crs(4326, 6933, always_xy=True)   # lon/lat -> meters
    # to_wgs used only via helper when reprojecting geometry

    # Normalize and sanity check
    df0 = df[[id_col, x_col, y_col, ts_col]].dropna().copy()
    df0.columns = ["animal_id", "lon", "lat", "timestamp"]
    df0["timestamp"] = pd.to_datetime(df0["timestamp"], errors="coerce")
    df0 = df0.dropna(subset=["timestamp"])  # drop non-parsable times

    results: Dict[str, DBBMMResult] = {}
    os.makedirs(outputs_dir, exist_ok=True)

    for animal, sub in df0.groupby("animal_id"):
        sub = sub.sort_values("timestamp").reset_index(drop=True)
        if len(sub) < 2:
            continue

        # Project to meters for calculations
        xs, ys = to_eq.transform(sub["lon"].values, sub["lat"].values)
        ts = sub["timestamp"].astype("int64").to_numpy() / 1e9  # seconds since epoch

        # Guard against duplicated timestamps
        dt = np.diff(ts)
        valid = dt > 0
        if not np.all(valid):
            keep_idx = np.insert(valid, 0, True)
            xs, ys, ts = xs[keep_idx], ys[keep_idx], ts[keep_idx]
            if len(xs) < 2:
                continue
            dt = np.diff(ts)

        coords = np.column_stack([xs, ys])
        steps = np.diff(coords, axis=0)
        d = np.hypot(steps[:, 0], steps[:, 1])  # meters
        v = d / np.maximum(dt, 1e-6)            # m/s

        # --- Local variance of velocity by rolling window (pragmatic dBBMM) ---
        w = int(max(5, params.window_size if params.window_size % 2 == 1 else params.window_size + 1))
        m = int(max(1, params.margin))
        pad = w // 2
        v_pad = np.pad(v, (pad, pad), mode="edge")
        v2 = pd.Series(v_pad).rolling(window=w, center=True, min_periods=max(5, w // 3)).var().to_numpy()[pad:-pad]
        if len(v2) != len(v):
            v2 = np.resize(v2, len(v))
        baseline_var_v = (params.raster_resolution_m / 5.0) ** 2  # (m/s)^2 floor
        v2 = np.where(np.isfinite(v2) & (v2 > 0), v2, baseline_var_v)

        # Segment times
        T = dt  # seconds
        # Brownian motion variance parameter per segment (m^2 / s)
        sigma2 = v2

        # --- Build raster grid (EPSG:6933 meters) ---
        res = float(params.raster_resolution_m)
        buf = float(params.buffer_m)
        minx, maxx = float(np.min(xs) - buf), float(np.max(xs) + buf)
        miny, maxy = float(np.min(ys) - buf), float(np.max(ys) + buf)
        width = int(max(1, math.ceil((maxx - minx) / res)))
        height = int(max(1, math.ceil((maxy - miny) / res)))
        transform = Affine.translation(minx, maxy) * Affine.scale(res, -res)

        UD = np.zeros((height, width), dtype=np.float64)
        cell_area = res * res  # m^2 per pixel

        # --- Integrate along each segment ---
        nseg = len(steps)
        n_sub = int(max(5, params.n_substeps))
        loc_err2 = float(params.location_error_m) ** 2

        for i in range(nseg):
            x0, y0 = coords[i]
            x1, y1 = coords[i + 1]
            Ti = max(T[i], 1e-3)  # seconds
            sig2 = max(sigma2[i], (params.raster_resolution_m / 10.0) ** 2)

            # bbox window ~3 sigma + half segment length
            seg_len = max(1.0, np.hypot(x1 - x0, y1 - y0))
            sigma_max = math.sqrt(loc_err2 + sig2 * (0.25 * Ti))
            radius = 3.0 * (sigma_max + 0.5 * seg_len)
            minx_w = min(x0, x1) - radius
            maxx_w = max(x0, x1) + radius
            miny_w = min(y0, y1) - radius
            maxy_w = max(y0, y1) + radius

            # pixel window indices
            c0, r0 = ~transform * (minx_w, miny_w)
            c1, r1 = ~transform * (maxx_w, maxy_w)
            r0 = int(max(0, math.floor(r0)))
            r1 = int(min(height - 1, math.ceil(r1)))
            c0 = int(max(0, math.floor(c0)))
            c1 = int(min(width - 1, math.ceil(c1)))
            if r1 < r0 or c1 < c0:
                continue

            # Substep positions and s-fractions along the segment
            ss = np.linspace(0.0, 1.0, n_sub, endpoint=True)
            xs_sub = x0 + ss * (x1 - x0)
            ys_sub = y0 + ss * (y1 - y0)

            # Vectorized pixel centers for the window
            rows = np.arange(r0, r1 + 1)
            cols = np.arange(c0, c1 + 1)
            xx = minx + (cols + 0.5) * res
            yy = maxy - (rows + 0.5) * res
            XX, YY = np.meshgrid(xx, yy)

            # Accumulate Gaussian kernels for each sub-step
            for x_s, y_s, s in zip(xs_sub, ys_sub, ss):
                var_s = loc_err2 + sig2 * (s * (1.0 - s) * Ti)
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

        # Write GeoTIFF (EPSG:6933)
        tif_path = os.path.join(outputs_dir, f"dbbmm_{str(animal).replace(' ', '_')}.tif")
        with rasterio.open(
            tif_path,
            "w",
            driver="GTiff",
            height=UD.shape[0],
            width=UD.shape[1],
            count=1,
            dtype=rasterio.float32,
            crs="EPSG:6933",
            transform=transform,
            compress="lzw",
        ) as dst:
            dst.write(UD.astype(np.float32), 1)

        # --- Extract isopleth polygons (50/95 by default) ---
        iso_list = _isopleths_from_ud(
            prob_arr=UD,
            transform=transform,
            src_crs="EPSG:6933",
            levels=params.isopleths if params.isopleths else (95,),
        )

        results[str(animal)] = DBBMMResult(geotiff=tif_path, isopleths=iso_list)

    return results
