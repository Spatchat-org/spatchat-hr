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


@dataclass
class DBBMMParams:
    location_error_m: float = 30.0
    window_size: int = 31
    margin: int = 11
    raster_resolution_m: float = 50.0
    buffer_m: float = 1000.0
    n_substeps: int = 40
    isopleths: Tuple[int, ...] = (50, 95)


@dataclass
class DBBMMResult:
    geotiff: str
    isopleths: List[Dict]


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
    Compute dBBMM per animal on a Web Mercator (EPSG:3857) meter grid.

    Returns
    -------
    { animal_id: DBBMMResult(geotiff=<webmerc tif>, isopleths=[{percent, area_sq_km, geometry}, ...]) }
    """
    if params is None:
        params = DBBMMParams()

    # Use Web Mercator for a globally valid meter-based grid.
    to_eq  = Transformer.from_crs(4326, 3857, always_xy=True)  # lon/lat -> Web Mercator (m)
    to_wgs = Transformer.from_crs(3857, 4326, always_xy=True)  # Web Mercator -> lon/lat

    def reproj_geom(geom):
        return shp_transform(lambda x, y, z=None: to_wgs.transform(x, y), geom)

    # Normalize inputs
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

        xs, ys = to_eq.transform(sub["lon"].values, sub["lat"].values)
        ts = sub["timestamp"].astype("int64").to_numpy() / 1e9  # seconds

        # drop zero/negative time gaps
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

        # Rolling variance of velocity (pragmatic local sigma^2)
        w = int(max(5, params.window_size if params.window_size % 2 == 1 else params.window_size + 1))
        pad = w // 2
        v_pad = np.pad(v, (pad, pad), mode="edge")
        v2 = pd.Series(v_pad).rolling(window=w, center=True, min_periods=max(5, w // 3)).var().to_numpy()[pad:-pad]
        if len(v2) != len(v):
            v2 = np.resize(v2, len(v))
        baseline_var_v = (params.raster_resolution_m / 5.0) ** 2
        v2 = np.where(np.isfinite(v2) & (v2 > 0), v2, baseline_var_v)

        T = dt
        sigma2 = v2

        # Raster grid in EPSG:3857 (note y goes DOWN as row increases)
        res = float(params.raster_resolution_m)
        buf = float(params.buffer_m)
        minx, maxx = float(np.min(xs) - buf), float(np.max(xs) + buf)
        miny, maxy = float(np.min(ys) - buf), float(np.max(ys) + buf)
        width = int(max(1, math.ceil((maxx - minx) / res)))
        height = int(max(1, math.ceil((maxy - miny) / res)))
        transform = Affine.translation(minx, maxy) * Affine.scale(res, -res)

        # Helper: world -> (col,row) in this grid (no 0.5 center offset; we do that when sampling)
        def world_to_cr(x: float, y: float) -> Tuple[float, float]:
            c = (x - minx) / res
            r = (maxy - y) / res  # inverted Y
            return c, r

        UD = np.zeros((height, width), dtype=np.float64)
        cell_area = res * res

        nseg = len(steps)
        n_sub = int(max(5, params.n_substeps))
        loc_err2 = float(params.location_error_m) ** 2

        updated_windows = 0
        for i in range(nseg):
            x0, y0 = coords[i]
            x1, y1 = coords[i + 1]
            Ti = max(T[i], 1e-3)
            sig2 = max(sigma2[i], (params.raster_resolution_m / 10.0) ** 2)

            # generous spatial window (3 sigma + half segment length)
            seg_len = max(1.0, np.hypot(x1 - x0, y1 - y0))
            sigma_max = math.sqrt(loc_err2 + sig2 * (0.25 * Ti))  # max at s=0.5
            radius = 3.0 * (sigma_max + 0.5 * seg_len)
            minx_w = min(x0, x1) - radius
            maxx_w = max(x0, x1) + radius
            miny_w = min(y0, y1) - radius
            maxy_w = max(y0, y1) + radius

            # --- CORRECT Y ORIENTATION: top uses maxy_w, bottom uses miny_w ---
            c0f, r_topf = world_to_cr(minx_w, maxy_w)  # top-left
            c1f, r_botf = world_to_cr(maxx_w, miny_w)  # bottom-right

            c0 = int(max(0, math.floor(min(c0f, c1f))))
            c1 = int(min(width - 1, math.ceil(max(c0f, c1f))))
            r0 = int(max(0, math.floor(min(r_topf, r_botf))))
            r1 = int(min(height - 1, math.ceil(max(r_topf, r_botf))))

            if r1 < r0 or c1 < c0:
                continue

            # Substep positions along the segment
            ss = np.linspace(0.0, 1.0, n_sub, endpoint=True)
            xs_sub = x0 + ss * (x1 - x0)
            ys_sub = y0 + ss * (y1 - y0)

            # Pixel center coords within window
            rows = np.arange(r0, r1 + 1)
            cols = np.arange(c0, c1 + 1)
            if rows.size == 0 or cols.size == 0:
                continue

            xx = minx + (cols + 0.5) * res
            yy = maxy - (rows + 0.5) * res
            XX, YY = np.meshgrid(xx, yy)

            # Accumulate Gaussian kernels
            for x_s, y_s, s in zip(xs_sub, ys_sub, ss):
                var_s = loc_err2 + sig2 * (s * (1.0 - s) * Ti)
                if var_s <= 0:
                    continue
                dx = XX - x_s
                dy = YY - y_s
                inv_two = 1.0 / (2.0 * var_s)
                kernel = np.exp(-(dx * dx + dy * dy) * inv_two) / (2.0 * math.pi * var_s)
                UD[rows[:, None], cols[None, :]] += kernel

            updated_windows += 1

        # Debug before normalization
        nz = int(np.count_nonzero(UD))
        print(
            f"[dBBMM] animal={animal} pre-norm UD stats: "
            f"sum={UD.sum() * cell_area:.6e}, max={UD.max():.6e}, nonzero={nz}/{UD.size}, windows={updated_windows}"
        )

        total = UD.sum() * cell_area
        if total > 0:
            UD /= total
        else:
            print(f"[dBBMM] animal={animal} WARNING: UD total mass is zero; check timestamps/units.")

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

        # Extract isopleth polygons
        flat = UD.ravel()
        order = np.argsort(flat)[::-1]
        flat_sorted = flat[order]
        mass = np.cumsum(flat_sorted * cell_area)
        total_mass = mass[-1] if mass.size else 0.0

        env_list: List[Dict] = []
        if total_mass > 0 and len(flat_sorted) > 0:
            for p in sorted(set(int(x) for x in params.isopleths if 1 <= int(x) <= 100)):
                target = (p / 100.0) * total_mass
                idx = int(np.searchsorted(mass, target, side="left"))
                thr = float(flat_sorted[min(idx, len(flat_sorted) - 1)])

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
                    mp_wgs = reproj_geom(mp)
                    env_list.append({
                        "percent": int(p),
                        "area_sq_km": area_sq_km,
                        "geometry": shp_mapping(mp_wgs),
                    })

        nz_post = int(np.count_nonzero(UD))
        print(
            f"[dBBMM] animal={animal} post-norm UD stats: "
            f"sum={UD.sum() * cell_area:.6e}, max={UD.max():.6e}, nonzero={nz_post}/{UD.size}, tif={tif_path}"
        )

        results[str(animal)] = DBBMMResult(geotiff=tif_path, isopleths=env_list)

    return results
