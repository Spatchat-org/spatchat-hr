# estimators/kde.py
import os
import json
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Polygon, MultiPolygon, mapping
from shapely.ops import unary_union
from pyproj import Transformer
from skimage import measure
from sklearn.neighbors import KernelDensity
import storage  # absolute import

def _utm_epsg_for_region(latitudes, longitudes):
    lon0, lat0 = np.mean(longitudes), np.mean(latitudes)
    utm_zone = int((lon0 + 180) // 6) + 1
    return (32600 + utm_zone) if lat0 >= 0 else (32700 + utm_zone)

def kde_home_range(latitudes, longitudes, percent=95, animal_id="animal", grid_size=200):
    epsg_utm = _utm_epsg_for_region(latitudes, longitudes)
    to_utm = Transformer.from_crs("epsg:4326", f"epsg:{epsg_utm}", always_xy=True)
    to_latlon = Transformer.from_crs(f"epsg:{epsg_utm}", "epsg:4326", always_xy=True)

    x, y = to_utm.transform(longitudes, latitudes)
    xy = np.vstack([x, y]).T
    n = xy.shape[0]

    # Silverman's rule of thumb with guards
    if n > 1:
        stds = np.std(xy, axis=0, ddof=1)
        h = np.power(4 / (3 * n), 1 / 5) * float(np.mean(stds))
        if h < 1:
            h = 30.0
    else:
        h = 30.0

    margin = 3 * h
    xmin, xmax = x.min() - margin, x.max() + margin
    ymin, ymax = y.min() - margin, y.max() + margin
    x_grid = np.linspace(xmin, xmax, grid_size)
    y_grid = np.linspace(ymin, ymax, grid_size)
    X, Y = np.meshgrid(x_grid, y_grid)
    grid_points = np.vstack([X.ravel(), Y.ravel()]).T

    kde = KernelDensity(bandwidth=h, kernel="gaussian")
    kde.fit(xy)
    Z = np.exp(kde.score_samples(grid_points)).reshape(X.shape)

    # Normalize to integrate ~1 over area
    cell_area = (x_grid[1] - x_grid[0]) * (y_grid[1] - y_grid[0])
    Z /= (Z.sum() * cell_area)

    # Threshold for desired percent
    Z_flat = Z.ravel()
    idx_desc = np.argsort(Z_flat)[::-1]
    cumsum = np.cumsum(Z_flat[idx_desc] * cell_area)
    target = percent / 100.0
    idx = min(np.searchsorted(cumsum, target), len(Z_flat[idx_desc]) - 1)
    threshold = Z_flat[idx_desc][idx]
    mask = Z >= threshold

    Z_masked = np.where(mask, Z, 0)
    total_prob = Z_masked.sum() * cell_area
    if total_prob > 0:
        Z_masked /= total_prob

    # Raster-to-vector contour
    contours = measure.find_contours(mask.astype(float), 0.5)
    polygons = []
    for contour in contours:
        px, py = contour[:, 1], contour[:, 0]
        utm_xs = np.interp(px, np.arange(grid_size), x_grid)
        utm_ys = np.interp(py, np.arange(grid_size), y_grid)
        poly = Polygon(zip(utm_xs, utm_ys)).buffer(0)
        if poly.is_valid and poly.area > 0:
            polygons.append(poly)

    if not polygons:
        return None, None, None, None

    mpoly_utm = unary_union(polygons)

    def utm_poly_to_latlon(poly):
        if poly.is_empty:
            return None
        if isinstance(poly, Polygon):
            ext_lon, ext_lat = to_latlon.transform(*poly.exterior.xy)
            interiors = [to_latlon.transform(*interior.xy) for interior in poly.interiors]
            return Polygon(list(zip(ext_lon, ext_lat)),
                           [list(zip(int_lon, int_lat)) for int_lon, int_lat in interiors])
        elif isinstance(poly, MultiPolygon):
            geoms = [utm_poly_to_latlon(p) for p in poly.geoms if not p.is_empty]
            return MultiPolygon([g for g in geoms if g is not None])
        return None

    mpoly_latlon = utm_poly_to_latlon(mpoly_utm)
    area_km2 = float(mpoly_utm.area) / 1e6

    os.makedirs("outputs", exist_ok=True)
    safe_id = str(animal_id).replace(" ", "_").replace("/", "_")
    tiff_fp = os.path.join("outputs", f"kde_{safe_id}_{percent}.tif")
    geojson_fp = os.path.join("outputs", f"kde_{safe_id}_{percent}.geojson")

    # Save raster in EPSG:4326
    lon_sw, lat_sw = to_latlon.transform(xmin, ymin)
    lon_ne, lat_ne = to_latlon.transform(xmax, ymax)
    with rasterio.open(
        tiff_fp, "w",
        driver="GTiff",
        height=Z_masked.shape[0], width=Z_masked.shape[1],
        count=1, dtype=Z_masked.dtype,
        crs="EPSG:4326",
        transform=from_origin(
            lon_sw, lat_ne,
            (lon_ne - lon_sw) / Z_masked.shape[1],
            (lat_ne - lat_sw) / Z_masked.shape[0],
        )
    ) as dst:
        dst.write(np.flipud(Z_masked), 1)

    with open(geojson_fp, "w") as f:
        json.dump(mapping(mpoly_latlon), f)

    return mpoly_latlon, area_km2, tiff_fp, geojson_fp

def add_kdes(df, percent_list):
    for percent in percent_list:
        for animal in df["animal_id"].unique():
            if animal not in storage.kde_results:
                storage.kde_results[animal] = {}
            if percent in storage.kde_results[animal]:
                continue

            track = df[df["animal_id"] == animal]
            res = kde_home_range(
                track['latitude'].values, track['longitude'].values,
                percent=percent, animal_id=animal
            )
            if res is None:
                continue
            mpoly, area_km2, tiff_fp, geojson_fp = res
            if mpoly is None:
                continue

            storage.kde_results[animal][percent] = {
                "contour": mpoly,
                "area": area_km2,
                "geotiff": tiff_fp,
                "geojson": geojson_fp,
            }
