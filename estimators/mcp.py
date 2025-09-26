import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon
from pyproj import Transformer
from .. import storage

def mcp_polygon(latitudes, longitudes, percent=95):
    points = np.column_stack((longitudes, latitudes))
    if len(points) < 3:
        return None
    centroid = points.mean(axis=0)
    dists = np.linalg.norm(points - centroid, axis=1)
    n_keep = max(3, int(len(points) * (percent / 100.0)))
    keep_idx = np.argsort(dists)[:n_keep]
    points_kept = points[keep_idx]
    if len(points_kept) < 3:
        return None
    hull = ConvexHull(points_kept)
    hull_points = points_kept[hull.vertices]
    return hull_points

def add_mcps(df, percent_list):
    for percent in percent_list:
        for animal in df["animal_id"].unique():
            if animal not in storage.mcp_results:
                storage.mcp_results[animal] = {}
            if percent not in storage.mcp_results[animal]:
                track = df[df["animal_id"] == animal]
                hull_points = mcp_polygon(track['latitude'].values, track['longitude'].values, percent)
                if hull_points is not None:
                    poly = Polygon([(lon, lat) for lon, lat in hull_points])
                    transformer = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
                    coords_proj = [transformer.transform(lon, lat) for lon, lat in hull_points]
                    poly_proj = Polygon(coords_proj)
                    area_km2 = poly_proj.area / 1e6
                    storage.mcp_results[animal][percent] = {"polygon": poly, "area": area_km2}
