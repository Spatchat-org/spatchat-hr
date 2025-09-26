# estimators/mcp.py
import numpy as np
from shapely.geometry import Polygon
from pyproj import Transformer
import storage  # absolute import; relies on storage.py in root

def _mcp_polygon(latitudes, longitudes, percent=95):
    pts = np.column_stack((longitudes, latitudes))
    if len(pts) < 3: return None
    centroid = pts.mean(axis=0)
    dists = np.linalg.norm(pts - centroid, axis=1)
    n_keep = max(3, int(len(pts) * (percent / 100.0)))
    keep_idx = np.argsort(dists)[:n_keep]
    pts_kept = pts[keep_idx]
    if len(pts_kept) < 3: return None
    from scipy.spatial import ConvexHull
    hull = ConvexHull(pts_kept)
    return pts_kept[hull.vertices]

def add_mcps(df, percent_list):
    for percent in percent_list:
        for animal in df["animal_id"].unique():
            storage.mcp_results.setdefault(animal, {})
            if percent in storage.mcp_results[animal]:
                continue
            track = df[df["animal_id"] == animal]
            hp = _mcp_polygon(track['latitude'].values, track['longitude'].values, percent)
            if hp is None: continue
            poly = Polygon([(lon, lat) for lon, lat in hp])

            tf = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
            coords_proj = [tf.transform(lon, lat) for lon, lat in hp]
            poly_proj = Polygon(coords_proj)
            area_km2 = poly_proj.area / 1e6

            storage.mcp_results[animal][percent] = {"polygon": poly, "area": area_km2}
