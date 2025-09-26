# estimators/mcp.py
import numpy as np
from shapely.geometry import Polygon
from pyproj import Transformer
import storage  # absolute import (works when app.py is run as a script)

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
    # Convex hull via shapely (robust, avoids scipy dependency here if desired)
    try:
        from scipy.spatial import ConvexHull  # keep your original approach if installed
        hull = ConvexHull(points_kept)
        hull_points = points_kept[hull.vertices]
    except Exception:
        # Fallback using shapely convex hull of a MultiPoint
        from shapely.geometry import MultiPoint
        hull_poly = MultiPoint([(x, y) for x, y in points_kept]).convex_hull
        if hull_poly.is_empty or not isinstance(hull_poly, Polygon):
            return None
        hull_points = np.array(hull_poly.exterior.coords)[:-1]  # drop closing vertex

    return hull_points

def add_mcps(df, percent_list):
    """
    Populate storage.mcp_results with polygons and areas for each animal_id/percent.
    """
    for percent in percent_list:
        for animal in df["animal_id"].unique():
            if animal not in storage.mcp_results:
                storage.mcp_results[animal] = {}
            if percent in storage.mcp_results[animal]:
                continue

            track = df[df["animal_id"] == animal]
            hull_points = mcp_polygon(track['latitude'].values, track['longitude'].values, percent)
            if hull_points is None:
                continue

            # Build polygon in lon/lat order
            poly = Polygon([(lon, lat) for lon, lat in hull_points])

            # Area in km^2 via Web Mercator projection
            transformer = Transformer.from_crs("epsg:4326", "epsg:3857", always_xy=True)
            coords_proj = [transformer.transform(lon, lat) for lon, lat in hull_points]
            poly_proj = Polygon(coords_proj)
            area_km2 = float(poly_proj.area) / 1e6

            storage.mcp_results[animal][percent] = {"polygon": poly, "area": area_km2}
