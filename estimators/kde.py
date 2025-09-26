# estimators/kde.py
import os, json
import numpy as np
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Polygon, MultiPolygon, mapping
from shapely.ops import unary_union
from pyproj import Transformer
from skimage import measure
from sklearn.neighbors import KernelDensity
import storage  # absolute import

def _kde_core(latitudes, longitudes, percent=95, grid_size=200):
    lon0, lat0 = np.mean(longitudes), np.mean(latitudes)
    zone = int((lon0 + 180) // 6) + 1
    epsg_utm = 32600 + zone if lat0 >= 0 else 32700 + zone
    to_utm = Transformer.from_crs("epsg:4326", f"epsg:{epsg_utm}", always_xy=True)
    to_ll  = Transformer.from_crs(f"epsg:{epsg_utm}", "epsg:4326", always_xy=True)

    x, y = to_utm.transform(longitudes, latitudes)
    XY = np.vstack([x, y]).T

    n = len(XY)
    if n > 1:
        stds = np.std(XY, axis=0, ddof=1)
        h = (4/(3*n))**(1/5) * np.mean(stds)
        if h < 1: h = 30.0
    else:
        h = 30.0

    m = 3*h
    xmin, xmax = x.min()-m, x.max()+m
    ymin, ymax = y.min()-m, y.max()+m
    gx = np.linspace(xmin, xmax, grid_size)
    gy = np.linspace(ymin, ymax, grid_size)
    Xg, Yg = np.meshgrid(gx, gy)
    grid = np.vstack([Xg.ravel(), Yg.ravel()]).T

    kde = KernelDensity(bandwidth=h, kernel="gaussian").fit(XY)
    Z = np.exp(kde.score_samples(grid)).reshape(Xg.shape)

    cell_area = (gx[1]-gx[0])*(gy[1]-gy[0])
    Z /= (Z.sum() * cell_area)

    Zf = Z.ravel()
    idx = np.argsort(Zf)[::-1]
    csum = np.cumsum(Zf[idx]*cell_area)
    k = min(np.searchsorted(csum, percent/100.0), len(idx)-1)
    thr = Zf[idx][k]
    mask = Z >= thr

    Zm = np.where(mask, Z, 0)
    tot = Zm.sum()*cell_area
    if tot > 0: Zm /= tot

    # contours → polygons in UTM
    contours = measure.find_contours(mask.astype(float), 0.5)
    polys=[]
    for c in contours:
        px, py = c[:,1], c[:,0]
        xs = np.interp(px, np.arange(grid_size), gx)
        ys = np.interp(py, np.arange(grid_size), gy)
        p = Polygon(zip(xs, ys)).buffer(0)
        if p.is_valid and p.area>0: polys.append(p)
    if not polys: return None, None, None, None, None

    mp_utm = unary_union(polys)
    # back to lat/lon
    def utm_to_ll(poly):
        if poly.is_empty: return None
        if isinstance(poly, Polygon):
            elon, elat = to_ll.transform(*poly.exterior.xy)
            holes = [to_ll.transform(*ring.xy) for ring in poly.interiors]
            return Polygon(list(zip(elon, elat)),
                           [list(zip(hlon, hlat)) for hlon, hlat in holes])
        if isinstance(poly, MultiPolygon):
            return MultiPolygon([utm_to_ll(p) for p in poly.geoms if not p.is_empty])
        return None

    mp_ll = utm_to_ll(mp_utm)
    area_km2 = mp_utm.area / 1e6

    return mp_ll, area_km2, Zm, (xmin, ymin, xmax, ymax), to_ll

def add_kdes(df, percent_list):
    os.makedirs("outputs", exist_ok=True)
    for percent in percent_list:
        for animal in df["animal_id"].unique():
            storage.kde_results.setdefault(animal, {})
            if percent in storage.kde_results[animal]:
                continue

            trk = df[df["animal_id"]==animal]
            poly_ll, area_km2, Zm, bbox, to_ll = _kde_core(trk['latitude'].values, trk['longitude'].values, percent)

            if poly_ll is None: continue

            xmin, ymin, xmax, ymax = bbox
            lon_sw, lat_sw = to_ll.transform(xmin, ymin)
            lon_ne, lat_ne = to_ll.transform(xmax, ymax)

            safe = str(animal).replace(" ","_").replace("/","_")
            tif = os.path.join("outputs", f"kde_{safe}_{percent}.tif")
            with rasterio.open(
                tif, "w", driver="GTiff",
                height=Zm.shape[0], width=Zm.shape[1], count=1, dtype=Zm.dtype,
                crs="EPSG:4326",
                transform=from_origin(lon_sw, lat_ne, (lon_ne-lon_sw)/Zm.shape[1], (lat_ne-lat_sw)/Zm.shape[0])
            ) as dst:
                dst.write(np.flipud(Zm), 1)

            gj = os.path.join("outputs", f"kde_{safe}_{percent}.geojson")
            with open(gj, "w") as f:
                json.dump(mapping(poly_ll), f)

            storage.kde_results[animal][percent] = {
                "contour": poly_ll, "area": area_km2,
                "geotiff": tif, "geojson": gj
            }
