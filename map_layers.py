# map_layers.py
import random
import numpy as np
import folium
import rasterio
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, MultiPolygon

from map_utils import fit_map_to_bounds

def _base_map(center_lat, center_lon, control_scale=True, zoom=9):
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom, control_scale=control_scale)
    folium.TileLayer("OpenStreetMap").add_to(m)
    folium.TileLayer("CartoDB positron", attr='CartoDB').add_to(m)
    folium.TileLayer(
        tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        attr="OpenTopoMap", name="Topographic"
    ).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri", name="Satellite"
    ).add_to(m)
    return m

def build_preview_map(df):
    """Points + (optional) tracks, layer control, zoom to data."""
    has_timestamp = "timestamp" in df.columns
    m = _base_map(df["latitude"].mean(), df["longitude"].mean(), control_scale=True, zoom=9)

    points_layer = folium.FeatureGroup(name="Points", show=True)
    lines_layer  = folium.FeatureGroup(name="Tracks", show=True)

    # handle presence/absence of animal_id
    animal_ids = df["animal_id"].unique() if "animal_id" in df.columns else ["sample"]
    color_map = {aid: f"#{random.randint(0, 0xFFFFFF):06x}" for aid in animal_ids}

    for animal in animal_ids:
        track = df[df["animal_id"] == animal] if "animal_id" in df.columns else df
        if has_timestamp:
            track = track.sort_values("timestamp")
            coords = list(zip(track["latitude"], track["longitude"]))
            if len(coords) > 1:
                folium.PolyLine(coords, color=color_map[animal], weight=2.5, opacity=0.8, popup=str(animal)).add_to(lines_layer)

        for _, row in track.iterrows():
            label = f"{animal}" + (f"<br>{row['timestamp']}" if has_timestamp and row.get('timestamp') is not None else "")
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=3,
                popup=label,
                color=color_map[animal],
                fill=True,
                fill_opacity=0.7
            ).add_to(points_layer)

    points_layer.add_to(m)
    if has_timestamp:
        lines_layer.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m = fit_map_to_bounds(m, df)
    return m._repr_html_()

def make_locoh_layers(locoh_result: dict, name_prefix: str = "LoCoH"):
    """Return a list of (layer_name, folium.GeoJson) for 50/95 isopleths per animal."""
    import folium

    layers = []
    for animal_id, data in locoh_result.get("animals", {}).items():
        for item in data.get("isopleths", []):
            iso = item["isopleth"]
            gj = item["geometry"]
            layer_name = f"{name_prefix} {iso}% — {animal_id}"
            g = folium.GeoJson(
                gj,
                name=layer_name,
                tooltip=folium.GeoJsonTooltip(fields=[], aliases=[], labels=False),
                style_function=lambda _feat, iso=iso: {
                    "fillOpacity": 0.25 if iso == 95 else 0.45,
                    "weight": 2,
                },
                show=(iso in (50,)),
            )
            layers.append((layer_name, g))
    return layers

def build_results_map(df, mcp_results, kde_results, requested_percents, requested_kde_percents):
    """Full map: points/tracks + MCP polygons + KDE raster/contours."""
    m = _base_map(df["latitude"].mean(), df["longitude"].mean(), control_scale=False, zoom=9)

    # points/tracks
    points_layer = folium.FeatureGroup(name="Points", show=True)
    paths_layer  = folium.FeatureGroup(name="Tracks", show=True)
    has_timestamp = "timestamp" in df.columns
    animal_ids = df["animal_id"].unique() if "animal_id" in df.columns else ["sample"]
    color_map = {aid: f"#{random.randint(0, 0xFFFFFF):06x}" for aid in animal_ids}

    for animal in animal_ids:
        track = df[df["animal_id"] == animal] if "animal_id" in df.columns else df
        color = color_map[animal]
        if has_timestamp:
            track = track.sort_values("timestamp")
            coords = list(zip(track["latitude"], track["longitude"]))
            if len(coords) > 1:
                folium.PolyLine(coords, color=color, weight=2.5, opacity=0.8, popup=f"{animal} Track").add_to(paths_layer)
        for _, row in track.iterrows():
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=3,
                color=color,
                fill=True,
                fill_opacity=0.7,
                popup=f"{animal}"
            ).add_to(points_layer)

    points_layer.add_to(m)
    paths_layer.add_to(m)

    # MCP layers
    for percent in requested_percents:
        for animal in animal_ids:
            if animal in mcp_results and percent in mcp_results[animal]:
                v = mcp_results[animal][percent]
                layer = folium.FeatureGroup(name=f"{animal} MCP {percent}%", show=True)
                layer.add_child(
                    folium.Polygon(
                        locations=[(lat, lon) for lon, lat in np.array(v["polygon"].exterior.coords)],
                        color=color_map[animal],
                        fill=True,
                        fill_opacity=0.15 + 0.15 * (percent / 100),
                        popup=f"{animal} MCP {percent}%"
                    )
                )
                m.add_child(layer)

    # KDE raster + contours
    for animal in animal_ids:
        kde_percs = [p for p in requested_kde_percents if animal in kde_results and p in kde_results[animal]]

        # Raster: only for the highest % requested per animal
        if kde_percs:
            max_perc = max(kde_percs)
            v = kde_results[animal][max_perc]
            raster_layer = folium.FeatureGroup(name=f"{animal} KDE Raster", show=True)
            with rasterio.open(v["geotiff"]) as src:
                arr = src.read(1)
                arr_norm = (arr - arr.min()) / (arr.max() - arr.min() + 1e-10)
                cmap = plt.get_cmap('plasma')
                rgba = (cmap(arr_norm) * 255).astype(np.uint8)
                bounds = src.bounds
                img = np.dstack([
                    rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], (rgba[:, :, 3]*0.7).astype(np.uint8)
                ])
                raster_layer.add_child(
                    folium.raster_layers.ImageOverlay(
                        image=img,
                        bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
                        opacity=0.7,
                        interactive=False
                    )
                )
            m.add_child(raster_layer)

        # Contours: all requested %
        for percent in kde_percs:
            v = kde_results[animal][percent]
            contour_layer = folium.FeatureGroup(name=f"{animal} KDE {percent}% Contour", show=True)
            contour = v["contour"]
            if contour:
                if isinstance(contour, MultiPolygon):
                    for poly in contour.geoms:
                        contour_layer.add_child(
                            folium.Polygon(
                                locations=[(lat, lon) for lon, lat in poly.exterior.coords],
                                color=color_map[animal],
                                fill=True,
                                fill_opacity=0.2,
                                popup=f"{animal} KDE {percent}% Contour"
                            )
                        )
                elif isinstance(contour, Polygon):
                    contour_layer.add_child(
                        folium.Polygon(
                            locations=[(lat, lon) for lon, lat in contour.exterior.coords],
                            color=color_map[animal],
                            fill=True,
                            fill_opacity=0.2,
                            popup=f"{animal} KDE {percent}% Contour"
                        )
                    )
            m.add_child(contour_layer)

    folium.LayerControl(collapsed=False).add_to(m)
    m = fit_map_to_bounds(m, df)
    return m._repr_html_()
