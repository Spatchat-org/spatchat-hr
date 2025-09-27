# map_layers.py
import os
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

# ---------- New small builders ----------

def make_points_tracks_layers(df, color_map):
    """Return (points_layer, tracks_layer) FeatureGroups."""
    points_layer = folium.FeatureGroup(name="Points", show=True)
    tracks_layer = folium.FeatureGroup(name="Tracks", show=True)
    has_timestamp = "timestamp" in df.columns
    animal_ids = df["animal_id"].unique() if "animal_id" in df.columns else ["sample"]

    for animal in animal_ids:
        track = df[df["animal_id"] == animal] if "animal_id" in df.columns else df
        color = color_map[animal]
        if has_timestamp:
            track = track.sort_values("timestamp")
            coords = list(zip(track["latitude"], track["longitude"]))
            if len(coords) > 1:
                folium.PolyLine(coords, color=color, weight=2.5, opacity=0.8, popup=f"{animal} Track").add_to(tracks_layer)
        for _, row in track.iterrows():
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=3,
                color=color,
                fill=True,
                fill_opacity=0.7,
                popup=f"{animal}"
            ).add_to(points_layer)
    return points_layer, tracks_layer

def make_mcp_layers(mcp_results, requested_percents, animal_ids, color_map):
    """Return a list of FeatureGroups for MCP polygons."""
    layers = []
    for percent in requested_percents:
        for animal in animal_ids:
            if animal in mcp_results and percent in mcp_results[animal]:
                v = mcp_results[animal][percent]
                layer = folium.FeatureGroup(name=f"{animal} MCP {percent}%", show=True)
                # v["polygon"] is shapely poly with (lon,lat). Folium expects (lat,lon)
                layer.add_child(
                    folium.Polygon(
                        locations=[(lat, lon) for lon, lat in np.array(v["polygon"].exterior.coords)],
                        color=color_map[animal],
                        fill=True,
                        fill_opacity=0.15 + 0.15 * (percent / 100),
                        popup=f"{animal} MCP {percent}%"
                    )
                )
                layers.append(layer)
    return layers

def make_kde_layers(kde_results, requested_kde_percents, animal_ids, color_map):
    """Return a list of FeatureGroups for KDE raster + contour layers."""
    layers = []
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
            layers.append(raster_layer)

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
            layers.append(contour_layer)
    return layers

def make_locoh_layers(locoh_result: dict, animal_ids, color_map, name_prefix: str = "LoCoH"):
    """
    Envelopes at requested isopleths (e.g., 50/95). Visible by default.
    """
    import folium

    layers = []
    animals_dict = locoh_result.get("animals", {}) if locoh_result else {}
    for animal in animal_ids:
        data = animals_dict.get(str(animal)) or animals_dict.get(animal)
        if not data:
            continue

        for item in data.get("isopleths", []):
            iso = int(item["isopleth"])
            gj_geom = item["geometry"]
            area_km2 = float(item.get("area_sq_km", 0.0))

            feature = {
                "type": "Feature",
                "properties": {
                    "animal_id": str(animal),
                    "isopleth": iso,
                    "area_km2": round(area_km2, 3),
                },
                "geometry": gj_geom,
            }

            layer = folium.FeatureGroup(name=f"{animal} {name_prefix} {iso}%", show=True)  # ← show by default
            folium.GeoJson(
                data=feature,
                name=f"{name_prefix} {iso}% — {animal}",
                style_function=lambda _feat, iso=iso, animal=animal: {
                    "fillOpacity": 0.45 if iso != 95 else 0.25,
                    "weight": 2,
                    "color": color_map[animal],
                },
                tooltip=folium.GeoJsonTooltip(
                    fields=["animal_id", "isopleth", "area_km2"],
                    aliases=["Animal", "Isopleth (%)", "Area (km²)"],
                    localize=True,
                ),
            ).add_to(layer)
            layers.append(layer)

    return layers

def make_locoh_facets_layers(locoh_result: dict, animal_ids, color_map, name_prefix: str = "LoCoH facets"):
    """
    Render individual local convex hulls ("facets") colored by cumulative percent (red→yellow).
    Visible by default.
    """
    import folium

    # bins & palette (deep red → pale yellow)
    bins = [20, 40, 60, 80, 95, 100]
    palette = {
        20: "#d7301f",
        40: "#ef6548",
        60: "#fc8d59",
        80: "#fdbb84",
        95: "#fdd49e",
        100:"#fee8c8",
    }
    def color_for(pct: int) -> str:
        for b in bins:
            if pct <= b:
                return palette[b]
        return palette[100]

    layers = []
    animals_dict = locoh_result.get("animals", {}) if locoh_result else {}
    for animal in animal_ids:
        data = animals_dict.get(str(animal)) or animals_dict.get(animal)
        if not data:
            continue
        facets = data.get("facets", [])
        if not facets:
            continue

        # One GeoJSON per animal for performance
        features = []
        for f in facets:
            pct = int(f.get("cum_percent", 100))
            area = float(f.get("area_sq_km", 0.0))
            features.append({
                "type": "Feature",
                "properties": {
                    "animal_id": str(animal),
                    "cum_percent": pct,
                    "area_km2": round(area, 3),
                    "_fill": color_for(pct),
                },
                "geometry": f["geometry"],
            })

        fc = {"type": "FeatureCollection", "features": features}
        layer = folium.FeatureGroup(name=f"{animal} {name_prefix}", show=True)  # ← show by default
        folium.GeoJson(
            data=fc,
            name=f"{name_prefix} — {animal}",
            style_function=lambda feat: {
                "fillOpacity": 0.6,
                "weight": 1,
                "color": "#222222",
                "fillColor": feat["properties"]["_fill"],
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["animal_id", "cum_percent", "area_km2"],
                aliases=["Animal", "Cum. %", "Hull area (km²)"],
                localize=True,
            ),
        ).add_to(layer)

        layers.append(layer)

    return layers

def make_dbbmm_layers(dbbmm_results: dict, animal_ids, color_map, name_prefix="dBBMM"):
    """
    Build per-animal raster + isopleth contours from dBBMM results.
    Accepts per-animal entries as either dicts or DBBMMResult dataclasses.
    """
    import os
    import folium
    import numpy as np
    import rasterio

    layers = []

    for animal in animal_ids:
        data = dbbmm_results.get(str(animal)) or dbbmm_results.get(animal)
        if not data:
            continue

        # Support both dict and dataclass
        if isinstance(data, dict):
            tif_path = data.get("geotiff")
            iso_list = data.get("isopleths", []) or []
        else:
            tif_path = getattr(data, "geotiff", None)
            iso_list = getattr(data, "isopleths", []) or []

        color = color_map[animal]

        # 1) Raster UD layer (shown by default)
        if tif_path and os.path.exists(tif_path):
            raster_layer = folium.FeatureGroup(name=f"{animal} {name_prefix} Raster", show=True)
            with rasterio.open(tif_path) as src:
                arr = src.read(1)
                if np.isfinite(arr).any() and float(np.nanmax(arr)) > 0:
                    arr = np.nan_to_num(arr, nan=0.0)
                    arr_norm = (arr - arr.min()) / (arr.max() - arr.min() + 1e-12)
                    import matplotlib.pyplot as plt
                    cmap = plt.get_cmap("viridis")
                    rgba = (cmap(arr_norm) * 255).astype(np.uint8)
                    bounds = src.bounds
                    img = np.dstack([
                        rgba[:, :, 0], rgba[:, :, 1], rgba[:, :, 2], (rgba[:, :, 3] * 0.70).astype(np.uint8)
                    ])
                    raster_layer.add_child(
                        folium.raster_layers.ImageOverlay(
                            image=img,
                            bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
                            opacity=0.7,
                            interactive=False
                        )
                    )
            layers.append(raster_layer)

        # 2) Isopleth polygons (shown by default)
        for item in iso_list:
            percent = int(item.get("percent"))
            geom = item.get("geometry")
            area_km2 = float(item.get("area_sq_km", 0.0))
            layer = folium.FeatureGroup(name=f"{animal} {name_prefix} {percent}%", show=True)
            if geom:
                folium.GeoJson(
                    data={
                        "type": "Feature",
                        "properties": {"animal_id": str(animal), "percent": percent, "area_km2": round(area_km2, 3)},
                        "geometry": geom,
                    },
                    style_function=lambda _feat, color=color: {
                        "fillOpacity": 0.25,
                        "weight": 2,
                        "color": color,
                    },
                    tooltip=folium.GeoJsonTooltip(
                        fields=["animal_id", "percent", "area_km2"],
                        aliases=["Animal", "Isopleth (%)", "Area (km²)"],
                        localize=True,
                    ),
                ).add_to(layer)
            layers.append(layer)

    return layers

# ---------- Composition entrypoint ----------

def build_results_map(
    df,
    mcp_results,
    kde_results,
    requested_percents,
    requested_kde_percents,
    locoh_result=None,
    dbbmm_result=None
):
    """Full map: points/tracks + estimator-specific layers (MCP, KDE, LoCoH)."""
    m = _base_map(df["latitude"].mean(), df["longitude"].mean(), control_scale=False, zoom=9)

    # Consistent colors per animal across all estimators
    animal_ids = df["animal_id"].unique() if "animal_id" in df.columns else ["sample"]
    color_map = {aid: f"#{random.randint(0, 0xFFFFFF):06x}" for aid in animal_ids}

    # Points/Tracks
    points_layer, tracks_layer = make_points_tracks_layers(df, color_map)
    points_layer.add_to(m)
    tracks_layer.add_to(m)

    # MCP
    for layer in make_mcp_layers(mcp_results or {}, requested_percents or [], animal_ids, color_map):
        layer.add_to(m)

    # KDE
    for layer in make_kde_layers(kde_results or {}, requested_kde_percents or [], animal_ids, color_map):
        layer.add_to(m)

    # LoCoH
    if locoh_result is not None:
        # Envelopes (50/95 or custom)
        for layer in make_locoh_layers(locoh_result, animal_ids, color_map, name_prefix="LoCoH"):
            layer.add_to(m)
        # Facets (tiny hulls, red→yellow)
        for layer in make_locoh_facets_layers(locoh_result, animal_ids, color_map, name_prefix="LoCoH facets"):
            layer.add_to(m)

    # dBBMM
    if dbbmm_result is not None:
        for layer in make_dbbmm_layers(dbbmm_result, animal_ids, color_map, name_prefix="dBBMM"):
            layer.add_to(m)
    
    folium.LayerControl(collapsed=False).add_to(m)
    m = fit_map_to_bounds(m, df)
    return m._repr_html_()

