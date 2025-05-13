# app.py — Flexible coordinate upload with UTM zone or EPSG support
import gradio as gr
import pandas as pd
import folium
import os
import shutil
import random
from pyproj import CRS, Transformer

def parse_crs_input(crs_input):
    crs_input = str(crs_input).strip().upper()
    if crs_input.startswith("EPSG:"):
        return int(crs_input.split(":")[1])
    if crs_input.isdigit():
        return 32600 + int(crs_input)  # default to northern
    if len(crs_input) >= 2 and crs_input[:-1].isdigit() and crs_input[-1] in ("N", "S"):
        zone = int(crs_input[:-1])
        return 32600 + zone if crs_input[-1] == "N" else 32700 + zone
    raise ValueError("Invalid CRS input. Use EPSG code or UTM zone like '33N'.")

def render_empty_map():
    m = folium.Map(location=[0, 0], zoom_start=2, control_scale=True)
    folium.TileLayer("OpenStreetMap").add_to(m)
    folium.TileLayer("CartoDB positron", attr='CartoDB').add_to(m)
    folium.TileLayer(
        tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        attr="OpenTopoMap",
        name="Topographic"
    ).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Satellite"
    ).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    return m._repr_html_()

def handle_upload(file, x_col, y_col, crs_input):
    os.makedirs("uploads", exist_ok=True)
    filename = os.path.join("uploads", os.path.basename(file))
    shutil.copy(file, filename)

    try:
        df = pd.read_csv(filename)
    except Exception as e:
        return f"<p>Error reading CSV: {e}</p>"

    if x_col not in df.columns or y_col not in df.columns:
        return "<p>Selected coordinate columns not found in data.</p>"

    # If data already in lat/lon use directly, else transform
    if x_col.lower() in ["longitude", "lon"] and y_col.lower() in ["latitude", "lat"]:
        df['longitude'] = df[x_col]
        df['latitude'] = df[y_col]
    else:
        try:
            epsg = parse_crs_input(crs_input)
            transformer = Transformer.from_crs(CRS.from_epsg(epsg), CRS.from_epsg(4326), always_xy=True)
            df['longitude'], df['latitude'] = transformer.transform(df[x_col].values, df[y_col].values)
        except Exception as e:
            return f"<p>Failed to convert coordinates: {e}</p>"

    has_timestamp = "timestamp" in df.columns
    has_animal_id = "animal_id" in df.columns
    if has_timestamp:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    if not has_animal_id:
        df["animal_id"] = "sample"

    center = [df["latitude"].mean(), df["longitude"].mean()]
    m = folium.Map(location=center, zoom_start=6, control_scale=True)

    folium.TileLayer("OpenStreetMap").add_to(m)
    folium.TileLayer("CartoDB positron", attr='CartoDB').add_to(m)
    folium.TileLayer(
        tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        attr="OpenTopoMap",
        name="Topographic"
    ).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        attr="Esri",
        name="Satellite"
    ).add_to(m)

    points_layer = folium.FeatureGroup(name="Points", show=True)
    lines_layer = folium.FeatureGroup(name="Tracks", show=True)

    animal_ids = df["animal_id"].unique()
    color_map = {aid: f"#{random.randint(0, 0xFFFFFF):06x}" for aid in animal_ids}

    for animal in animal_ids:
        track = df[df["animal_id"] == animal]
        coords = list(zip(track["latitude"], track["longitude"]))
        color = color_map[animal]

        if has_timestamp:
            track = track.sort_values("timestamp")
            folium.PolyLine(coords, color=color, weight=2.5, opacity=0.8, popup=animal).add_to(lines_layer)

        for _, row in track.iterrows():
            label = f"{animal}" + (f"<br>{row['timestamp']}" if has_timestamp else "")
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=3,
                popup=label,
                color=color,
                fill=True,
                fill_opacity=0.7
            ).add_to(points_layer)

    points_layer.add_to(m)
    if has_timestamp:
        lines_layer.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    return m._repr_html_()

with gr.Blocks() as demo:
    gr.Markdown("## SpatChat: Home Range - Movement Preview")
    with gr.Row():
        with gr.Column(scale=3):
            map_output = gr.HTML(label="Map Preview", value=render_empty_map(), show_label=False)
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="SpatChat", show_label=True, type="messages")
            file_input = gr.File(label="Upload Movement CSV")
            x_col = gr.Text(label="X column (e.g. 'longitude' or 'easting')", value="longitude")
            y_col = gr.Text(label="Y column (e.g. 'latitude' or 'northing')", value="latitude")
            crs_input = gr.Text(label="CRS (e.g. '32633', '33N', or 'EPSG:32633')", value="")
            file_input.change(fn=handle_upload, inputs=[file_input, x_col, y_col, crs_input], outputs=map_output)

demo.launch()
