# app.py — SpatChat-style with folium preview map only (simplified)
import gradio as gr
import pandas as pd
import folium
import os
import shutil
import random

def handle_upload(file):
    os.makedirs("uploads", exist_ok=True)
    filename = os.path.join("uploads", os.path.basename(file))
    shutil.copy(file, filename)

    try:
        df = pd.read_csv(filename)
    except Exception as e:
        return f"<p>Error reading CSV: {e}</p>"

    required_cols = {"latitude", "longitude", "timestamp", "animal_id"}
    if not required_cols.issubset(df.columns):
        return f"<p>CSV must contain columns: {', '.join(required_cols)}</p>"

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    center = [df["latitude"].mean(), df["longitude"].mean()]
    m = folium.Map(location=center, zoom_start=6, control_scale=True)

    # Only use basemaps with proper attribution
    folium.TileLayer("OpenStreetMap").add_to(m)
    folium.TileLayer("CartoDB positron", attr='CartoDB').add_to(m)

    points_layer = folium.FeatureGroup(name="Points", show=True)
    lines_layer = folium.FeatureGroup(name="Tracks", show=True)

    animal_ids = df["animal_id"].unique()
    color_map = {aid: f"#{random.randint(0, 0xFFFFFF):06x}" for aid in animal_ids}

    for animal in animal_ids:
        track = df[df["animal_id"] == animal].sort_values("timestamp")
        coords = list(zip(track["latitude"], track["longitude"]))
        color = color_map[animal]

        folium.PolyLine(coords, color=color, weight=2.5, opacity=0.8, popup=animal).add_to(lines_layer)

        for _, row in track.iterrows():
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=3,
                popup=f"{animal}<br>{row['timestamp']}",
                color=color,
                fill=True,
                fill_opacity=0.7
            ).add_to(points_layer)

    points_layer.add_to(m)
    lines_layer.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    return m._repr_html_()

with gr.Blocks() as demo:
    gr.Markdown("## SpatChat: Home Range - Movement Preview")
    with gr.Row():
        with gr.Column(scale=3):
            map_output = gr.HTML(label="Map Preview", value="<p>Waiting for movement data...</p>", show_label=False)
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="SpatChat", show_label=True, type="messages")
            file_input = gr.File(label="Upload Movement CSV")
            file_input.change(fn=handle_upload, inputs=file_input, outputs=map_output)

demo.launch()