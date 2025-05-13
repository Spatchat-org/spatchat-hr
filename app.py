# app.py — SpatChat-style with folium, filters, color-coded animals, and basemaps
import gradio as gr
import pandas as pd
import folium
import os
import shutil
import random

last_df = None  # to persist uploaded data across interactions

def handle_upload(file):
    global last_df
    os.makedirs("uploads", exist_ok=True)
    filename = os.path.join("uploads", os.path.basename(file))
    shutil.copy(file, filename)

    try:
        df = pd.read_csv(filename)
    except Exception as e:
        return f"Error reading CSV: {e}", gr.update(choices=[]), None, None

    required_cols = {"latitude", "longitude", "timestamp", "animal_id"}
    if not required_cols.issubset(df.columns):
        return f"CSV must contain columns: {', '.join(required_cols)}", gr.update(choices=[]), None, None

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    last_df = df
    animals = sorted(df["animal_id"].unique().tolist())
    tmin, tmax = df["timestamp"].min(), df["timestamp"].max()
    return render_map(df), gr.update(choices=["All"] + animals, value="All"), str(tmin), str(tmax)

def render_map(df, selected_animal="All", time_range=None):
    if df is None or df.empty:
        return "<p>No data loaded yet.</p>"

    if selected_animal != "All":
        df = df[df["animal_id"] == selected_animal]
    if time_range:
        df = df[(df["timestamp"] >= pd.to_datetime(time_range[0])) & (df["timestamp"] <= pd.to_datetime(time_range[1]))]

    if df.empty:
        return "<p>No points match current filters.</p>"

    center = [df["latitude"].mean(), df["longitude"].mean()]
    m = folium.Map(location=center, zoom_start=6, control_scale=True)

    folium.TileLayer("OpenStreetMap").add_to(m)
    folium.TileLayer("Stamen Terrain").add_to(m)
    folium.TileLayer("Stamen Toner").add_to(m)
    folium.TileLayer("CartoDB positron").add_to(m)
    folium.TileLayer("CartoDB dark_matter").add_to(m)

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

def update_filters(selected_animal, tmin, tmax):
    global last_df
    return render_map(last_df, selected_animal, (tmin, tmax))

with gr.Blocks() as demo:
    gr.Markdown("## SpatChat: Home Range - Movement Preview")
    with gr.Row():
        with gr.Column(scale=3):
            map_output = gr.HTML(label="Map Preview", value="<p>Waiting for movement data...</p>", show_label=False)
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="SpatChat", show_label=True, type="messages")
            file_input = gr.File(label="Upload Movement CSV")
            animal_selector = gr.Dropdown(choices=[], label="Filter by Animal")
            tmin = gr.Text(label="Start Time")
            tmax = gr.Text(label="End Time")

            file_input.change(fn=handle_upload, inputs=file_input, outputs=[map_output, animal_selector, tmin, tmax])
            animal_selector.change(fn=update_filters, inputs=[animal_selector, tmin, tmax], outputs=map_output)
            tmin.change(fn=update_filters, inputs=[animal_selector, tmin, tmax], outputs=map_output)
            tmax.change(fn=update_filters, inputs=[animal_selector, tmin, tmax], outputs=map_output)

demo.launch()
