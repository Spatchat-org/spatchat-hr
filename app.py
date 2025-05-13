# app.py — folium-powered movement map preview
import gradio as gr
import pandas as pd
import folium
import os

def handle_upload(file):
    os.makedirs("uploads", exist_ok=True)
    filename = os.path.join("uploads", os.path.basename(file))
    with open(filename, "wb") as f:
        f.write(file.read())

    try:
        df = pd.read_csv(filename)
    except Exception as e:
        return f"Error reading CSV: {e}"

    required_cols = {"latitude", "longitude", "timestamp", "animal_id"}
    if not required_cols.issubset(df.columns):
        return f"CSV must contain columns: {', '.join(required_cols)}"

    # Center map on the mean location
    center = [df["latitude"].mean(), df["longitude"].mean()]
    m = folium.Map(location=center, zoom_start=6)

    # Plot points and lines per animal
    for animal in df["animal_id"].unique():
        track = df[df["animal_id"] == animal].sort_values("timestamp")
        coords = list(zip(track["latitude"], track["longitude"]))

        folium.PolyLine(coords, color="blue", weight=2.5, opacity=0.8, popup=animal).add_to(m)

        for i, row in track.iterrows():
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=3,
                popup=f"{animal}<br>{row['timestamp']}",
                color="black",
                fill=True,
                fill_opacity=0.7
            ).add_to(m)

    return m._repr_html_()

with gr.Blocks() as demo:
    gr.Markdown("## SpatChat: Interactive Movement Map")

    with gr.Row():
        file_input = gr.File(label="Upload Movement CSV")
    map_output = gr.HTML(label="Map Preview")

    file_input.change(fn=handle_upload, inputs=file_input, outputs=map_output)

demo.launch()
