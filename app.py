import subprocess
import os

def r_package_installed(pkg):
    try:
        subprocess.run(
            ['Rscript', '-e', f"stopifnot(requireNamespace('{pkg}', quietly=TRUE))"],
            check=True,
            capture_output=True
        )
        return True
    except Exception:
        return False

if not r_package_installed('adehabitatHR'):
    print("Running install.R to install missing R packages...")
    subprocess.run(["Rscript", "install.R"], check=True)

import os
import re
import json
import gradio as gr
import pandas as pd
import folium
import shutil
import random
import subprocess
import glob
from pyproj import CRS, Transformer
from together import Together
from dotenv import load_dotenv

import shutil
print("Rscript path at startup:", shutil.which("Rscript"))


# ========== LLM SETUP (Together API) ==========

load_dotenv()
client = Together(api_key=os.getenv("TOGETHER_API_KEY"))

SYSTEM_PROMPT = """
You are SpatChat, an expert wildlife home range analysis assistant.
If the user asks for a home range calculation (MCP, KDE, dBBMM, AKDE, etc.), reply ONLY in JSON using this format:
{"tool": "home_range", "method": "mcp", "level": 95}
- method: one of "mcp", "kde", "akde", "bbmm", "dbbmm"
- level: percentage for the home range (default 95 if user doesn't specify)
- Optionally, include animal_id if the user specifies a particular animal.
For any other questions, answer as an expert movement ecologist in plain text (keep to 2-3 sentences).
"""
FALLBACK_PROMPT = """
You are SpatChat, a wildlife movement expert.
If you can't map a request to a home range tool, just answer naturally.
Keep replies under three sentences.
"""

def ask_llm(chat_history, user_input):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in chat_history:
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append({"role": "user", "content": user_input})
    resp = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=messages,
        temperature=0.0
    ).choices[0].message.content
    try:
        call = json.loads(resp)
        return call, resp
    except Exception:
        # fallback: try conversational response
        conv = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            messages=[{"role": "system", "content": FALLBACK_PROMPT}] + messages,
            temperature=0.7
        ).choices[0].message.content
        return None, conv

# ========== App Logic ==========

cached_df = None
cached_headers = []

def parse_crs_input(crs_input):
    crs_input = str(crs_input).strip().upper()
    if crs_input.startswith("EPSG:"):
        return int(crs_input.split(":")[1])
    if crs_input.isdigit():
        return 32600 + int(crs_input)
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

def handle_upload_initial(file):
    global cached_df, cached_headers
    os.makedirs("uploads", exist_ok=True)
    filename = os.path.join("uploads", os.path.basename(file))
    shutil.copy(file, filename)

    try:
        df = pd.read_csv(filename)
        cached_df = df
        cached_headers = list(df.columns)
    except Exception as e:
        return [], gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), render_empty_map(), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

    lower_cols = [col.lower() for col in df.columns]
    if "latitude" in lower_cols and "longitude" in lower_cols:
        return [
            {"role": "assistant", "content": "CSV uploaded successfully. Latitude and longitude detected. You may now proceed to create home ranges."}
        ], gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), handle_upload_confirm("longitude", "latitude", ""), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)

    preferred_x = next((col for col in df.columns if col.lower() in ["x", "easting"]), df.columns[0])
    preferred_y = next((col for col in df.columns if col.lower() in ["y", "northing"]), df.columns[1] if len(df.columns) > 1 else df.columns[0])

    return [
        {"role": "assistant", "content": "CSV uploaded. Coordinate columns not clearly labeled. Please confirm X/Y columns and provide a CRS if needed. Be sure to click the Confirm button after filling these fields."}
    ], \
    gr.update(choices=cached_headers, value=preferred_x, visible=True), \
    gr.update(choices=cached_headers, value=preferred_y, visible=True), \
    gr.update(visible=True), \
    render_empty_map(), \
    gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)

def handle_upload_confirm(x_col, y_col, crs_input):
    global cached_df
    df = cached_df.copy()
    if x_col not in df.columns or y_col not in df.columns:
        return "<p>Selected coordinate columns not found in data.</p>"

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

def handle_chat(chat_history, user_message):
    global cached_df
    import shutil
    rscript_path = shutil.which("Rscript")
    print("Rscript path before call:", rscript_path)
    tool, llm_output = ask_llm(chat_history, user_message)
    if tool and tool.get("tool") == "home_range":
        method = tool.get("method")
        percent = tool.get("level", 95)
        input_csv = "uploads/latest.csv"
        cached_df.to_csv(input_csv, index=False)
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)
        for f in glob.glob(os.path.join(output_dir, f"hr_*_{method}.geojson")):
            os.remove(f)
        try:
            subprocess.run([
                rscript_path or "Rscript", "build_home_range_dbbmm.R", input_csv, output_dir, method, str(percent)
            ], check=True)
            geojsons = glob.glob(os.path.join(output_dir, f"hr_*_{method}.geojson"))
        except Exception as e:
            chat_history = chat_history + [{"role": "user", "content": user_message}]
            chat_history = chat_history + [{"role": "assistant", "content": f"Failed to calculate {method.upper()}: {e}"}]
            return chat_history, gr.update()
        if geojsons:
            m = folium.Map(location=[cached_df['latitude'].mean(), cached_df['longitude'].mean()], zoom_start=6)
            folium.TileLayer("OpenStreetMap").add_to(m)
            for gj in geojsons:
                folium.GeoJson(gj, name=os.path.basename(gj), style_function=lambda x: {
                    "color": "#FF0000", "weight": 2, "fillOpacity": 0.2
                }).add_to(m)
            folium.LayerControl(collapsed=False).add_to(m)
            map_html = m._repr_html_()
            chat_history = chat_history + [{"role": "user", "content": user_message}]
            chat_history = chat_history + [{"role": "assistant", "content": f"{method.upper()} {percent}% home range calculated and displayed on the map."}]
            return chat_history, gr.update(value=map_html)
        else:
            chat_history = chat_history + [{"role": "user", "content": user_message}]
            chat_history = chat_history + [{"role": "assistant", "content": f"No {method.upper()} output produced by the R script."}]
            return chat_history, gr.update()
    else:
        chat_history = chat_history + [{"role": "user", "content": user_message}]
        chat_history = chat_history + [{"role": "assistant", "content": llm_output.strip()}]
        return chat_history, gr.update()

with gr.Blocks() as demo:
    gr.Markdown("## SpatChat: Home Range - Movement Preview")
    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="SpatChat",
                show_label=True,
                type="messages",
                value=[{"role": "assistant", "content": "Welcome to SpatChat! Please upload a CSV containing coordinates (lat/lon or UTM) and optional timestamp/animal_id to begin."}]
            )
            user_input = gr.Textbox(label=None, placeholder="Ask SpatChat...", lines=1)
            file_input = gr.File(label="Upload Movement CSV")
            x_col = gr.Dropdown(label="X column", choices=[], visible=False)
            y_col = gr.Dropdown(label="Y column", choices=[], visible=False)
            crs_input = gr.Text(label="CRS (e.g. '32633', '33N', or 'EPSG:32633')", visible=False)
            confirm_btn = gr.Button("Confirm Coordinate Settings", visible=False)
        with gr.Column(scale=3):
            map_output = gr.HTML(label="Map Preview", value=render_empty_map(), show_label=False)

    file_input.change(
        fn=handle_upload_initial,
        inputs=file_input,
        outputs=[
            chatbot, x_col, y_col, crs_input, map_output,
            x_col, y_col, crs_input, confirm_btn
        ]
    )
    confirm_btn.click(
        fn=handle_upload_confirm,
        inputs=[x_col, y_col, crs_input],
        outputs=map_output
    )
    user_input.submit(
        fn=handle_chat,
        inputs=[chatbot, user_input],
        outputs=[chatbot, map_output]
    )

demo.launch()
