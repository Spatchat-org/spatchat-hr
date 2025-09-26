import os
import re
import sys
import json
import time
import shutil
import random
import zipfile
from typing import List

import gradio as gr
import pandas as pd
import numpy as np
import folium
import rasterio
from shapely.geometry import Polygon, MultiPolygon, mapping
import matplotlib.pyplot as plt

# ---- Local modules ----
import storage
from storage import (
    get_cached_df, set_cached_df,
    get_cached_headers, set_cached_headers,
    clear_all_results,
    mcp_results, kde_results,
    requested_percents, requested_kde_percents,
)
from llm_utils import ask_llm
from crs_utils import parse_crs_input
from map_utils import render_empty_map, fit_map_to_bounds
from estimators.mcp import add_mcps
from estimators.kde import add_kdes
from schema_detect import (
    detect_and_standardize,
    parse_metadata_command,
    try_apply_user_mapping,
    ID_COL, TS_COL
)

print("Starting SpatChat: Home Range Analysis (app.py)")

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def looks_like_latlon(df, x_col, y_col):
    """Guess if two columns look like lon/lat or lat/lon numeric degrees."""
    try:
        x_vals = df[x_col].astype(float)
        y_vals = df[y_col].astype(float)
        if x_vals.between(-180, 180).all() and y_vals.between(-90, 90).all():
            return "lonlat"
        if x_vals.between(-90, 90).all() and y_vals.between(-180, 180).all():
            return "latlon"
    except Exception:
        return None
    return None

def looks_invalid_latlon(df, lat_col, lon_col):
    """Return True if labeled lat/lon columns are out of geographic bounds."""
    try:
        lat = df[lat_col].astype(float)
        lon = df[lon_col].astype(float)
        return not (lat.between(-90, 90).all() and lon.between(-180, 180).all())
    except Exception:
        return True

def parse_levels_from_text(text):
    levels = [int(val) for val in re.findall(r'\b([1-9][0-9]?|100)\b', text)]
    levels = [min(l, 99) for l in levels]
    if not levels:
        return [95]
    return sorted(set(levels))

def _render_points_tracks_map(df: pd.DataFrame) -> str:
    """
    Build the preview map with points; draw tracks if we have a usable timestamp column.
    (Uses the same basemap composition as your UI.)
    """
    has_timestamp = TS_COL in df.columns and df[TS_COL].notna().any()
    has_animal_id = ID_COL in df.columns

    m = folium.Map(
        location=[df["latitude"].mean(), df["longitude"].mean()],
        zoom_start=9,
        control_scale=True
    )
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

    points_layer = folium.FeatureGroup(name="Points", show=True)
    lines_layer  = folium.FeatureGroup(name="Tracks", show=True)

    # group by animal if available; else treat all as one animal
    if has_animal_id:
        animal_ids = df[ID_COL].astype(str).fillna("sample").unique()
    else:
        df = df.copy()
        df[ID_COL] = "sample"
        animal_ids = ["sample"]

    color_map = {aid: f"#{random.randint(0, 0xFFFFFF):06x}" for aid in animal_ids}

    for animal in animal_ids:
        track = df[df[ID_COL] == animal]
        color = color_map[animal]

        if has_timestamp:
            track = track.sort_values(TS_COL)

        coords = list(zip(track["latitude"], track["longitude"]))
        if has_timestamp and len(coords) > 1:
            folium.PolyLine(coords, color=color, weight=2.5, opacity=0.8, popup=f"{animal} Track").add_to(lines_layer)

        for _, row in track.iterrows():
            label = f"{animal}"
            if has_timestamp and pd.notna(row[TS_COL]):
                label += f"<br>{row[TS_COL]}"
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
    m = fit_map_to_bounds(m, df)
    return m._repr_html_()

def _zip_outputs():
    """Create outputs/spatchat_results.zip with MCP & KDE artifacts + areas CSV."""
    os.makedirs("outputs", exist_ok=True)
    features = []
    rows = []

    # MCP features
    if any(mcp_results.values()):
        for animal, percents in mcp_results.items():
            for percent, v in percents.items():
                features.append({
                    "type": "Feature",
                    "properties": {"animal_id": animal, "percent": percent, "area_km2": v["area"]},
                    "geometry": mapping(v["polygon"])
                })
                rows.append((animal, f"MCP-{percent}", v["area"]))
        geojson = {"type": "FeatureCollection", "features": features}
        with open(os.path.join("outputs", "mcps_all.geojson"), "w") as f:
            json.dump(geojson, f)

    # KDE areas
    for animal, percents in kde_results.items():
        for percent, v in percents.items():
            rows.append((animal, f"KDE-{percent}", v["area"]))

    if rows:
        df = pd.DataFrame(rows, columns=["animal_id", "type", "area_km2"])
        df.to_csv(os.path.join("outputs", "home_range_areas.csv"), index=False)

    archive = "outputs/spatchat_results.zip"
    if os.path.exists(archive):
        os.remove(archive)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk("outputs"):
            for file in files:
                if file.endswith(".zip"):
                    continue
                full_path = os.path.join(root, file)
                rel_path = os.path.relpath(full_path, "outputs")
                zipf.write(full_path, arcname=rel_path)
    print("ZIP written:", archive)
    return archive

# --------------------------------------------------------------------------------------
# Upload flow
# --------------------------------------------------------------------------------------
def handle_upload_initial(file):
    """
    1) Cache uploaded CSV
    2) Try to auto-detect lat/lon columns
    3) Auto-detect ID/timestamp (schema_detect)
    4) If ambiguous or projected, show pickers and CRS input
    Returns exactly the same UI outputs your UI expects (10 outputs).
    """
    clear_all_results()  # reset outputs/results

    os.makedirs("uploads", exist_ok=True)
    filename = os.path.join("uploads", os.path.basename(file))
    shutil.copy(file, filename)

    try:
        df = pd.read_csv(filename)
        set_cached_df(df)
        set_cached_headers(list(df.columns))
    except Exception as ex:
        # chatbot, x, y, crs, map, x, y, crs, confirm, download  (10)
        return ([],
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                render_empty_map(),
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                gr.update(visible=False), gr.update(visible=False))

    cached_headers = get_cached_headers()
    lower_cols = [c.lower() for c in cached_headers]

    # A) explicit latitude/longitude names
    if "latitude" in lower_cols and "longitude" in lower_cols:
        lat_col = cached_headers[lower_cols.index("latitude")]
        lon_col = cached_headers[lower_cols.index("longitude")]

        # If labeled as lat/lon but values look projected → ask for CRS
        raw_df = get_cached_df()
        if looks_invalid_latlon(raw_df, lat_col, lon_col):
            return ([
                        {"role": "assistant", "content":
                            "CSV uploaded. Your coordinates do not appear to be latitude/longitude. "
                            "Please specify X (easting), Y (northing), and the CRS/UTM zone below "
                            "(e.g., 'UTM 10T' or 'EPSG:32610')."}
                    ],
                    gr.update(choices=cached_headers, value=lon_col, visible=True),
                    gr.update(choices=cached_headers, value=lat_col, visible=True),
                    gr.update(visible=True),
                    render_empty_map(),
                    gr.update(visible=True), gr.update(visible=True), gr.update(visible=True),
                    gr.update(visible=True),  # confirm button
                    gr.update(visible=False))  # download button

        # Looks valid; standardize schema (ID/timestamp), render map, give examples
        df2, meta_msgs = detect_and_standardize(raw_df)
        set_cached_df(df2)

        chat_msgs = [{
            "role": "assistant",
            "content": (
                "CSV uploaded. Latitude and longitude detected. You may now create home ranges. "
                "For example: **“I want 100% MCP”**, **“I want 95 KDE”**, or **“MCP 95 50”**.\n"
                + ("\n".join(meta_msgs) if meta_msgs else "")
            )
        }]
        map_html = _render_points_tracks_map(get_cached_df())
        # (10 outputs)
        return (chat_msgs,
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                map_html,
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False))

    # B) try to guess lon/lat by ranges for common x/y/easting/northing labels
    df = get_cached_df()
    x_names = ["x", "easting", "lon", "longitude"]
    y_names = ["y", "northing", "lat", "latitude"]
    found_x = next((col for col in df.columns if col.lower() in x_names), df.columns[0])
    found_y = next((col for col in df.columns if col.lower() in y_names and col != found_x),
                   df.columns[1] if len(df.columns) > 1 else df.columns[0])
    if found_x == found_y and len(df.columns) > 1:
        found_y = df.columns[1 if df.columns[0] == found_x else 0]

    latlon_guess = looks_like_latlon(df, found_x, found_y)
    if latlon_guess:
        # standardize lon/lat columns then run schema detection
        df2 = df.copy()
        df2["longitude"] = df2[found_x] if latlon_guess == "lonlat" else df2[found_y]
        df2["latitude"]  = df2[found_y] if latlon_guess == "lonlat" else df2[found_x]
        df2, meta_msgs = detect_and_standardize(df2)
        set_cached_df(df2)

        chat_msgs = [{
            "role": "assistant",
            "content": (
                f"CSV uploaded. {found_x}/{found_y} interpreted as latitude/longitude.\n"
                "You may now create home ranges. For example: **“I want 100% MCP”**, **“I want 95 KDE”**, or **“MCP 95 50”**.\n"
                + ("\n".join(meta_msgs) if meta_msgs else "")
            )
        }]
        map_html = _render_points_tracks_map(get_cached_df())
        return (chat_msgs,
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                map_html,
                gr.update(visible=False), gr.update(visible=False), gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False))

    # C) need user to pick X/Y and provide CRS
    return ([
                {"role": "assistant", "content":
                    "CSV uploaded. Your coordinates do not appear to be latitude/longitude. "
                    "Please specify X (easting), Y (northing), and the CRS/UTM zone below."}
            ],
            gr.update(choices=cached_headers, value=found_x, visible=True),
            gr.update(choices=cached_headers, value=found_y, visible=True),
            gr.update(visible=True),
            render_empty_map(),
            gr.update(visible=True), gr.update(visible=True), gr.update(visible=True),
            gr.update(visible=True),  # confirm button visible
            gr.update(visible=False))  # download hidden


def handle_upload_confirm(x_col, y_col, crs_text):
    """
    Confirm coordinate columns and (if required) reproject to WGS84.
    Then detect standard schema (ID/timestamp) and render the map.
    Returns: HTML map (single output as wired in UI).
    """
    df = get_cached_df().copy()

    if x_col not in df.columns or y_col not in df.columns:
        return "<p>Selected coordinate columns not found in data.</p>"

    # If already lon/lat columns by name, validate; else require CRS
    if x_col.lower() in ["longitude", "lon"] and y_col.lower() in ["latitude", "lat"]:
        try:
            lon_ok = df[x_col].astype(float).between(-180, 180).all()
            lat_ok = df[y_col].astype(float).between(-90, 90).all()
        except Exception:
            lon_ok = lat_ok = False

        if lon_ok and lat_ok:
            df["longitude"] = df[x_col]
            df["latitude"]  = df[y_col]
        else:
            if not str(crs_text).strip():
                return "<p>Your columns are named lon/lat but values are not geographic. Please enter a CRS (e.g., 'UTM 10T' or 'EPSG:32610').</p>"
            try:
                epsg = parse_crs_input(crs_text)
                from pyproj import Transformer
                transformer = Transformer.from_crs(epsg, 4326, always_xy=True)
                df["longitude"], df["latitude"] = transformer.transform(df[x_col].values, df[y_col].values)
            except Exception as e:
                return f"<p>Failed to convert coordinates: {e}</p>"

    else:
        # Generic X/Y → need CRS to convert
        if not str(crs_text).strip():
            return "<p>Please enter a CRS or UTM zone before confirming (e.g., 'UTM 10T' or 'EPSG:32610').</p>"
        try:
            epsg = parse_crs_input(crs_text)
            from pyproj import Transformer
            transformer = Transformer.from_crs(epsg, 4326, always_xy=True)
            df["longitude"], df["latitude"] = transformer.transform(df[x_col].values, df[y_col].values)
        except Exception as e:
            return f"<p>Failed to convert coordinates: {e}</p>"

    # Detect & standardize ID/timestamp after coordinates are valid
    df, meta_msgs = detect_and_standardize(df)
    set_cached_df(df)

    return _render_points_tracks_map(df)

# --------------------------------------------------------------------------------------
# Chat / Analysis
# --------------------------------------------------------------------------------------
def handle_chat(chat_history, user_message):
    """
    Handles user chat. If a home range request is detected (via LLM tool or keywords),
    run MCP/KDE and refresh the map. Also handles metadata commands like
    “timestamp column is GMT_DateTime” / “no id”.
    """
    chat_history = list(chat_history)

    # 1) Check if user is telling us schema info (id/timestamp mapping)
    cmd = parse_metadata_command(user_message or "")
    if cmd:
        df = get_cached_df()
        if df is None:
            chat_history.append({"role": "assistant", "content": "Please upload a CSV first."})
            return chat_history, gr.update(), gr.update(visible=False)

        df2, msg = try_apply_user_mapping(df, cmd)
        set_cached_df(df2)

        # Rebuild map to reflect tracks if timestamp just became available
        map_html = _render_points_tracks_map(get_cached_df())

        # If timestamp detected now, optionally nudge about date range
        hints = []
        if TS_COL in df2.columns and df2[TS_COL].notna().any():
            tmin = pd.to_datetime(df2[TS_COL]).min()
            tmax = pd.to_datetime(df2[TS_COL]).max()
            hints.append(f"Detected time range: **{tmin} → {tmax}**. You can say things like “MCP 95”, “KDE 99”, or “MCP 95 50”.")
        else:
            hints.append("If you have times, you can say e.g. **“timestamp column is GMT_DateTime”**. Otherwise you can still run MCP/KDE.")

        chat_history.append({"role": "assistant", "content": f"{msg}\n" + "\n".join(hints)})
        return chat_history, gr.update(value=map_html), gr.update(visible=False)

    # 2) Normal LLM routing
    tool, llm_output = ask_llm(chat_history, user_message)

    mcp_list, kde_list = [], []
    # Tool JSON branch
    if tool and tool.get("tool") == "home_range":
        method = tool.get("method")
        levels = tool.get("levels", [95])
        levels = [min(int(p), 99) for p in levels if 1 <= int(p) <= 100]
        if method == "mcp":
            mcp_list = levels
        elif method == "kde":
            kde_list = levels

    # Fallback: keyword parse
    if "mcp" in (user_message or "").lower():
        mcp_list = parse_levels_from_text(user_message)
    if "kde" in (user_message or "").lower():
        kde_list = parse_levels_from_text(user_message)

    # If not an analysis request, reply naturally (short)
    if not mcp_list and not kde_list:
        if llm_output:
            chat_history.append({"role": "user", "content": user_message})
            chat_history.append({"role": "assistant", "content": llm_output})
            return chat_history, gr.update(), gr.update(visible=False)

        # If we failed schema detection on upload, remind user they can tell us columns.
        df = get_cached_df()
        if df is not None:
            nudges = []
            if ID_COL not in df.columns:
                nudges.append('If you have an individual column, say: **“ID column is your_id_col”** or **“no id”**.')
            if TS_COL not in df.columns or df[TS_COL].notna().mean() < 0.5:
                nudges.append('If you have a timestamp column, say: **“Timestamp column is your_time_col”** or **“no timestamp”**.')
            if nudges:
                chat_history.append({"role": "assistant", "content": " ".join(nudges)})
                return chat_history, gr.update(), gr.update(visible=False)

        chat_history.append({"role": "assistant", "content": "How can I help you? Please upload a CSV for analysis or ask a question."})
        return chat_history, gr.update(), gr.update(visible=False)

    # Must have lon/lat prepared
    df = get_cached_df()
    if df is None or "latitude" not in df or "longitude" not in df:
        chat_history.append({"role": "assistant", "content": "CSV must be uploaded with 'latitude' and 'longitude' columns."})
        return chat_history, gr.update(), gr.update(visible=False)

    results_exist = False
    warned_about_kde_100 = False

    # KDE 100% → clamp to 99% and warn (as before)
    if kde_list:
        if 100 in kde_list or any("100" in s for s in (user_message or "").split()):
            warned_about_kde_100 = True
        kde_list = [min(k, 99) for k in kde_list]

    # Run analyses (delegated to estimators/*)
    if mcp_list:
        add_mcps(df, mcp_list)
        requested_percents.update(mcp_list)
        results_exist = True
    if kde_list:
        add_kdes(df, kde_list)
        requested_kde_percents.update(kde_list)
        results_exist = True

    # Build map (points, tracks, + overlays)
    m = folium.Map(location=[df["latitude"].mean(), df["longitude"].mean()], zoom_start=9)
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

    # Points & tracks again (to keep exactly your previous look)
    # Reuse the helper to avoid drift:
    base_html = _render_points_tracks_map(df)

    # MCP layers
    animal_ids = df[ID_COL].astype(str).fillna("sample").unique() if ID_COL in df.columns else ["sample"]
    color_map = {aid: f"#{random.randint(0, 0xFFFFFF):06x}" for aid in animal_ids}

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
    map_html = m._repr_html_()

    # Compose assistant message & ZIP
    msg_bits = []
    if requested_percents:
        msg_bits.append(f"MCP home ranges ({', '.join(str(p) for p in sorted(requested_percents))}%) calculated.")
    if requested_kde_percents:
        msg_bits.append(f"KDE home ranges ({', '.join(str(p) for p in sorted(requested_kde_percents))}%) calculated (raster & contours).")
    if warned_about_kde_100:
        msg_bits.append("Note: KDE at 100% is not supported and has been replaced by 99% for compatibility (as done in scientific software).")

    # Wording fix: download is to the right, below the map
    msg_bits.append("Use the **📥 Download Results** button to the right (below the map).")

    chat_history.append({"role": "user", "content": user_message})
    chat_history.append({"role": "assistant", "content": " ".join(msg_bits)})

    archive_path = _zip_outputs()
    return chat_history, gr.update(value=map_html), gr.update(value=archive_path, visible=True)

# --------------------------------------------------------------------------------------
# UI (layout unchanged)
# --------------------------------------------------------------------------------------
with gr.Blocks(title="SpatChat: Home Range Analysis") as demo:
    gr.Image(
        value="logo_long1.png",
        show_label=False,
        show_download_button=False,
        show_share_button=False,
        type="filepath",
        elem_id="logo-img"
    )
    gr.HTML("""
    <style>
    #logo-img img {
        height: 90px;
        margin: 10px 50px 10px 10px;
        border-radius: 6px;
    }
    </style>
    """)
    gr.Markdown("## 🏠 SpatChat: Home Range Analysis {hr}  🦊🦉🐢")
    gr.HTML("""
    <div style="margin-top: -10px; margin-bottom: 15px;">
      <input type="text" value="https://spatchat.org/browse/?room=hr" id="shareLink" readonly style="width: 50%; padding: 5px; background-color: #f8f8f8; color: #222; font-weight: 500; border: 1px solid #ccc; border-radius: 4px;">
      <button onclick="navigator.clipboard.writeText(document.getElementById('shareLink').value)" style="padding: 5px 10px; background-color: #007BFF; color: white; border: none; border-radius: 4px; cursor: pointer;">
        📋 Copy Share Link
      </button>
      <div style="margin-top: 10px; font-size: 14px;">
        <b>Share:</b>
        <a href="https://twitter.com/intent/tweet?text=Checkout+Spatchat!&url=https://spatchat.org/browse/?room=hr" target="_blank">🐦 Twitter</a> |
        <a href="https://www.facebook.com/sharer/sharer.php?u=https://spatchat.org/browse/?room=hr" target="_blank">📘 Facebook</a>
      </div>
    </div>
    """)
    gr.Markdown("""
        <div style="font-size: 14px;">
        © 2025 Ho Yi Wan & Logan Hysen. All rights reserved.<br>
        If you use Spatchat in research, please cite:<br>
        <b>Wan, H.Y.</b> & <b>Hysen, L.</b> (2025). <i>SpatChat: Home Range Analysis.</i>
        </div>
    """)

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="SpatChat",
                show_label=True,
                type="messages",
                value=[{"role": "assistant", "content": "Welcome to SpatChat! Please upload a CSV containing coordinates (lat/lon or UTM) and optional timestamp/animal_id to begin."}]
            )
            user_input = gr.Textbox(
                label="Ask Spatchat",
                placeholder="Type commands...",
                lines=1
            )
            file_input = gr.File(
                label="Upload Movement CSV (.csv or .txt only)",
                file_types=[".csv", ".txt"]
            )
            x_col = gr.Dropdown(label="X column", choices=[], visible=False)
            y_col = gr.Dropdown(label="Y column", choices=[], visible=False)
            crs_text = gr.Text(label="CRS (e.g. '32633', '33N', or 'EPSG:32633')", visible=False)
            confirm_btn = gr.Button("Confirm Coordinate Settings", visible=False)
        with gr.Column(scale=3):
            map_output = gr.HTML(label="Map Preview", value=render_empty_map(), show_label=False)
            download_btn = gr.DownloadButton(
                "📥 Download Results",
                value=None,
                visible=False
            )

    # Queue (unchanged)
    demo.queue(max_size=16)

    # Wire events (same outputs ordering)
    file_input.change(
        fn=handle_upload_initial,
        inputs=file_input,
        outputs=[
            chatbot, x_col, y_col, crs_text, map_output,
            x_col, y_col, crs_text, confirm_btn, download_btn
        ]
    )
    confirm_btn.click(
        fn=confirm_and_hint,
        inputs=[x_col, y_col, crs_text, chatbot],
        outputs=[map_output, chatbot]
    )
    user_input.submit(
        fn=handle_chat,
        inputs=[chatbot, user_input],
        outputs=[chatbot, map_output, download_btn]
    )
    user_input.submit(lambda *args: "", inputs=None, outputs=user_input)

# HF Spaces-friendly launch
demo.launch(ssr_mode=False)
