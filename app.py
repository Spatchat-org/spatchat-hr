# ================================
# app.py (core, without UI)
# ================================
import os
import json
import re
import time
import shutil
import random
import sys
import zipfile

import gradio as gr
import pandas as pd
import numpy as np
import folium
import rasterio
from rasterio.transform import from_origin
import matplotlib.pyplot as plt
from skimage import measure
from shapely.geometry import Polygon, MultiPolygon, mapping

# ---- Local modules ----
from storage import (
    get_cached_df, set_cached_df,
    get_cached_headers, set_cached_headers,
    clear_all_results,
    mcp_results, kde_results,
    requested_percents, requested_kde_percents,
    save_all_mcps_zip,       # if you prefer to reuse storage.zip builder
)
from llm_utils import ask_llm
from crs_utils import parse_crs_input
from map_utils import render_empty_map, fit_map_to_bounds
from estimators.mcp import add_mcps
from estimators.kde import add_kdes

# Auto-detection for ID / timestamp
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

# Track per-upload “pending questions” we need to ask the user (id/timestamp)
PENDING_QUESTIONS = {
    "need_id": False,
    "need_ts": False,
    "ts_prompted": False,   # to avoid repeating the same question
    "id_prompted": False,
}

# --------------------------------------------------------------------------------------
# Upload flow
# --------------------------------------------------------------------------------------
def handle_upload_initial(file):
    """
    1) Cache uploaded CSV
    2) Try to auto-detect lat/lon columns
    3) If ambiguous or projected, show pickers and CRS input
    4) Also detect animal_id / timestamp and report to the user
    Returns the same outputs ordering your UI expects.
    """
    clear_all_results()

    os.makedirs("uploads", exist_ok=True)
    filename = os.path.join("uploads", os.path.basename(file))
    shutil.copy(file, filename)

    try:
        df = pd.read_csv(filename)
        set_cached_df(df)
        set_cached_headers(list(df.columns))
    except Exception as e:
        print(f"[upload] failed to read CSV: {e}", file=sys.stderr)
        # chatbot, x, y, crs, map, x, y, crs, confirm, download  (total 10)
        return [], *(gr.update(visible=False) for _ in range(4)), render_empty_map(), *(gr.update(visible=False) for _ in range(5))

    cached_headers = get_cached_headers()
    lower_cols = [c.lower() for c in cached_headers]

    # ---------- Branch A: explicit latitude/longitude column names ----------
    if "latitude" in lower_cols and "longitude" in lower_cols:
        lat_col = cached_headers[lower_cols.index("latitude")]
        lon_col = cached_headers[lower_cols.index("longitude")]

        # If labeled as lat/lon but values look projected → ask for CRS
        if looks_invalid_latlon(get_cached_df(), lat_col, lon_col):
            return [
                {"role": "assistant", "content":
                 "CSV uploaded. Your coordinates do not appear to be latitude/longitude. "
                 "Please specify X (easting), Y (northing), and the CRS/UTM zone below "
                 "(e.g., 'UTM 10T' or 'EPSG:32610')."}
            ], \
            gr.update(choices=cached_headers, value=lon_col, visible=True), \
            gr.update(choices=cached_headers, value=lat_col, visible=True), \
            gr.update(visible=True), \
            render_empty_map(), \
            *(gr.update(visible=True) for _ in range(4)), gr.update(visible=False)

        # Looks valid; standardize ID/timestamp too, then render map immediately
        df0 = get_cached_df().copy()
        # Keep the same lon/lat column names that downstream expects
        df0["longitude"] = df0[lon_col]
        df0["latitude"]  = df0[lat_col]
        # Detect ID / timestamp
        df1, meta_msgs = detect_and_standardize(df0)
        set_cached_df(df1)

        # Build the map using confirmed lon/lat
        map_html = handle_upload_confirm("longitude", "latitude", "")

        # Compose a transparent message about what we detected
        id_found  = "animal_id" in df1.columns
        ts_found  = "timestamp" in df1.columns
        id_note   = f"• **ID column**: `{ 'animal_id' if id_found else 'not detected' }`"
        ts_note   = f"• **Timestamp column**: `{ 'timestamp' if ts_found else 'not detected' }`"
        tips = []
        if not id_found:
            tips.append("If your data has one, say: **“ID column is <your_col>”** or **“no id”**.")
        if not ts_found:
            tips.append("If your data has one, say: **“Timestamp column is <your_col>”** or **“no timestamp”**.")

        msg = (
            "CSV uploaded. Latitude and longitude detected.\n\n"
            "Detected schema:\n"
            f"• **Longitude**: `{lon_col}`\n"
            f"• **Latitude**: `{lat_col}`\n"
            f"{id_note}\n"
            f"{ts_note}\n\n"
        )
        if tips:
            msg += "You can correct me in chat:\n" + "\n".join(f"  - {t}" for t in tips) + "\n\n"
        msg += (
            "You may now create home ranges. For example:\n"
            "• “I want 100% MCP”\n"
            "• “I want 95 KDE”\n"
            "• “MCP 95 50”\n\n"
        )

        return [
            {"role": "assistant", "content": msg}
        ], *(gr.update(visible=False) for _ in range(3)), map_html, *(gr.update(visible=False) for _ in range(4)), gr.update(visible=False)

    # ---------- Branch B: heuristic lat/lon guess ----------
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
        df0 = df.copy()
        # Assign canonical lon/lat columns
        df0["longitude"] = df0[found_x] if latlon_guess == "lonlat" else df0[found_y]
        df0["latitude"]  = df0[found_y] if latlon_guess == "lonlat" else df0[found_x]
        # Detect ID / timestamp
        df1, meta_msgs = detect_and_standardize(df0)
        set_cached_df(df1)

        map_html = handle_upload_confirm("longitude", "latitude", "")

        id_found = "animal_id" in df1.columns
        ts_found = "timestamp" in df1.columns
        id_note  = f"• **ID column**: `{ 'animal_id' if id_found else 'not detected' }`"
        ts_note  = f"• **Timestamp column**: `{ 'timestamp' if ts_found else 'not detected' }`"
        tips = []
        if not id_found:
            tips.append("If your data has one, say: **“ID column is <your_col>”** or **“no id”**.")
        if not ts_found:
            tips.append("If your data has one, say: **“Timestamp column is <your_col>”** or **“no timestamp”**.")

        msg = (
            f"CSV uploaded. `{found_x}`/`{found_y}` interpreted as longitude/latitude.\n\n"
            "Detected schema:\n"
            f"• **Longitude** source: `{found_x if latlon_guess == 'lonlat' else found_y}`\n"
            f"• **Latitude** source: `{found_y if latlon_guess == 'lonlat' else found_x}`\n"
            f"{id_note}\n"
            f"{ts_note}\n\n"
        )
        if tips:
            msg += "You can correct me in chat:\n" + "\n".join(f"  - {t}" for t in tips) + "\n\n"
        msg += (
            "You may now create home ranges. For example:\n"
            "• “I want 100% MCP”\n"
            "• “I want 95 KDE”\n"
            "• “MCP 95 50”\n\n"
            "_(The download button is below the preview map.)_"
        )

        return [
            {"role": "assistant", "content": msg}
        ], *(gr.update(visible=False) for _ in range(3)), map_html, *(gr.update(visible=False) for _ in range(4)), gr.update(visible=False)

    # ---------- Branch C: need user to pick X/Y and provide CRS ----------
    return [
        {"role": "assistant", "content":
         "CSV uploaded. Your coordinates do not appear to be latitude/longitude. "
         "Please specify X (easting), Y (northing), and the CRS/UTM zone below."}
    ], \
    gr.update(choices=cached_headers, value=found_x, visible=True), \
    gr.update(choices=cached_headers, value=found_y, visible=True), \
    gr.update(visible=True), \
    render_empty_map(), \
    *(gr.update(visible=True) for _ in range(4)), gr.update(visible=False)

def handle_upload_confirm(x_col, y_col, crs_text):
    """
    Confirm coordinate columns and (if required) reproject to WGS84.
    Also runs schema detection for animal_id/timestamp and records a
    one-line detection summary for the chat.
    Returns: HTML map (same single-output signature you already wired).
    """
    import storage  # local state
    from crs_utils import parse_crs_input
    from schema_detect import detect_and_standardize, ID_COL, TS_COL

    df = storage.get_cached_df()
    if df is None:
        return "<p>No data loaded. Please upload a CSV first.</p>"
    df = df.copy()

    if x_col not in df.columns or y_col not in df.columns:
        return "<p>Selected coordinate columns not found in data.</p>"

    crs_note = ""
    # If already lon/lat columns by name, validate; if invalid ranges → require CRS
    if x_col.lower() in ["longitude", "lon"] and y_col.lower() in ["latitude", "lat"]:
        try:
            lon_ok = df[x_col].astype(float).between(-180, 180).all()
            lat_ok = df[y_col].astype(float).between(-90, 90).all()
        except Exception:
            lon_ok = lat_ok = False

        if lon_ok and lat_ok:
            df["longitude"] = df[x_col]
            df["latitude"]  = df[y_col]
            crs_note = "interpreted as WGS84 lon/lat"
        else:
            if not str(crs_text).strip():
                return "<p>Your columns are named lon/lat but values are not geographic. Please enter a CRS (e.g., 'UTM 10T' or 'EPSG:32610').</p>"
            try:
                epsg = parse_crs_input(crs_text)
                from pyproj import Transformer
                transformer = Transformer.from_crs(epsg, 4326, always_xy=True)
                df["longitude"], df["latitude"] = transformer.transform(df[x_col].values, df[y_col].values)
                crs_note = f"reprojected from EPSG:{epsg} → WGS84"
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
            crs_note = f"reprojected from EPSG:{epsg} → WGS84"
        except Exception as e:
            return f"<p>Failed to convert coordinates: {e}</p>"

    # Detect & standardize metadata columns (animal_id/timestamp)
    df, meta_msgs = detect_and_standardize(df)

    # Persist
    storage.set_cached_df(df)

    # Build a concise detection summary (for chat)
    id_msg = TS_msg = "not detected"
    if "animal_id" in df.columns: id_msg = "animal_id"
    if "timestamp" in df.columns:  TS_msg = "timestamp (UTC)"

    summary = (
        "Detected columns → "
        f"X: `{x_col}`, Y: `{y_col}` ({crs_note}); "
        f"ID: `{id_msg}`, time: `{TS_msg}`. "
        "If anything's wrong, say e.g. **“ID column is tag_id”** or **“Timestamp column is GMT_DateTime”**."
    )
    # If schema supplied guidance (e.g., not detected), append
    if meta_msgs:
        summary += " " + " ".join(meta_msgs)

    storage.set_detection_summary(summary)

    # Build preview map (unchanged)
    has_timestamp = "timestamp" in df.columns
    m = folium.Map(location=[df["latitude"].mean(), df["longitude"].mean()], zoom_start=9, control_scale=True)
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

    animal_ids = df["animal_id"].unique() if "animal_id" in df.columns else ["sample"]
    color_map = {aid: f"#{random.randint(0, 0xFFFFFF):06x}" for aid in animal_ids}

    for animal in animal_ids:
        track = df[df["animal_id"] == animal] if "animal_id" in df.columns else df
        coords = list(zip(track["latitude"], track["longitude"]))
        color = color_map.get(animal, "#3388ff")
        if has_timestamp:
            track = track.sort_values("timestamp")
            folium.PolyLine(coords, color=color, weight=2.5, opacity=0.8, popup=str(animal)).add_to(lines_layer)
        for _, row in track.iterrows():
            label = f"{animal}" + (f"<br>{row['timestamp']}" if has_timestamp and pd.notna(row.get('timestamp')) else "")
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

def confirm_and_hint(x_col, y_col, crs_text):
    """
    Small wrapper used by some UIs. Returns exactly the map HTML like handle_upload_confirm.
    """
    return handle_upload_confirm(x_col, y_col, crs_text)

# --------------------------------------------------------------------------------------
# Analysis + chat handler
# --------------------------------------------------------------------------------------
def handle_chat(chat_history, user_message):
    """
    Handles user chat. If a home range request is detected (via LLM tool or keywords),
    run MCP/KDE and refresh the map. Otherwise answer briefly (via llm_utils.ask_llm).

    Also processes user commands like:
      - "timestamp column is <name>"
      - "id column is <name>"
      - "no timestamp"
      - "no id"
    """
    chat_history = list(chat_history)

    # If user is supplying metadata mapping (timestamp/id), apply it first
    cmd = parse_metadata_command(user_message)
    if cmd:
        df = get_cached_df()
        if df is None or "latitude" not in df or "longitude" not in df:
            chat_history.append({"role": "assistant", "content": "Please upload a CSV first."})
            return chat_history, gr.update(), gr.update(visible=False)

        df2, msg = try_apply_user_mapping(df, cmd)
        set_cached_df(df2)

        # Update pending questions flags based on result
        PENDING_QUESTIONS["need_id"] = (ID_COL not in df2.columns)
        PENDING_QUESTIONS["need_ts"] = (TS_COL not in df2.columns)

        # Build a minimal reply (no map update here; user can run analysis)
        follow = []
        if not PENDING_QUESTIONS["need_id"] and not PENDING_QUESTIONS["need_ts"]:
            follow.append("Great — ID and timestamps detected.")
        elif PENDING_QUESTIONS["need_id"] and not PENDING_QUESTIONS["id_prompted"]:
            follow.append("I couldn’t detect an individual ID column. If your data has one, say: “ID column is tag_id”.")
            PENDING_QUESTIONS["id_prompted"] = True
        elif PENDING_QUESTIONS["need_ts"] and not PENDING_QUESTIONS["ts_prompted"]:
            follow.append("I couldn’t detect a timestamp column. If your data has one, say: “Timestamp column is datetime”.")
            PENDING_QUESTIONS["ts_prompted"] = True

        chat_history.append({"role": "assistant", "content": msg + (" " + " ".join(follow) if follow else "")})
        return chat_history, gr.update(), gr.update(visible=False)

    # Normal tool-intent call
    tool, llm_output = ask_llm(chat_history, user_message)

    mcp_list, kde_list = [], []
    if tool and tool.get("tool") == "home_range":
        method = tool.get("method")
        levels = tool.get("levels", [95])
        levels = [min(int(p), 99) for p in levels if 1 <= int(p) <= 100]
        if method == "mcp":
            mcp_list = levels
        elif method == "kde":
            kde_list = levels

    # Fallback: keyword parse
    if "mcp" in user_message.lower():
        mcp_list = parse_levels_from_text(user_message)
    if "kde" in user_message.lower():
        kde_list = parse_levels_from_text(user_message)

    # If not an analysis request, reply naturally (short)
    if not mcp_list and not kde_list:
        # If we still owe the user clarifications about missing columns, prompt once
        if PENDING_QUESTIONS["need_id"] and not PENDING_QUESTIONS["id_prompted"]:
            chat_history.append({"role": "assistant", "content":
                                 "I couldn’t detect an individual ID column. If your data has one, say: “ID column is tag_id”. "
                                 "Otherwise, you can still proceed — I’ll treat all rows as one animal."})
            PENDING_QUESTIONS["id_prompted"] = True
            return chat_history, gr.update(), gr.update(visible=False)

        if PENDING_QUESTIONS["need_ts"] and not PENDING_QUESTIONS["ts_prompted"]:
            chat_history.append({"role": "assistant", "content":
                                 "I couldn’t detect a timestamp column. If your data has one, say: “Timestamp column is datetime”. "
                                 "Otherwise, you can proceed — I’ll plot points without drawing tracks."})
            PENDING_QUESTIONS["ts_prompted"] = True
            return chat_history, gr.update(), gr.update(visible=False)

        # Otherwise short LLM response
        if llm_output:
            chat_history.append({"role": "user", "content": user_message})
            chat_history.append({"role": "assistant", "content": llm_output})
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
        if 100 in kde_list or any("100" in s for s in user_message.split()):
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

    # Build map (layout unchanged)
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

    points_layer = folium.FeatureGroup(name="Points", show=True)
    paths_layer  = folium.FeatureGroup(name="Tracks", show=True)

    # Use standardized names if present
    has_timestamp = (TS_COL in df.columns)
    use_id = ID_COL if ID_COL in df.columns else None

    animal_ids = (df[use_id].astype(str).unique() if use_id else ["sample"])
    color_map = {aid: f"#{random.randint(0, 0xFFFFFF):06x}" for aid in animal_ids}

    # Tracks & points
    for animal in animal_ids:
        if use_id:
            track = df[df[use_id].astype(str) == str(animal)]
        else:
            track = df

        color = color_map[animal]
        if has_timestamp:
            track = track.sort_values(TS_COL)
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
    msg = []
    if requested_percents:
        msg.append(f"MCP home ranges ({', '.join(str(p) for p in sorted(requested_percents))}%) calculated.")
    if requested_kde_percents:
        msg.append(f"KDE home ranges ({', '.join(str(p) for p in sorted(requested_kde_percents))}%) calculated (raster & contours).")
    if warned_about_kde_100:
        msg.append("Note: KDE at 100% is not supported and has been replaced by 99% for compatibility (as done in scientific software).")
    
    # Only mention the download button if we actually produced results
    if results_exist:
        msg.append("_The download button is below the preview map._")

    chat_history.append({"role": "user", "content": user_message})
    chat_history.append({"role": "assistant", "content": " ".join(msg)})

    # Either reuse storage.save_all_mcps_zip() or keep the local builder; both are fine.
    archive_path = save_all_mcps_zip()
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
