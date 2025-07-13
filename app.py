import os
import json
import gradio as gr
import pandas as pd
import folium
import shutil
import random
import numpy as np
from pyproj import Transformer
from together import Together
from dotenv import load_dotenv
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon, mapping, MultiPolygon
import zipfile
import time
import re
import rasterio
from rasterio.transform import from_origin
import matplotlib.pyplot as plt
from skimage import measure
from sklearn.neighbors import KernelDensity
import tempfile

print("Starting SpatChat (multi-MCP/KDE, robust download version)")

# ─── GLOBAL STATE ─────────────────────────────────────────────────────────────
mcp_results = {}
kde_results = {}
requested_percents = set()
requested_kde_percents = set()
cached_df = None
cached_headers = []

def clear_all_results():
    global mcp_results, kde_results, requested_percents, requested_kde_percents
    mcp_results = {}
    kde_results = {}
    requested_percents = set()
    requested_kde_percents = set()
    # Clean outputs folder for new session
    if os.path.exists("outputs"):
        try:
            shutil.rmtree("outputs")
        except Exception as e:
            print(f"Warning: Failed to remove outputs directory: {e}")
            for fname in os.listdir("outputs"):
                file_path = os.path.join("outputs", fname)
                try:
                    os.remove(file_path)
                except Exception as e2:
                    print(f"Could not remove file {file_path}: {e2}")
    os.makedirs("outputs", exist_ok=True)

# ─── LLM SETUP (Together API) ─────────────────────────────────────────────────
load_dotenv()
client = Together(api_key=os.getenv("TOGETHER_API_KEY"))

SYSTEM_PROMPT = """
You are SpatChat, an expert wildlife home range analysis assistant.
If the user asks for a home range calculation (MCP, KDE, dBBMM, AKDE, etc.), reply ONLY in JSON using this format:
{"tool": "home_range", "method": "mcp", "levels": [95, 50]}
- method: one of "mcp", "kde", "akde", "bbmm", "dbbmm"
- levels: list of percentages for the home range (default [95] if user doesn't specify)
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
        conv = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            messages=[{"role": "system", "content": FALLBACK_PROMPT}] + messages,
            temperature=0.7
        ).choices[0].message.content
        return None, conv

# ─── MAP UTILITIES ─────────────────────────────────────────────────────────────
def render_empty_map():
    m = folium.Map(location=[0, 0], zoom_start=2, control_scale=True)
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
    folium.LayerControl(collapsed=False).add_to(m)
    return m._repr_html_()

def fit_map_to_bounds(m, df):
    min_lat, max_lat = df['latitude'].min(), df['latitude'].max()
    min_lon, max_lon = df['longitude'].min(), df['longitude'].max()
    if np.isfinite([min_lat, max_lat, min_lon, max_lon]).all():
        m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])
    return m

def looks_like_latlon(df, x_col, y_col):
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
    try:
        lat = df[lat_col].astype(float)
        lon = df[lon_col].astype(float)
        return not (lat.between(-90, 90).all() and lon.between(-180, 180).all())
    except Exception:
        return True

# ─── UPLOAD HANDLERS ───────────────────────────────────────────────────────────
def handle_upload_initial(file):
    global cached_df, cached_headers
    clear_all_results()
    os.makedirs("uploads", exist_ok=True)
    filename = os.path.join("uploads", os.path.basename(file))
    shutil.copy(file, filename)
    try:
        df = pd.read_csv(filename)
        cached_df = df
        cached_headers = list(df.columns)
    except Exception:
        return [], *(gr.update(visible=False) for _ in range(8)), render_empty_map(), gr.update(visible=False)
    lower = [c.lower() for c in df.columns]
    if "latitude" in lower and "longitude" in lower:
        lat_c = df.columns[lower.index("latitude")]
        lon_c = df.columns[lower.index("longitude")]
        if looks_invalid_latlon(df, lat_c, lon_c):
            return (
                [{"role":"assistant","content":"Columns look labeled 'latitude'/'longitude' but values seem off. Please confirm CRS."}],
                gr.update(choices=cached_headers, value=lon_c, visible=True),
                gr.update(choices=cached_headers, value=lat_c, visible=True),
                gr.update(visible=True),
                render_empty_map(),
                *(gr.update(visible=True) for _ in range(4)),
                gr.update(visible=False)
            )
        else:
            return (
                [{"role":"assistant","content":"CSV uploaded. Latitude/Longitude detected."}],
                *(gr.update(visible=False) for _ in range(3)),
                handle_upload_confirm("longitude","latitude",""),
                *(gr.update(visible=False) for _ in range(4)),
                gr.update(visible=False)
            )
    # auto-detect X/Y
    x_names = ["x","easting","lon","longitude"]
    y_names = ["y","northing","lat","latitude"]
    found_x = next((c for c in df.columns if c.lower() in x_names), df.columns[0])
    found_y = next((c for c in df.columns if c.lower() in y_names and c!=found_x), df.columns[1] if len(df.columns)>1 else df.columns[0])
    if found_x==found_y and len(df.columns)>1:
        found_y = df.columns[1 if df.columns[0]==found_x else 0]
    guess = looks_like_latlon(df, found_x, found_y)
    if guess:
        df["longitude"] = df[found_x] if guess=="lonlat" else df[found_y]
        df["latitude"]  = df[found_y] if guess=="lonlat" else df[found_x]
        cached_df = df
        return (
            [{"role":"assistant","content":f"CSV uploaded. `{found_x}`/`{found_y}` used as lon/lat."}],
            *(gr.update(visible=False) for _ in range(3)),
            handle_upload_confirm("longitude","latitude",""),
            *(gr.update(visible=False) for _ in range(4)),
            gr.update(visible=False)
        )
    # fallback: need CRS
    return (
        [{"role":"assistant","content":"CSV uploaded. Coordinates not lat/lon — please pick X/Y and CRS."}],
        gr.update(choices=cached_headers, value=found_x, visible=True),
        gr.update(choices=cached_headers, value=found_y, visible=True),
        gr.update(visible=True),
        render_empty_map(),
        *(gr.update(visible=True) for _ in range(4)),
        gr.update(visible=False)
    )

def parse_crs_input(s):
    s = str(s).strip().upper()
    if s.startswith("EPSG:"):
        return int(s.split(":")[1])
    if s.isdigit():
        return 32600 + int(s)
    if len(s)>=2 and s[:-1].isdigit() and s[-1] in ("N","S"):
        z=int(s[:-1])
        return (32600+z) if s[-1]=="N" else (32700+z)
    raise ValueError("Invalid CRS input")

def handle_upload_confirm(x_col,y_col,crs_input):
    global cached_df
    df = cached_df.copy()
    if x_col not in df.columns or y_col not in df.columns:
        return "<p>Column not found.</p>"
    if x_col.lower() in ["longitude","lon"] and y_col.lower() in ["latitude","lat"]:
        df["longitude"]=df[x_col]; df["latitude"]=df[y_col]
    else:
        epsg=parse_crs_input(crs_input)
        transformer=Transformer.from_crs(epsg,4326,always_xy=True)
        df["longitude"],df["latitude"] = transformer.transform(df[x_col].values,df[y_col].values)
    if "timestamp" in df.columns:
        df["timestamp"]=pd.to_datetime(df["timestamp"])
    if "animal_id" not in df.columns:
        df["animal_id"]="sample"
    cached_df = df

    # build map
    m=folium.Map(location=[df["latitude"].mean(),df["longitude"].mean()],zoom_start=9,control_scale=True)
    folium.TileLayer("OpenStreetMap").add_to(m)
    folium.TileLayer("CartoDB positron",attr="CartoDB").add_to(m)
    folium.TileLayer(tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",attr="OpenTopoMap",name="Topographic").add_to(m)
    folium.TileLayer(tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",attr="Esri",name="Satellite").add_to(m)

    pts=folium.FeatureGroup(name="Points",show=True)
    lines=folium.FeatureGroup(name="Tracks",show=True)
    cmap={}
    for aid in df["animal_id"].unique():
        cmap[aid] = f"#{random.randint(0,0xFFFFFF):06x}"
        track = df[df["animal_id"]==aid]
        if "timestamp" in track.columns:
            t_sorted = track.sort_values("timestamp")
            coords = list(zip(t_sorted["latitude"],t_sorted["longitude"]))
            folium.PolyLine(coords,color=cmap[aid],weight=2.5,opacity=0.8).add_to(lines)
        for row in track.itertuples():
            folium.CircleMarker(
                location=[row.latitude,row.longitude],
                radius=3,
                color=cmap[aid],
                fill=True,
                fill_opacity=0.7,
                popup=f"{aid}" + (f"<br>{row.timestamp}" if "timestamp" in df else "")
            ).add_to(pts)
    pts.add_to(m); lines.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m = fit_map_to_bounds(m, df)
    return m._repr_html_()

# ─── PARSING ───────────────────────────────────────────────────────────────────
def parse_levels_from_text(text):
    vals = [int(v) for v in re.findall(r"\b([1-9][0-9]?|100)\b", text)]
    return sorted(set(vals)) if vals else [95]

# ─── MCP FUNCTIONS ─────────────────────────────────────────────────────────────
def mcp_polygon(latitudes, longitudes, percent=95):
    pts = np.column_stack((longitudes, latitudes))
    if len(pts) < 3:
        return None
    ctr = pts.mean(axis=0)
    dists = np.linalg.norm(pts - ctr, axis=1)
    k = max(3, int(len(pts) * (percent/100.0)))
    kept = pts[np.argsort(dists)[:k]]
    if len(kept) < 3:
        return None
    hull = ConvexHull(kept)
    return kept[hull.vertices]

def add_mcps(df, percent_list):
    global mcp_results
    for pct in percent_list:
        for aid in df["animal_id"].unique():
            if aid not in mcp_results:
                mcp_results[aid] = {}
            if pct not in mcp_results[aid]:
                hull_pts = mcp_polygon(df["latitude"].values, df["longitude"].values, pct)
                if hull_pts is not None:
                    poly = Polygon([(x,y) for y,x in hull_pts])
                    proj_coords = [Transformer.from_crs(4326,3857,always_xy=True).transform(x,y) for x,y in hull_pts]
                    proj_poly = Polygon(proj_coords)
                    area = proj_poly.area / 1e6
                    mcp_results[aid][pct] = {"polygon": poly, "area": area}

# ─── KDE FUNCTIONS ─────────────────────────────────────────────────────────────
def kde_home_range(latitudes, longitudes, percent=95, grid_size=200):
    lon0,lat0 = np.mean(longitudes), np.mean(latitudes)
    zone = int((lon0 + 180) // 6) + 1
    epsg = 32600+zone if lat0>=0 else 32700+zone
    to_utm = Transformer.from_crs(4326, epsg, always_xy=True)
    to_ll  = Transformer.from_crs(epsg, 4326, always_xy=True)
    x,y = to_utm.transform(longitudes, latitudes)
    xy = np.vstack([x,y]).T
    n=xy.shape[0]
    if n>1:
        stds=np.std(xy,axis=0,ddof=1)
        h=max(30, np.mean(stds)*(4/(3*n))**0.2)
    else:
        h=30
    margin=3*h
    xmin,xmax = x.min()-margin, x.max()+margin
    ymin,ymax = y.min()-margin, y.max()+margin
    xs = np.linspace(xmin,xmax,grid_size)
    ys = np.linspace(ymin,ymax,grid_size)
    X,Y = np.meshgrid(xs,ys)
    grid = np.vstack([X.ravel(),Y.ravel()]).T
    kde = KernelDensity(bandwidth=h,kernel="gaussian").fit(xy)
    Z = np.exp(kde.score_samples(grid)).reshape(X.shape)
    cell = (xs[1]-xs[0])*(ys[1]-ys[0])
    Z /= Z.sum()*cell
    flat = Z.flatten()
    idx = np.argsort(flat)[::-1]
    thr = flat[idx][np.searchsorted(flat[idx]*cell, percent/100.0)]
    mask = Z>=thr
    Zm = np.where(mask,Z,0)
    tot = Zm.sum()*cell
    if tot>0:
        Zm /= tot
    contours = measure.find_contours(mask.astype(float),0.5)
    polys=[]
    for c in contours:
        ux = np.interp(c[:,1], np.arange(grid_size), xs)
        uy = np.interp(c[:,0], np.arange(grid_size), ys)
        p = Polygon(zip(ux,uy)).buffer(0)
        if p.is_valid and p.area>0: polys.append(p)
    if not polys:
        return None,None,None,None,None,None,None
    from shapely.ops import unary_union
    mpoly = unary_union(polys)
    def to_latlon(poly):
        if poly.is_empty: return None
        if isinstance(poly, Polygon):
            ex,ey = to_ll.transform(*poly.exterior.xy)
            ins = [list(zip(*to_ll.transform(*interior.xy))) for interior in poly.interiors]
            return Polygon(list(zip(ex,ey)), ins)
        return MultiPolygon([to_latlon(p) for p in poly.geoms])
    latpoly = to_latlon(mpoly)
    area = mpoly.area/1e6
    tiff = tempfile.mktemp(suffix=f"_kde_{percent}.tif",dir="outputs")
    sw = to_ll.transform(xmin,ymin)
    ne = to_ll.transform(xmax,ymax)
    with rasterio.open(
        tiff,"w",
        driver="GTiff",
        height=Zm.shape[0],width=Zm.shape[1],
        count=1,dtype=Zm.dtype,
        crs="EPSG:4326",
        transform=from_origin(sw[0],ne[1],(ne[0]-sw[0])/grid_size,(ne[1]-sw[1])/grid_size)
    ) as dst:
        dst.write(np.flipud(Zm),1)
    geojs = tempfile.mktemp(suffix=f"_kde_{percent}.geojson",dir="outputs")
    with open(geojs,"w") as f:
        json.dump(mapping(latpoly),f)
    return latpoly,area,tiff,geojs,Zm,None,None

def add_kdes(df, percent_list):
    global kde_results
    os.makedirs("outputs",exist_ok=True)
    for pct in percent_list:
        for aid in df["animal_id"].unique():
            if aid not in kde_results:
                kde_results[aid] = {}
            if pct not in kde_results[aid]:
                res = kde_home_range(df["latitude"].values, df["longitude"].values, pct)
                if res[0] is not None:
                    kde_results[aid][pct] = {
                        "contour":res[0],
                        "area":res[1],
                        "geotiff":res[2],
                        "geojson":res[3]
                    }

# ─── CHAT HANDLER ─────────────────────────────────────────────────────────────
def handle_chat(chat_history, user_message):
    global cached_df, mcp_results, kde_results, requested_percents, requested_kde_percents
    chat_history = list(chat_history)
    tool, _ = ask_llm(chat_history, user_message)
    mcp_list, kde_list = [], []
    if tool and tool.get("tool")=="home_range":
        lvl = tool.get("levels",[95])
        if tool.get("method")=="mcp":  mcp_list = lvl
        else:                          kde_list = lvl
    if "mcp" in user_message.lower(): mcp_list = parse_levels_from_text(user_message)
    if "kde" in user_message.lower(): kde_list = parse_levels_from_text(user_message)
    if not mcp_list and not kde_list: mcp_list=[95]

    if cached_df is None or "latitude" not in cached_df or "longitude" not in cached_df:
        chat_history.append({"role":"assistant","content":"CSV must be uploaded with 'latitude' and 'longitude' columns."})
        return chat_history, gr.update(), gr.update(visible=False)

    if mcp_list:
        add_mcps(cached_df, mcp_list); requested_percents.update(mcp_list)
    if kde_list:
        add_kdes(cached_df, kde_list); requested_kde_percents.update(kde_list)

    # build map
    df = cached_df
    m = folium.Map(location=[df["latitude"].mean(),df["longitude"].mean()],zoom_start=9)
    folium.TileLayer("OpenStreetMap").add_to(m)
    folium.TileLayer("CartoDB positron",attr='CartoDB').add_to(m)
    folium.TileLayer(tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",attr="OpenTopoMap").add_to(m)
    folium.TileLayer(tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",attr="Esri").add_to(m)

    # points & tracks
    pts=folium.FeatureGroup(name="Points",show=True)
    lines=folium.FeatureGroup(name="Tracks",show=True)
    cmap={}
    for aid in df["animal_id"].unique():
        cmap[aid]=f"#{random.randint(0,0xFFFFFF):06x}"
        track=df[df["animal_id"]==aid]
        if "timestamp" in track.columns:
            t_sorted=track.sort_values("timestamp")
            folium.PolyLine(list(zip(t_sorted["latitude"],t_sorted["longitude"])),color=cmap[aid],weight=2.5,opacity=0.8).add_to(lines)
        for row in track.itertuples():
            folium.CircleMarker(location=[row.latitude,row.longitude],radius=3,color=cmap[aid],fill=True,fill_opacity=0.7,popup=f"{aid}").add_to(pts)
    pts.add_to(m); lines.add_to(m)

    # MCPs
    for pct in requested_percents:
        for aid in df["animal_id"].unique():
            if aid in mcp_results and pct in mcp_results[aid]:
                v=mcp_results[aid][pct]
                fg=folium.FeatureGroup(name=f"{aid} MCP {pct}%",show=True)
                folium.Polygon(locations=[(y,x) for x,y in np.array(v["polygon"].exterior.coords)],
                               color=cmap[aid],fill=True,fill_opacity=0.15+0.15*(pct/100)).add_to(fg)
                m.add_child(fg)

    # KDE
    for pct in requested_kde_percents:
        for aid in df["animal_id"].unique():
            if aid in kde_results and pct in kde_results[aid]:
                v=kde_results[aid][pct]
                # raster
                fg1=folium.FeatureGroup(name=f"{aid} KDE {pct}% Raster",show=True)
                with rasterio.open(v["geotiff"]) as src:
                    arr=src.read(1); norm=(arr-arr.min())/(arr.max()-arr.min()+1e-10)
                    rgba=(plt.get_cmap('plasma')(norm)*255).astype(np.uint8)
                    b=src.bounds
                    img=np.dstack([rgba[:,:,0],rgba[:,:,1],rgba[:,:,2],(rgba[:,:,3]*0.7).astype(np.uint8)])
                    folium.raster_layers.ImageOverlay(image=img,bounds=[[b.bottom,b.left],[b.top,b.right]],opacity=0.7).add_to(fg1)
                m.add_child(fg1)
                # contour
                fg2=folium.FeatureGroup(name=f"{aid} KDE {pct}% Contour",show=True)
                c=v["contour"]
                if isinstance(c,MultiPolygon):
                    for poly in c.geoms:
                        folium.Polygon(locations=[(y,x) for x,y in poly.exterior.coords],color=cmap[aid],fill=True,fill_opacity=0.2).add_to(fg2)
                elif isinstance(c,Polygon):
                    folium.Polygon(locations=[(y,x) for x,y in c.exterior.coords],color=cmap[aid],fill=True,fill_opacity=0.2).add_to(fg2)
                m.add_child(fg2)

    folium.LayerControl(collapsed=False).add_to(m)
    m = fit_map_to_bounds(m, df)
    map_html = m._repr_html_()

    chat_history.append({"role":"user","content":user_message})
    chat_history.append({"role":"assistant","content":"Home ranges calculated. Download below."})

    # ─── Write & return ZIP path so DownloadButton works immediately ─────────────
    archive_path = save_all_mcps_zip()
    return (
        chat_history,
        gr.update(value=map_html),
        gr.update(value=archive_path, visible=True)
    )

# ─── ZIP RESULTS ───────────────────────────────────────────────────────────────
def save_all_mcps_zip():
    outputs_dir = "outputs"
    os.makedirs(outputs_dir, exist_ok=True)

    archive = os.path.join(outputs_dir, "spatchat_results.zip")
    if os.path.exists(archive):
        os.remove(archive)

    # 1) MCP GeoJSON
    features = []
    for aid, percs in mcp_results.items():
        for pct, v in percs.items():
            features.append({
                "type":"Feature",
                "properties":{"animal_id":aid,"percent":pct,"area_km2":v["area"]},
                "geometry":mapping(v["polygon"])
            })
    if features:
        with open(os.path.join(outputs_dir,"mcps_all.geojson"),"w") as f:
            json.dump({"type":"FeatureCollection","features":features},f)

    # 2) summary CSV
    rows = []
    for aid, percs in mcp_results.items():
        for pct, v in percs.items():
            rows.append((aid,f"MCP-{pct}",v["area"]))
    for aid, percs in kde_results.items():
        for pct, v in percs.items():
            rows.append((aid,f"KDE-{pct}",v["area"]))
    if rows:
        pd.DataFrame(rows,columns=["animal_id","type","area_km2"])\
          .to_csv(os.path.join(outputs_dir,"home_range_areas.csv"),index=False)

    # 3) zip
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as zf:
        for root,_,files in os.walk(outputs_dir):
            for fname in files:
                if fname.endswith(".zip"):
                    continue
                src = os.path.join(root,fname)
                arc = os.path.relpath(src, outputs_dir)
                zf.write(src, arcname=arc)

    return archive

# ─── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="SpatChat: Home Range Analysis") as demo:
    gr.Image(
        value="logo_long1.png", show_label=False,
        show_download_button=False, show_share_button=False,
        type="filepath", elem_id="logo-img"
    )
    gr.HTML("""
    <style>
      #logo-img img { height:90px; margin:10px 50px 10px 10px; border-radius:6px; }
    </style>
    """)
    gr.Markdown("## 🏠 SpatChat: Home Range Analysis {hr}  🦊🦉🐢")
    gr.HTML("""
    <div style="margin-top:-10px;margin-bottom:15px;">
      <input id="shareLink" type="text" value="https://spatchat.org/browse/?room=hr" readonly
             style="width:50%;padding:5px;background:#f8f8f8;color:#222;font-weight:500;border:1px solid #ccc;border-radius:4px;">
      <button onclick="navigator.clipboard.writeText(document.getElementById('shareLink').value)"
              style="padding:5px 10px;background:#007BFF;color:white;border:none;border-radius:4px;cursor:pointer;">
        📋 Copy Share Link
      </button>
      <div style="margin-top:10px;font-size:14px;">
        <b>Share:</b>
        <a href="https://twitter.com/intent/tweet?text=Checkout+SpatChat!&url=https://spatchat.org/browse/?room=hr" target="_blank">🐦 Twitter</a> |
        <a href="https://facebook.com/sharer/sharer.php?u=https://spatchat.org/browse/?room=hr" target="_blank">📘 Facebook</a>
      </div>
    </div>
    """)
    gr.Markdown("""
    <div style="font-size:14px;">
      © 2025 Ho Yi Wan & Logan Hysen. All rights reserved.<br>
      If you use SpatChat in research, please cite:<br>
      <b>Wan, H.Y.</b> & <b>Hysen, L.</b> (2025). <i>SpatChat: Home Range Analysis.</i>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(
                label="SpatChat", show_label=True,
                type="messages",
                value=[{"role":"assistant","content":"Welcome! Upload a CSV of coordinates to begin."}]
            )
            user_input = gr.Textbox(label="Ask SpatChat", placeholder="Type commands…", lines=1)
            file_input  = gr.File(label="Upload Movement CSV (.csv/.txt)", file_types=[".csv","txt"])
            x_col       = gr.Dropdown(label="X column",   choices=[], visible=False)
            y_col       = gr.Dropdown(label="Y column",   choices=[], visible=False)
            crs_input   = gr.Textbox(label="CRS (e.g. '33N')", visible=False)
            confirm_btn = gr.Button("Confirm Coordinate Settings", visible=False)
        with gr.Column(scale=3):
            map_output = gr.HTML(label="Map Preview", value=render_empty_map(), show_label=False)
            download_btn = gr.DownloadButton(
                "📥 Download Results",
                save_all_mcps_zip,
                visible=False
            )

    file_input.change(
        fn=handle_upload_initial,
        inputs=file_input,
        outputs=[chatbot, x_col, y_col, crs_input, map_output,
                 x_col, y_col, crs_input, confirm_btn, download_btn]
    )
    confirm_btn.click(
        fn=handle_upload_confirm,
        inputs=[x_col, y_col, crs_input],
        outputs=map_output
    )
    user_input.submit(
        fn=handle_chat,
        inputs=[chatbot, user_input],
        outputs=[chatbot, map_output, download_btn]
    )
    user_input.submit(lambda *args: "", None, user_input)

demo.launch(ssr_mode=False)
