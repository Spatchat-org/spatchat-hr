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
import re
from sklearn.neighbors import KernelDensity
from skimage import measure

print("Starting SpatChat (multi-MCP/KDE, chat-driven ZIP version)")

# ===== SESSION STATE =====
mcp_results = {}
kde_results = {}
requested_percents = set()
requested_kde_percents = set()
cached_df = None
cached_headers = []

def clear_all_results():
    global mcp_results, kde_results, requested_percents, requested_kde_percents, cached_df, cached_headers
    mcp_results = {}
    kde_results = {}
    requested_percents = set()
    requested_kde_percents = set()
    cached_df = None
    cached_headers = []
    if os.path.exists("outputs"):
        shutil.rmtree("outputs")
    os.makedirs("outputs", exist_ok=True)

# ========== LLM SETUP ==========
load_dotenv()
client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
SYSTEM_PROMPT = """
You are SpatChat, an expert wildlife home range analysis assistant.
If the user asks for a home range calculation (MCP or KDE), reply ONLY in JSON:
{"tool":"home_range","method":"mcp","levels":[95,50]}
or
{"tool":"home_range","method":"kde","levels":[95,50]}
For any other questions, answer naturally in ≤3 sentences.
""".strip()
FALLBACK_PROMPT = """
You are SpatChat, a wildlife movement expert.
If you can’t map a request to the home-range tool, just answer naturally (≤3 sentences).
""".strip()

def ask_llm(history, user_input):
    msgs = [{"role":"system","content":SYSTEM_PROMPT}] + history + [{"role":"user","content":user_input}]
    resp = client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        messages=msgs, temperature=0.0
    ).choices[0].message.content
    try:
        return json.loads(resp), resp
    except:
        conv = client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
            messages=[{"role":"system","content":FALLBACK_PROMPT}] + msgs,
            temperature=0.7
        ).choices[0].message.content
        return None, conv

# ========== UTILITIES ==========
def render_empty_map():
    m = folium.Map(location=[0,0], zoom_start=2, control_scale=True)
    folium.TileLayer("OpenStreetMap").add_to(m)
    folium.TileLayer("CartoDB positron", name="CartoDB").add_to(m)
    folium.TileLayer(
        tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        name="Topographic"
    ).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        name="Satellite"
    ).add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    return m._repr_html_()

def fit_map_to_bounds(m, df):
    min_lat, max_lat = df.latitude.min(), df.latitude.max()
    min_lon, max_lon = df.longitude.min(), df.longitude.max()
    if np.isfinite([min_lat,max_lat,min_lon,max_lon]).all():
        m.fit_bounds([[min_lat,min_lon],[max_lat,max_lon]])
    return m

def parse_levels_from_text(text):
    lvls = [int(v) for v in re.findall(r"\b([1-9][0-9]?|100)\b", text)]
    return sorted(set(lvls)) if lvls else [95]

# ========== MCP & KDE ==========
def mcp_polygon(lats, lons, pct):
    pts = np.column_stack((lons,lats))
    if len(pts)<3: return None
    ctr = pts.mean(axis=0)
    d = np.linalg.norm(pts-ctr,axis=1)
    k = max(3, int(len(pts)*(pct/100)))
    keep = pts[np.argsort(d)[:k]]
    if len(keep)<3: return None
    hull = ConvexHull(keep)
    return keep[hull.vertices]

def add_mcps(df, pct_list):
    global mcp_results
    for pct in pct_list:
        for aid in df.animal_id.unique():
            mcp_results.setdefault(aid,{})
            if pct not in mcp_results[aid]:
                track = df[df.animal_id==aid]
                hull_pts = mcp_polygon(track.latitude.values, track.longitude.values, pct)
                if hull_pts is not None:
                    poly = Polygon([(x,y) for y,x in hull_pts])
                    trans = Transformer.from_crs("epsg:4326","epsg:3857",always_xy=True)
                    wm = [trans.transform(x,y) for y,x in hull_pts]
                    area_km2 = Polygon(wm).area/1e6
                    mcp_results[aid][pct] = {"polygon": poly, "area": area_km2}

def kde_home_range(lats,lons,pct,grid_size=200):
    lon0,lat0 = lons.mean(), lats.mean()
    zone = int((lon0+180)//6)+1
    epsg = 32600+zone if lat0>=0 else 32700+zone
    to_utm = Transformer.from_crs("epsg:4326",f"epsg:{epsg}",always_xy=True)
    to_ll = Transformer.from_crs(f"epsg:{epsg}","epsg:4326",always_xy=True)
    x,y = to_utm.transform(lons,lats)
    xy = np.vstack([x,y]).T
    n = len(xy)
    h = (np.mean(np.std(xy,axis=0,ddof=1))* (4/(3*n))**(1/5)) if n>1 else 30.0
    h = max(h,30.0)
    margin = 3*h
    xmin,xmax = x.min()-margin, x.max()+margin
    ymin,ymax = y.min()-margin, y.max()+margin
    xs = np.linspace(xmin,xmax,grid_size); ys = np.linspace(ymin,ymax,grid_size)
    X,Y = np.meshgrid(xs,ys)
    pts = np.vstack([X.ravel(),Y.ravel()]).T
    kde = KernelDensity(bandwidth=h).fit(xy)
    Z = np.exp(kde.score_samples(pts)).reshape(X.shape)
    cell = (xs[1]-xs[0])*(ys[1]-ys[0])
    Z /= (Z.sum()*cell)
    idx = np.argsort(Z.ravel())[::-1]
    csum = np.cumsum(Z.ravel()[idx]*cell)
    thr = Z.ravel()[idx][np.searchsorted(csum,pct/100)]
    mask = Z>=thr
    polys = []
    for cnt in measure.find_contours(mask.astype(float),0.5):
        px,py = cnt[:,1], cnt[:,0]
        ux = np.interp(px, np.arange(grid_size), xs)
        uy = np.interp(py, np.arange(grid_size), ys)
        p = Polygon(zip(ux,uy)).buffer(0)
        if p.is_valid and p.area>0: polys.append(p)
    if not polys: return None, None
    from shapely.ops import unary_union
    u = unary_union(polys)
    def to_latlon(poly):
        ex,ey = to_ll.transform(*poly.exterior.xy)
        ints = [to_ll.transform(*ring.xy) for ring in poly.interiors]
        return Polygon(list(zip(ex,ey)), [list(zip(ix,iy)) for ix,iy in ints])
    mpoly = to_latlon(u) if isinstance(u,Polygon) else MultiPolygon([to_latlon(p) for p in u.geoms])
    return mpoly, u.area/1e6

def add_kdes(df, pct_list):
    global kde_results
    os.makedirs("outputs", exist_ok=True)
    for pct in pct_list:
        for aid in df.animal_id.unique():
            kde_results.setdefault(aid,{})
            if pct not in kde_results[aid]:
                track = df[df.animal_id==aid]
                poly,area = kde_home_range(track.latitude.values, track.longitude.values, pct)
                if poly is not None:
                    geojson_fp = os.path.join("outputs",f"kde_{aid}_{pct}.geojson")
                    with open(geojson_fp,"w") as f:
                        json.dump(mapping(poly), f)
                    kde_results[aid][pct] = {"contour":poly, "area":area, "geojson":geojson_fp}

# ========== ZIP CREATOR ==========
def save_all_mcps_zip():
    os.makedirs("outputs", exist_ok=True)
    # MCP GeoJSON
    feats=[]
    for aid, percs in mcp_results.items():
        for pct,v in percs.items():
            feats.append({
              "type":"Feature",
              "properties":{"animal_id":aid,"percent":pct,"area_km2":v["area"]},
              "geometry":mapping(v["polygon"])
            })
    if feats:
        with open("outputs/mcps_all.geojson","w") as f:
            json.dump({"type":"FeatureCollection","features":feats},f)
    # Summary CSV
    rows=[]
    for aid, percs in mcp_results.items():
        for pct,v in percs.items():
            rows.append((aid,f"MCP-{pct}",v["area"]))
    for aid, percs in kde_results.items():
        for pct,v in percs.items():
            rows.append((aid,f"KDE-{pct}",v["area"]))
    if rows:
        pd.DataFrame(rows,columns=["animal_id","type","area_km2"]) \
          .to_csv("outputs/home_range_areas.csv",index=False)
    # Build zip
    archive="outputs/spatchat_results.zip"
    if os.path.exists(archive): os.remove(archive)
    with zipfile.ZipFile(archive,"w",zipfile.ZIP_DEFLATED) as zf:
        for root,_,files in os.walk("outputs"):
            for fn in files:
                if fn.endswith(".zip"): continue
                fp=os.path.join(root,fn)
                arc=os.path.relpath(fp,"outputs")
                zf.write(fp,arcname=arc)
    return archive

# ========== CALLBACKS ==========
def handle_upload_initial(file):
    global cached_df, cached_headers
    clear_all_results()
    dst=os.path.join("uploads",os.path.basename(file))
    os.makedirs("uploads", exist_ok=True)
    shutil.copy(file, dst)
    try:
        df=pd.read_csv(dst)
    except:
        return [{"role":"assistant","content":"❌ Failed to read CSV."}], "", render_empty_map()
    cached_df=df
    cached_headers=list(df.columns)
    # check for lat/lon
    cols_lower=[c.lower() for c in df.columns]
    if "latitude" in cols_lower and "longitude" in cols_lower:
        return [{"role":"assistant","content":"✅ CSV uploaded. Ready to compute home ranges."}], "", render_empty_map()
    else:
        return [{"role":"assistant","content":"❌ CSV uploaded but could not detect 'latitude' & 'longitude' columns."}], "", render_empty_map()

def handle_chat(history, user_msg):
    global cached_df
    chat_history=list(history)
    # reset?
    if re.search(r"\b(start over|restart|clear|reset)\b", user_msg, re.I):
        clear_all_results()
        return [{"role":"assistant","content":"🗑️ All cleared! Upload a new CSV to begin."}], gr.update(value=render_empty_map()), ""
    # parse intent
    tool,_ = ask_llm(chat_history, user_msg)
    mcp_list,kde_list=[],[]
    if tool and tool.get("tool")=="home_range":
        if tool.get("method")=="mcp": mcp_list=tool.get("levels",[95])
        if tool.get("method")=="kde": kde_list=tool.get("levels",[95])
    if "mcp" in user_msg.lower(): mcp_list=parse_levels_from_text(user_msg)
    if "kde" in user_msg.lower(): kde_list=parse_levels_from_text(user_msg)
    if not (mcp_list or kde_list): mcp_list=[95]
    if cached_df is None:
        chat_history.append({"role":"assistant","content":"⚠️ Please upload a CSV first."})
        return chat_history, gr.update(value=render_empty_map()), ""
    # compute
    if mcp_list:
        add_mcps(cached_df, mcp_list); requested_percents.update(mcp_list)
    if kde_list:
        add_kdes(cached_df, kde_list); requested_kde_percents.update(kde_list)
    # rebuild map
    df=cached_df
    m=folium.Map(location=[df.latitude.mean(),df.longitude.mean()],zoom_start=9,control_scale=True)
    folium.TileLayer("OpenStreetMap").add_to(m)
    folium.TileLayer("CartoDB positron",name="CartoDB").add_to(m)
    folium.TileLayer(
        tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
        name="Topographic"
    ).add_to(m)
    folium.TileLayer(
        tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        name="Satellite"
    ).add_to(m)
    fg_pts=folium.FeatureGroup("Points",True)
    for _,r in df.iterrows():
        folium.CircleMarker([r.latitude,r.longitude],radius=3,
                            color="blue",fill=True,fill_opacity=0.7).add_to(fg_pts)
    fg_pts.add_to(m)
    for pct in requested_percents:
        fg=folium.FeatureGroup(f"MCP {pct}%",True)
        for aid in df.animal_id.unique():
            v=mcp_results.get(aid,{}).get(pct)
            if v:
                coords=[(lat,lon) for lon,lat in v["polygon"].exterior.coords]
                folium.Polygon(coords, fill=True, color="red", fill_opacity=0.2).add_to(fg)
        fg.add_to(m)
    for pct in requested_kde_percents:
        fg=folium.FeatureGroup(f"KDE {pct}%",True)
        for aid in df.animal_id.unique():
            v=kde_results.get(aid,{}).get(pct)
            if v:
                coords=[(lat,lon) for lon,lat in v["contour"].exterior.coords]
                folium.Polygon(coords, fill=True, color="green", fill_opacity=0.2).add_to(fg)
        fg.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    m=fit_map_to_bounds(m,df)
    map_html=m._repr_html_()

    # build and return ZIP
    zip_fp=save_all_mcps_zip()

    # assistant message
    msgs=[]
    if mcp_list: msgs.append(f"MCPs ({','.join(map(str,mcp_list))}%) calculated.")
    if kde_list: msgs.append(f"KDEs ({','.join(map(str,kde_list))}%) calculated.")
    msgs.append("Download your results below.")
    chat_history.append({"role":"user","content":user_msg})
    chat_history.append({"role":"assistant","content":" ".join(msgs)})

    return chat_history, gr.update(value=map_html), zip_fp

# ========== UI ==========
with gr.Blocks(title="SpatChat: Home Range Analysis") as demo:
    gr.Image("logo_long1.png",show_label=False,type="filepath",elem_id="logo-img")
    gr.Markdown("## 🏠 SpatChat: Home Range Analysis")
    with gr.Row():
        with gr.Column(scale=2):
            chatbot   = gr.Chatbot(value=[{"role":"assistant","content":"Welcome! Upload a CSV of coordinates to begin."}])
            user_input= gr.Textbox(placeholder="Type commands…")
            file_input= gr.File(file_types=[".csv"])
        with gr.Column(scale=3):
            map_out   = gr.HTML(render_empty_map())
            download_btn = gr.DownloadButton(
                "📥 Download Results",
                save_all_mcps_zip,
                label="Download Results"
            )
    file_input.change(
        fn=handle_upload_initial,
        inputs=[file_input],
        outputs=[chatbot, user_input, map_out]
    )
    user_input.submit(
        fn=handle_chat,
        inputs=[chatbot, user_input],
        outputs=[chatbot, map_out, download_btn]
    )
    user_input.submit(lambda _: "", inputs=user_input, outputs=user_input)

demo.launch()
