"""
GeoRef Web App — v2 con zoom en imagen
Cambios: zoom 1×/2×/4×/8× en panel imagen, pan con botones,
         thumbnail de navegación, conversión de coords corregida.
"""

import streamlit as st
import folium
from streamlit_folium import st_folium
import numpy as np
from PIL import Image
import json, base64
from io import BytesIO
import cv2

# ─── Config ───────────────────────────────────────────────────────────────────
st.set_page_config(page_title="GeoRef Web", page_icon="🗺️",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background-color: #0f1117; color: #e8eaf0; }
.section-card {
    background: #1a1d27; border: 1px solid #2d3142;
    border-radius: 10px; padding: 14px; margin-bottom: 10px;
}
.badge { display:inline-block; padding:3px 10px; border-radius:12px;
         font-size:12px; font-weight:600; font-family:'IBM Plex Mono',monospace; }
.badge-pending { background:#3a3020; color:#f4a261; border:1px solid #f4a261; }
.badge-step    { background:#1a2035; color:#74b3ff; border:1px solid #74b3ff; }
.gcp-row { font-family:'IBM Plex Mono',monospace; font-size:11px; padding:5px 8px;
           background:#12141c; border-left:3px solid #e63946;
           margin-bottom:4px; border-radius:0 4px 4px 0; }
.zoom-info { font-family:'IBM Plex Mono',monospace; font-size:11px;
             color:#74b3ff; padding:4px 8px; background:#1a2035;
             border-radius:6px; display:inline-block; }
section[data-testid="stSidebar"] { background:#12141c !important; }
[data-testid="metric-container"] { background:#1a1d27; border-radius:8px;
    padding:12px; border:1px solid #2d3142; }
hr { border-color:#2d3142 !important; }
/* Botones de zoom compactos */
.stButton>button { border-radius:6px; font-family:'IBM Plex Mono',monospace;
                   font-size:12px; font-weight:600; padding:4px 8px; }
</style>
""", unsafe_allow_html=True)

# ─── Session State ─────────────────────────────────────────────────────────────
_defaults = {
    'gcps': [],
    'pending_pixel': None,
    'last_map_click': None,
    'image': None, 'img_w': 0, 'img_h': 0, 'img_name': '',
    # Zoom / pan del panel imagen
    'zoom': 1,           # factor: 1, 2, 4, 8
    'view_cx': -1,       # -1 = no inicializado; en px originales
    'view_cy': -1,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─── Helpers matemáticos ───────────────────────────────────────────────────────

def image_to_b64(img: Image.Image) -> str:
    buf = BytesIO(); img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def compute_affine(gcps):
    src = np.array([[g['px'], g['py']] for g in gcps], dtype=float)
    lon = np.array([g['lon'] for g in gcps], dtype=float)
    lat = np.array([g['lat'] for g in gcps], dtype=float)
    A = np.column_stack([src, np.ones(len(gcps))])
    c_lon, *_ = np.linalg.lstsq(A, lon, rcond=None)
    c_lat, *_ = np.linalg.lstsq(A, lat, rcond=None)
    return c_lon, c_lat

def px2geo(px, py, c_lon, c_lat):
    v = np.array([px, py, 1.0])
    return float(np.dot(c_lat, v)), float(np.dot(c_lon, v))

def image_corners_geo(w, h, c_lon, c_lat):
    return [px2geo(x, y, c_lon, c_lat) for x, y in [(0,0),(w,0),(w,h),(0,h)]]

def build_polygon_geojson(w, h, c_lon, c_lat, name="imagen"):
    pts = image_corners_geo(w, h, c_lon, c_lat)
    coords = [[lon, lat] for lat, lon in pts]
    coords.append(coords[0])
    return {"type": "FeatureCollection", "features": [{"type": "Feature",
        "geometry": {"type": "Polygon", "coordinates": [coords]},
        "properties": {"nombre": name, "n_gcps": len(st.session_state.gcps)}}]}

def build_world_file(c_lon, c_lat):
    return (f"{c_lon[0]:.10f}\n{c_lat[0]:.10f}\n"
            f"{c_lon[1]:.10f}\n{c_lat[1]:.10f}\n"
            f"{c_lon[2]:.10f}\n{c_lat[2]:.10f}")

def compute_rms(gcps, c_lon, c_lat):
    res = []
    for g in gcps:
        lat_c, lon_c = px2geo(g['px'], g['py'], c_lon, c_lat)
        dy = (lat_c - g['lat']) * 111_319
        dx = (lon_c - g['lon']) * 111_319 * np.cos(np.radians(g['lat']))
        res.append(dx**2 + dy**2)
    return float(np.sqrt(np.mean(res)))

# ─── Helpers de visualización imagen ──────────────────────────────────────────

def get_crop_box(img_w, img_h, zoom, cx, cy):
    """
    Devuelve (x1,y1,x2,y2) del recorte en coordenadas de la imagen original.
    cx, cy: centro deseado del recorte en píxeles originales.
    """
    vis_w = img_w / zoom
    vis_h = img_h / zoom
    x1 = int(max(0, min(cx - vis_w / 2, img_w - vis_w)))
    y1 = int(max(0, min(cy - vis_h / 2, img_h - vis_h)))
    x2 = int(x1 + vis_w)
    y2 = int(y1 + vis_h)
    return x1, y1, min(x2, img_w), min(y2, img_h)

def draw_markers_on_crop(crop_img, gcps, pending,
                          display_w, display_h,
                          ox, oy, crop_w, crop_h):
    """
    Dibuja GCPs y punto pendiente sobre el recorte ya escalado al display.
    ox, oy        : origen del recorte en coords originales
    crop_w, crop_h: tamaño del recorte en coords originales
    """
    sx = display_w / crop_w
    sy = display_h / crop_h

    r  = max(7, int(min(display_w, display_h) * 0.016))
    th = max(2, r // 5)

    disp = crop_img.convert("RGB").resize((display_w, display_h), Image.LANCZOS)
    arr  = np.array(disp)

    for i, g in enumerate(gcps):
        cx_d = int((g['px'] - ox) * sx)
        cy_d = int((g['py'] - oy) * sy)
        # Solo dibujar si está (aproximadamente) en el viewport
        if -r*3 <= cx_d <= display_w + r*3 and -r*3 <= cy_d <= display_h + r*3:
            cx_d = int(np.clip(cx_d, 0, display_w - 1))
            cy_d = int(np.clip(cy_d, 0, display_h - 1))
            cv2.circle(arr, (cx_d, cy_d), r, (230, 57, 70), -1)
            cv2.circle(arr, (cx_d, cy_d), r, (255, 255, 255), th)
            cv2.putText(arr, str(i + 1), (cx_d + r + 2, cy_d + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 57, 70), 2, cv2.LINE_AA)

    if pending:
        cx_d = int((pending[0] - ox) * sx)
        cy_d = int((pending[1] - oy) * sy)
        if -r*3 <= cx_d <= display_w + r*3 and -r*3 <= cy_d <= display_h + r*3:
            cx_d = int(np.clip(cx_d, 0, display_w - 1))
            cy_d = int(np.clip(cy_d, 0, display_h - 1))
            cv2.circle(arr, (cx_d, cy_d), r + 4, (82, 183, 136), th + 1)
            cv2.drawMarker(arr, (cx_d, cy_d), (82, 183, 136),
                           cv2.MARKER_CROSS, (r + 4) * 2, th + 1, cv2.LINE_AA)

    return Image.fromarray(arr)

def make_thumbnail(img, crop_box, gcps, pending, thumb_w=170):
    """
    Genera minimap: imagen completa + rectángulo naranja (viewport) + puntos GCP.
    """
    aspect = img.height / img.width
    thumb_h = int(thumb_w * aspect)
    thumb = img.convert("RGB").resize((thumb_w, thumb_h), Image.LANCZOS)
    arr   = np.array(thumb)

    sx = thumb_w / img.width
    sy = thumb_h / img.height

    # Rectángulo del viewport
    x1, y1, x2, y2 = crop_box
    cv2.rectangle(arr, (int(x1*sx), int(y1*sy)), (int(x2*sx), int(y2*sy)),
                  (255, 140, 0), 2)

    # GCPs
    r_t = max(3, int(thumb_w * 0.022))
    for g in gcps:
        gx, gy = int(g['px'] * sx), int(g['py'] * sy)
        cv2.circle(arr, (gx, gy), r_t, (230, 57, 70), -1)

    # Punto pendiente
    if pending:
        px_t, py_t = int(pending[0] * sx), int(pending[1] * sy)
        cv2.circle(arr, (px_t, py_t), r_t + 2, (82, 183, 136), 2)

    return Image.fromarray(arr)


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🗺️ **GeoRef Web**")
    st.markdown("Georreferenciación por superposición visual")
    st.divider()

    st.markdown("### 📁 Imagen")
    uploaded = st.file_uploader("PNG, JPG, TIFF, BMP",
                                 type=["png","jpg","jpeg","tif","tiff","bmp"],
                                 label_visibility="collapsed")
    if uploaded:
        img_pil = Image.open(uploaded).convert("RGBA")
        new_name = uploaded.name
        if new_name != st.session_state.img_name:
            # Nueva imagen → resetear todo
            st.session_state.image    = img_pil
            st.session_state.img_w    = img_pil.width
            st.session_state.img_h    = img_pil.height
            st.session_state.img_name = new_name
            st.session_state.gcps.clear()
            st.session_state.pending_pixel = None
            st.session_state.zoom   = 1
            st.session_state.view_cx = img_pil.width  // 2
            st.session_state.view_cy = img_pil.height // 2
        st.success(f"✓ {new_name}  ({img_pil.width}×{img_pil.height} px)")

    st.divider()
    st.markdown("### 🌐 Centro del mapa")
    ca, cb = st.columns(2)
    map_lat  = ca.number_input("Lat",  value=-33.4500, format="%.4f",
                                label_visibility="collapsed")
    map_lon  = cb.number_input("Lon",  value=-70.6700, format="%.4f",
                                label_visibility="collapsed")
    ca.caption("Latitud"); cb.caption("Longitud")
    map_zoom  = st.slider("Zoom mapa", 1, 20, 13)
    map_tiles = st.selectbox("Capa base",
                              ["Satélite (ESRI)", "OpenStreetMap", "CartoDB Dark"])

    st.divider()
    overlay_opacity = st.slider("🌫️ Opacidad imagen en mapa", 0.05, 1.0, 0.55, 0.05)

    st.divider()
    n_gcps = len(st.session_state.gcps)
    st.markdown(f"### 📍 Puntos de control &nbsp;{'✅' if n_gcps>=3 else f'⚠️ {n_gcps}/3'}",
                unsafe_allow_html=True)
    if st.session_state.gcps:
        for i, g in enumerate(st.session_state.gcps):
            c1, c2 = st.columns([6, 1])
            c1.markdown(
                f'<div class="gcp-row"><b>P{i+1}</b> '
                f'px({g["px"]:.0f},{g["py"]:.0f})<br>'
                f'{g["lat"]:.5f}, {g["lon"]:.5f}</div>',
                unsafe_allow_html=True)
            if c2.button("✕", key=f"del_{i}"):
                st.session_state.gcps.pop(i); st.rerun()
    else:
        st.caption("Sin puntos. Clic imagen → clic mapa.")

    r1, r2 = st.columns(2)
    if r1.button("🗑️ Limpiar GCPs", use_container_width=True):
        st.session_state.gcps.clear()
        st.session_state.pending_pixel = None
        st.rerun()
    if st.session_state.pending_pixel and r2.button("↩️ Cancelar", use_container_width=True):
        st.session_state.pending_pixel = None; st.rerun()


# ─── GUARD ────────────────────────────────────────────────────────────────────
if st.session_state.image is None:
    st.markdown("# 🗺️ GeoRef Web")
    st.markdown("### Georreferenciación visual — sin ArcMap, sin esperas")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**1. Sube imagen** en el panel lateral")
    c2.markdown("**2. Haz zoom** en la imagen, clic en punto reconocible")
    c3.markdown("**3. Clic en el mapa** en el mismo punto real → GCP listo")
    st.info("👈 Sube una imagen para comenzar.")
    st.stop()

# Variables locales
img: Image.Image = st.session_state.image
img_w: int       = st.session_state.img_w
img_h: int       = st.session_state.img_h
gcps: list       = st.session_state.gcps
n_gcps: int      = len(gcps)
transform_ready  = n_gcps >= 3
pending          = st.session_state.pending_pixel

# Inicializar centro de vista si es la primera vez
if st.session_state.view_cx == -1:
    st.session_state.view_cx = img_w // 2
    st.session_state.view_cy = img_h // 2

c_lon = c_lat = None
if transform_ready:
    c_lon, c_lat = compute_affine(gcps)

# ─── Barra de estado ──────────────────────────────────────────────────────────
if pending is None:
    st.markdown(
        f'<span class="badge badge-step">PASO 1</span> '
        f'Usa el zoom de la imagen (izquierda) y haz clic en un punto reconocible. '
        f'GCPs: <b>{n_gcps}</b>/3 mínimo.',
        unsafe_allow_html=True)
else:
    px0, py0 = pending
    st.markdown(
        f'<span class="badge badge-pending">PASO 2</span> '
        f'Punto imagen: <code>({px0:.0f}, {py0:.0f})</code> → '
        f'Ahora haz clic en el <b>mapa</b> en ese mismo lugar ➡️',
        unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── DOS COLUMNAS ─────────────────────────────────────────────────────────────
col_img, col_map = st.columns(2, gap="medium")

# ════════════════════════ PANEL IMAGEN ════════════════════════════════════════
with col_img:
    st.markdown("#### 🖼️ Imagen")

    zoom    = st.session_state.zoom
    view_cx = st.session_state.view_cx
    view_cy = st.session_state.view_cy

    # Display size fijo
    DISP_W = 620
    DISP_H = int(DISP_W * img_h / img_w)

    # Caja de recorte
    x1, y1, x2, y2 = get_crop_box(img_w, img_h, zoom, view_cx, view_cy)
    crop_w = x2 - x1
    crop_h = y2 - y1

    # Recorte de la imagen
    crop_img = img.crop((x1, y1, x2, y2))

    # Dibujar marcadores y renderizar
    disp = draw_markers_on_crop(crop_img, gcps, pending,
                                 DISP_W, DISP_H,
                                 x1, y1, crop_w, crop_h)

    # ── Captura de clic ───────────────────────────────────────────────────────
    try:
        from streamlit_image_coordinates import streamlit_image_coordinates
        coords = streamlit_image_coordinates(disp, key="img_coords")

        if coords is not None:
            # Convertir coords display → píxel original
            real_x = x1 + coords['x'] * crop_w / DISP_W
            real_y = y1 + coords['y'] * crop_h / DISP_H

            # Centrar la vista en el punto clicado (para zooms posteriores)
            st.session_state.view_cx = real_x
            st.session_state.view_cy = real_y

            # Guardar como punto pendiente (solo si es diferente)
            if (st.session_state.pending_pixel is None or
                    abs(real_x - st.session_state.pending_pixel[0]) > 1 or
                    abs(real_y - st.session_state.pending_pixel[1]) > 1):
                st.session_state.pending_pixel = (real_x, real_y)
                st.rerun()

    except ImportError:
        st.error("Falta: pip install streamlit-image-coordinates")
        st.image(disp)

    # ── Info del punto pendiente ──────────────────────────────────────────────
    if pending:
        st.markdown(
            f'<div class="section-card" style="border-color:#52b788; padding:10px;">'
            f'✅ px seleccionado: <code>({pending[0]:.1f}, {pending[1]:.1f})</code>'
            f'&emsp;→ ahora clic en el mapa</div>',
            unsafe_allow_html=True)

    st.markdown("---")

    # ══ CONTROLES DE ZOOM ════════════════════════════════════════════════════
    st.markdown("**🔍 Zoom**")

    # Fila de zoom buttons
    zb = st.columns(4)
    zoom_levels = [1, 2, 4, 8]
    zoom_labels = ["1× (todo)", "2×", "4×", "8×"]
    for i, (lv, lb) in enumerate(zip(zoom_levels, zoom_labels)):
        btn_type = "primary" if lv == zoom else "secondary"
        if zb[i].button(lb, key=f"z{lv}", use_container_width=True, type=btn_type):
            st.session_state.zoom = lv
            # Al entrar en zoom, centrar en el punto pendiente (si hay) o en el centro
            if pending:
                st.session_state.view_cx, st.session_state.view_cy = pending
            elif st.session_state.view_cx == img_w // 2:
                pass  # mantener centro
            st.rerun()

    # Fila de pan (solo visible cuando zoom > 1)
    if zoom > 1:
        step_px = int(img_w / zoom * 0.35)  # 35 % del área visible

        st.markdown("**🧭 Navegar**")
        pb = st.columns(5)

        if pb[0].button("◀", key="pan_l", use_container_width=True):
            st.session_state.view_cx = max(img_w/zoom/2,
                                           view_cx - step_px)
            st.rerun()
        if pb[1].button("▲", key="pan_u", use_container_width=True):
            st.session_state.view_cy = max(img_h/zoom/2,
                                           view_cy - step_px)
            st.rerun()
        if pb[2].button("🏠", key="pan_home", use_container_width=True,
                         help="Centrar en imagen completa"):
            st.session_state.view_cx = img_w // 2
            st.session_state.view_cy = img_h // 2
            st.rerun()
        if pb[3].button("▼", key="pan_d", use_container_width=True):
            st.session_state.view_cy = min(img_h - img_h/zoom/2,
                                           view_cy + step_px)
            st.rerun()
        if pb[4].button("▶", key="pan_r", use_container_width=True):
            st.session_state.view_cx = min(img_w - img_w/zoom/2,
                                           view_cx + step_px)
            st.rerun()

        # Minimap + info de posición
        mini_col, info_col = st.columns([2, 3])
        with mini_col:
            thumb = make_thumbnail(img, (x1, y1, x2, y2), gcps, pending, thumb_w=160)
            st.image(thumb, caption="Mapa completo", use_column_width=False, width=160)
        with info_col:
            pct_x = int(view_cx / img_w * 100)
            pct_y = int(view_cy / img_h * 100)
            st.markdown(
                f'<div class="zoom-info">'
                f'Zoom: <b>{zoom}×</b><br>'
                f'Centro: ({view_cx:.0f}, {view_cy:.0f}) px<br>'
                f'Posición: {pct_x}% / {pct_y}%<br>'
                f'Visible: {crop_w}×{crop_h} px orig.</div>',
                unsafe_allow_html=True)
        
        # Tip
        st.caption(
            "💡 Haz clic en la imagen para seleccionar el punto "
            "y centrar el zoom ahí automáticamente.")


# ════════════════════════ PANEL MAPA ══════════════════════════════════════════
with col_map:
    st.markdown("#### 🗺️ Mapa")

    # Tile layer
    if map_tiles == "Satélite (ESRI)":
        tile_url  = ("https://server.arcgisonline.com/ArcGIS/rest/services/"
                     "World_Imagery/MapServer/tile/{z}/{y}/{x}")
        tile_attr = "Esri | USGS, NASA"
    elif map_tiles == "CartoDB Dark":
        tile_url  = ("https://cartodb-basemaps-{s}.global.ssl.fastly.net/"
                     "dark_all/{z}/{x}/{y}.png")
        tile_attr = "© CartoDB, © OpenStreetMap"
    else:
        tile_url  = "OpenStreetMap"
        tile_attr = "OpenStreetMap"

    m = folium.Map(location=[map_lat, map_lon], zoom_start=map_zoom,
                   tiles=tile_url if tile_url != "OpenStreetMap" else "OpenStreetMap",
                   attr=tile_attr)

    # Overlay imagen georreferenciada
    if transform_ready and c_lon is not None:
        geos   = image_corners_geo(img_w, img_h, c_lon, c_lat)
        lats_g = [g[0] for g in geos]
        lons_g = [g[1] for g in geos]
        bounds = [[min(lats_g), min(lons_g)], [max(lats_g), max(lons_g)]]

        folium.raster_layers.ImageOverlay(
            image=f"data:image/png;base64,{image_to_b64(img)}",
            bounds=bounds, opacity=overlay_opacity,
            name="Imagen", zindex=1).add_to(m)

        folium.GeoJson(
            build_polygon_geojson(img_w, img_h, c_lon, c_lat, st.session_state.img_name),
            name="Extensión",
            style_function=lambda _: {
                "fillOpacity": 0, "color": "#e63946",
                "weight": 2, "dashArray": "6 4"}
        ).add_to(m)
        folium.LayerControl().add_to(m)

    # Markers GCP existentes
    for i, g in enumerate(gcps):
        folium.Marker(
            [g['lat'], g['lon']],
            popup=(f"<b>GCP {i+1}</b><br>"
                   f"px({g['px']:.0f},{g['py']:.0f})<br>"
                   f"{g['lat']:.5f}, {g['lon']:.5f}"),
            tooltip=f"P{i+1}",
            icon=folium.Icon(color="red", icon="map-pin", prefix="fa")
        ).add_to(m)

    map_data = st_folium(m, width="100%", height=560, key="folium_map",
                          returned_objects=["last_clicked"])

    # Procesar clic en mapa
    lc = (map_data or {}).get("last_clicked")
    if lc and st.session_state.pending_pixel is not None:
        click_id = f"{lc['lat']:.8f},{lc['lng']:.8f}"
        if click_id != st.session_state.last_map_click:
            px0, py0 = st.session_state.pending_pixel
            gcps.append({'px': px0, 'py': py0,
                         'lat': lc['lat'], 'lon': lc['lng']})
            st.session_state.last_map_click = click_id
            st.session_state.pending_pixel  = None
            st.rerun()


# ─── DESCARGA ─────────────────────────────────────────────────────────────────
st.divider()
st.markdown("### 📥 Exportar")
dl1, dl2, dl3 = st.columns(3)

with dl1:
    if transform_ready:
        gj = build_polygon_geojson(img_w, img_h, c_lon, c_lat, st.session_state.img_name)
        st.download_button("⬇️ Polígono GeoJSON",
            json.dumps(gj, indent=2, ensure_ascii=False),
            file_name=f"{st.session_state.img_name.rsplit('.',1)[0]}_poly.geojson",
            mime="application/geo+json", use_container_width=True, type="primary")
        st.caption("QGIS, ArcGIS, geojson.io")
    else:
        st.button("⬇️ Polígono GeoJSON", disabled=True, use_container_width=True)
        st.caption(f"Faltan {3-n_gcps} punto(s)")

with dl2:
    if transform_ready:
        st.download_button("⬇️ World File (.pgw)",
            build_world_file(c_lon, c_lat),
            file_name=f"{st.session_state.img_name.rsplit('.',1)[0]}.pgw",
            mime="text/plain", use_container_width=True)
        st.caption("Coloca junto al PNG en QGIS/ArcGIS")
    else:
        st.button("⬇️ World File (.pgw)", disabled=True, use_container_width=True)

with dl3:
    if n_gcps > 0:
        csv = "n,px,py,longitud,latitud\n" + "\n".join(
            f"{i+1},{g['px']:.2f},{g['py']:.2f},{g['lon']:.8f},{g['lat']:.8f}"
            for i, g in enumerate(gcps))
        st.download_button("⬇️ GCPs CSV",
            csv, file_name="puntos_control.csv",
            mime="text/csv", use_container_width=True)
    else:
        st.button("⬇️ GCPs CSV", disabled=True, use_container_width=True)


# ─── MÉTRICAS ─────────────────────────────────────────────────────────────────
if transform_ready:
    st.divider()
    geos   = image_corners_geo(img_w, img_h, c_lon, c_lat)
    lats_g = [g[0] for g in geos]; lons_g = [g[1] for g in geos]
    lat_m  = np.mean(lats_g)
    ns_m   = (max(lats_g) - min(lats_g)) * 111_319
    eo_m   = (max(lons_g) - min(lons_g)) * 111_319 * np.cos(np.radians(lat_m))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("GCPs", n_gcps)
    m2.metric("Extensión N-S", f"{ns_m:,.0f} m")
    m3.metric("Extensión E-O", f"{eo_m:,.0f} m")
    if n_gcps > 3:
        rms = compute_rms(gcps, c_lon, c_lat)
        m4.metric("RMS Residual", f"{rms:.2f} m",
                  delta="OK" if rms < 5 else "Revisar",
                  delta_color="normal" if rms < 5 else "inverse")
    else:
        m4.metric("RMS Residual", "—", help="Agrega >3 GCPs")

    st.success(f"✅ {n_gcps} GCPs — overlay visible en el mapa. "
               "Ajusta opacidad en el sidebar para verificar el alineamiento.")
