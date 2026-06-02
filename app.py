"""
GeoRef Web App — Georreferenciador de imágenes por puntos de control
Sube una imagen, coloca GCPs en imagen + mapa, descarga el polígono.
"""

import streamlit as st
import folium
from streamlit_folium import st_folium
import numpy as np
from PIL import Image
import json
import base64
from io import BytesIO
import cv2

# ─── Configuración de página ───────────────────────────────────────────────────
st.set_page_config(
    page_title="GeoRef Web",
    page_icon="🗺️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS personalizado ─────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Tipografía y colores generales */
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; }
.stApp { background-color: #0f1117; color: #e8eaf0; }

/* Tarjetas de sección */
.section-card {
    background: #1a1d27;
    border: 1px solid #2d3142;
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 12px;
}

/* Badge de estado */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 12px;
    font-weight: 600;
    font-family: 'IBM Plex Mono', monospace;
}
.badge-pending { background: #3a3020; color: #f4a261; border: 1px solid #f4a261; }
.badge-ready   { background: #1a3020; color: #52b788; border: 1px solid #52b788; }
.badge-step    { background: #1a2035; color: #74b3ff; border: 1px solid #74b3ff; }

/* GCP list */
.gcp-row {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    padding: 5px 8px;
    background: #12141c;
    border-left: 3px solid #e63946;
    margin-bottom: 4px;
    border-radius: 0 4px 4px 0;
}

/* Metric override */
[data-testid="metric-container"] { background: #1a1d27; border-radius: 8px; padding: 12px; border: 1px solid #2d3142; }

/* Sidebar */
section[data-testid="stSidebar"] { background: #12141c !important; }
section[data-testid="stSidebar"] > div { padding-top: 16px; }

/* Botones */
.stButton > button {
    border-radius: 6px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    font-weight: 600;
}

/* Divider */
hr { border-color: #2d3142 !important; }
</style>
""", unsafe_allow_html=True)

# ─── Session State ─────────────────────────────────────────────────────────────
_defaults = {
    'gcps': [],              # [{'px':x, 'py':y, 'lat':lat, 'lon':lon}, ...]
    'pending_pixel': None,   # (x, y) en coordenadas reales de la imagen
    'last_map_click': None,  # "lat,lon" string del último clic procesado en el mapa
    'image': None,
    'img_w': 0,
    'img_h': 0,
    'img_name': '',
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─── Funciones de transformación ───────────────────────────────────────────────

def image_to_b64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def compute_affine(gcps: list):
    """
    Transformación afín por mínimos cuadrados: (px, py) → (lon, lat).
    Requiere al menos 3 GCPs.
    """
    src = np.array([[g['px'], g['py']] for g in gcps], dtype=float)
    lon = np.array([g['lon'] for g in gcps], dtype=float)
    lat = np.array([g['lat'] for g in gcps], dtype=float)
    A = np.column_stack([src, np.ones(len(gcps))])
    c_lon, *_ = np.linalg.lstsq(A, lon, rcond=None)
    c_lat, *_ = np.linalg.lstsq(A, lat, rcond=None)
    return c_lon, c_lat


def px2geo(px, py, c_lon, c_lat):
    """Convierte pixel → (lat, lon)."""
    v = np.array([px, py, 1.0])
    return float(np.dot(c_lat, v)), float(np.dot(c_lon, v))


def image_corners_geo(w, h, c_lon, c_lat):
    """Retorna las 4 esquinas de la imagen en coordenadas geográficas."""
    return [px2geo(x, y, c_lon, c_lat) for x, y in [(0,0),(w,0),(w,h),(0,h)]]


def build_polygon_geojson(w, h, c_lon, c_lat, img_name="imagen"):
    """Genera GeoJSON del polígono de extensión de la imagen."""
    corners = image_corners_geo(w, h, c_lon, c_lat)
    coords = [[lon, lat] for lat, lon in corners]
    coords.append(coords[0])   # cerrar polígono
    return {
        "type": "FeatureCollection",
        "features": [{
            "type": "Feature",
            "geometry": {"type": "Polygon", "coordinates": [coords]},
            "properties": {
                "nombre": img_name,
                "fuente": "GeoRef Web App",
                "n_gcps": len(st.session_state.gcps)
            }
        }]
    }


def build_world_file(c_lon, c_lat):
    """
    Genera world file (.pgw) con la transformación afín completa (6 parámetros).
    Compatible con QGIS, ArcGIS, GDAL.
    """
    # Parámetros afines: lon = a*px + b*py + c_x | lat = d*px + e*py + c_y
    # World file: a, d, b, e, lon_centro_pixel_tl, lat_centro_pixel_tl
    a = c_lon[0]  # pixel size X (lon/px en columna)
    b = c_lon[1]  # rotación (lon/px en fila)
    d = c_lat[0]  # rotación (lat/px en columna)
    e = c_lat[1]  # pixel size Y (lat/px en fila)
    cx = c_lon[2]  # lon origen (pixel 0,0)
    cy = c_lat[2]  # lat origen (pixel 0,0)
    return f"{a:.10f}\n{d:.10f}\n{b:.10f}\n{e:.10f}\n{cx:.10f}\n{cy:.10f}"


def draw_markers_on_image(img: Image.Image, gcps, pending,
                          display_w: int, display_h: int) -> Image.Image:
    """
    Redimensiona la imagen al tamaño de display y luego dibuja los marcadores.

    Bug corregido: la versión anterior dibujaba en coordenadas escaladas sobre
    la imagen en resolución ORIGINAL y luego la redimensionaba, lo que aplicaba
    el factor scale dos veces (scale²). Ahora:
      1. Primero resize a display_w × display_h
      2. Luego dibuja marcadores con coordenadas reales × (display / original)
    """
    scale_x = display_w / img.width
    scale_y = display_h / img.height

    # Resize primero → coordenadas de display son las definitivas
    disp = img.convert("RGB").resize((display_w, display_h), Image.LANCZOS)
    arr = np.array(disp)

    # Radio y grosor proporcionales al display (no cambian con el zoom del usuario)
    r  = max(8, int(display_w * 0.013))   # ~8-12 px según ancho
    th = max(2, int(display_w * 0.003))

    # GCPs confirmados — coordenadas reales × factor display
    for i, g in enumerate(gcps):
        cx = int(g['px'] * scale_x)
        cy = int(g['py'] * scale_y)
        cv2.circle(arr, (cx, cy), r, (230, 57, 70), -1)
        cv2.circle(arr, (cx, cy), r, (255, 255, 255), th)
        cv2.putText(arr, str(i + 1), (cx + r + 2, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 57, 70), 2,
                    cv2.LINE_AA)

    # Punto pendiente (imagen seleccionada, esperando clic en mapa)
    if pending:
        cx = int(pending[0] * scale_x)
        cy = int(pending[1] * scale_y)
        cv2.circle(arr, (cx, cy), r + 3, (82, 183, 136), th)
        cv2.drawMarker(arr, (cx, cy), (82, 183, 136),
                       cv2.MARKER_CROSS, r * 2 + 4, th, cv2.LINE_AA)

    return Image.fromarray(arr)


def compute_rms(gcps, c_lon, c_lat):
    """Residual RMS en metros."""
    residuals = []
    for g in gcps:
        lat_c, lon_c = px2geo(g['px'], g['py'], c_lon, c_lat)
        dy = (lat_c - g['lat']) * 111_319
        dx = (lon_c - g['lon']) * 111_319 * np.cos(np.radians(g['lat']))
        residuals.append(dx**2 + dy**2)
    return float(np.sqrt(np.mean(residuals)))


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🗺️ **GeoRef Web**")
    st.markdown("Georreferenciación por superposición visual")
    st.divider()

    # ── Subir imagen
    st.markdown("### 📁 Imagen")
    uploaded = st.file_uploader(
        "Formatos: PNG, JPG, TIFF, BMP",
        type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
        label_visibility="collapsed"
    )
    if uploaded:
        img_pil = Image.open(uploaded).convert("RGBA")
        st.session_state.image = img_pil
        st.session_state.img_w = img_pil.width
        st.session_state.img_h = img_pil.height
        st.session_state.img_name = uploaded.name
        st.success(f"✓ {uploaded.name}  ({img_pil.width} × {img_pil.height} px)")

    st.divider()

    # ── Configuración del mapa
    st.markdown("### 🌐 Centro del mapa")
    col_a, col_b = st.columns(2)
    with col_a:
        map_lat = st.number_input("Lat", value=-33.4500, format="%.4f", label_visibility="collapsed")
        st.caption("Latitud")
    with col_b:
        map_lon = st.number_input("Lon", value=-70.6700, format="%.4f", label_visibility="collapsed")
        st.caption("Longitud")
    map_zoom = st.slider("Zoom inicial", 1, 20, 13)
    map_tiles = st.selectbox(
        "Capa base",
        ["OpenStreetMap", "Satélite (ESRI)", "CartoDB Dark"]
    )

    st.divider()

    # ── Transparencia overlay
    st.markdown("### 🌫️ Overlay")
    overlay_opacity = st.slider("Opacidad de la imagen en el mapa", 0.05, 1.0, 0.55, 0.05)

    st.divider()

    # ── Lista de GCPs
    n_gcps = len(st.session_state.gcps)
    status_label = ("✅ Listo" if n_gcps >= 3
                    else f"⚠️ {n_gcps}/3 mínimo")
    st.markdown(f"### 📍 Puntos de control &nbsp;&nbsp;<small>{status_label}</small>",
                unsafe_allow_html=True)

    if st.session_state.gcps:
        for i, g in enumerate(st.session_state.gcps):
            c1, c2 = st.columns([6, 1])
            c1.markdown(
                f'<div class="gcp-row">'
                f'<b>P{i+1}</b> px({g["px"]:.0f},{g["py"]:.0f})<br>'
                f'{g["lat"]:.5f}, {g["lon"]:.5f}</div>',
                unsafe_allow_html=True
            )
            if c2.button("✕", key=f"del_{i}"):
                st.session_state.gcps.pop(i)
                st.rerun()
    else:
        st.caption("Sin puntos aún. Haz clic en la imagen y luego en el mapa.")

    col_r1, col_r2 = st.columns(2)
    if col_r1.button("🗑️ Limpiar GCPs", use_container_width=True):
        st.session_state.gcps.clear()
        st.session_state.pending_pixel = None
        st.rerun()
    if st.session_state.pending_pixel and col_r2.button("↩️ Cancelar punto", use_container_width=True):
        st.session_state.pending_pixel = None
        st.rerun()


# ─── MAIN ─────────────────────────────────────────────────────────────────────
if st.session_state.image is None:
    # Pantalla de bienvenida
    st.markdown("# 🗺️ GeoRef Web")
    st.markdown("### Georreferenciación visual — sin ArcMap, sin esperas")
    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **1. Sube tu imagen**
        Cualquier PNG, JPG o TIFF sin coordenadas.
        """)
    with col2:
        st.markdown("""
        **2. Coloca puntos de control**
        Clic en la imagen → clic en el punto
        equivalente en el mapa. Mínimo 3 pares.
        """)
    with col3:
        st.markdown("""
        **3. Descarga el resultado**
        Polígono GeoJSON + World File listos
        para QGIS, ArcGIS o Google Earth.
        """)
    st.info("👈 Sube una imagen en el panel lateral para comenzar.")
    st.stop()

# Variables locales
img: Image.Image = st.session_state.image
img_w: int       = st.session_state.img_w
img_h: int       = st.session_state.img_h
gcps: list       = st.session_state.gcps
n_gcps: int      = len(gcps)
transform_ready  = n_gcps >= 3

c_lon = c_lat = None
if transform_ready:
    c_lon, c_lat = compute_affine(gcps)

# ── Barra de estado del flujo ─────────────────────────────────────────────────
pending = st.session_state.pending_pixel
if pending is None:
    st.markdown(
        f'<span class="badge badge-step">PASO 1</span> '
        f'Haz clic en un punto reconocible de tu imagen (izquierda). '
        f'GCPs: <b>{n_gcps}</b>/3 mínimo.',
        unsafe_allow_html=True
    )
else:
    px0, py0 = pending
    st.markdown(
        f'<span class="badge badge-pending">PASO 2</span> '
        f'Punto en imagen seleccionado → <code>({px0:.0f}, {py0:.0f})</code>. '
        f'Ahora haz clic en ese mismo punto en el <b>mapa</b> (derecha) ➡️',
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── Dos columnas: imagen | mapa ────────────────────────────────────────────────
col_img, col_map = st.columns(2, gap="medium")

# ════════════════════════ PANEL IZQUIERDO: IMAGEN ════════════════════════════
with col_img:
    st.markdown("#### 🖼️ Imagen")

    # Tamaño de display: max 640 px en el lado más largo
    MAX_PX = 640
    scale = min(MAX_PX / img_w, MAX_PX / img_h, 1.0)
    display_w = int(img_w * scale)
    display_h = int(img_h * scale)

    # draw_markers_on_image ya hace el resize internamente (fix del bug scale²)
    disp = draw_markers_on_image(img, gcps, pending, display_w, display_h)

    try:
        from streamlit_image_coordinates import streamlit_image_coordinates
        # Coords devueltas en espacio de disp (0..display_w × 0..display_h)
        coords = streamlit_image_coordinates(disp, key="img_coords")

        if coords is not None:
            real_x = coords['x'] / scale   # display px → pixel original
            real_y = coords['y'] / scale
            # Solo registrar si el clic es diferente al pendiente actual
            if (st.session_state.pending_pixel is None or
                    abs(real_x - st.session_state.pending_pixel[0]) > 2 or
                    abs(real_y - st.session_state.pending_pixel[1]) > 2):
                st.session_state.pending_pixel = (real_x, real_y)
                st.rerun()

    except ImportError:
        st.error("Falta el paquete `streamlit-image-coordinates`. Ver README.")
        st.image(disp)

    if pending:
        st.markdown(
            f'<div class="section-card" style="border-color:#52b788">'
            f'✅ Pixel seleccionado: <code>({pending[0]:.0f}, {pending[1]:.0f})</code><br>'
            f'<small>Ahora haz clic en el mapa →</small></div>',
            unsafe_allow_html=True
        )

# ════════════════════════ PANEL DERECHO: MAPA ════════════════════════════════
with col_map:
    st.markdown("#### 🗺️ Mapa")

    # Seleccionar tile
    if map_tiles == "Satélite (ESRI)":
        tiles_url = (
            "https://server.arcgisonline.com/ArcGIS/rest/services/"
            "World_Imagery/MapServer/tile/{z}/{y}/{x}"
        )
        tiles_attr = "Esri | Source: USGS, NASA"
    elif map_tiles == "CartoDB Dark":
        tiles_url = (
            "https://cartodb-basemaps-{s}.global.ssl.fastly.net/"
            "dark_all/{z}/{x}/{y}.png"
        )
        tiles_attr = "© CartoDB, © OpenStreetMap contributors"
    else:
        tiles_url = "OpenStreetMap"
        tiles_attr = "OpenStreetMap"

    m = folium.Map(
        location=[map_lat, map_lon],
        zoom_start=map_zoom,
        tiles=tiles_url if tiles_url != "OpenStreetMap" else "OpenStreetMap",
        attr=tiles_attr,
    )

    # Overlay de la imagen (si hay transformación)
    if transform_ready and c_lon is not None and c_lat is not None:
        geos = image_corners_geo(img_w, img_h, c_lon, c_lat)
        lats_g = [g[0] for g in geos]
        lons_g = [g[1] for g in geos]
        bounds = [[min(lats_g), min(lons_g)], [max(lats_g), max(lons_g)]]

        folium.raster_layers.ImageOverlay(
            image=f"data:image/png;base64,{image_to_b64(img)}",
            bounds=bounds,
            opacity=overlay_opacity,
            name="Imagen georreferenciada",
            zindex=1,
        ).add_to(m)

        # Polígono de extensión
        poly_gj = build_polygon_geojson(img_w, img_h, c_lon, c_lat,
                                         st.session_state.img_name)
        folium.GeoJson(
            poly_gj,
            name="Extensión",
            style_function=lambda _: {
                "fillOpacity": 0,
                "color": "#e63946",
                "weight": 2,
                "dashArray": "6 4",
            },
        ).add_to(m)
        folium.LayerControl().add_to(m)

    # Markers de GCPs existentes
    for i, g in enumerate(gcps):
        folium.Marker(
            [g['lat'], g['lon']],
            popup=f"<b>GCP {i+1}</b><br>px({g['px']:.0f},{g['py']:.0f})<br>"
                  f"{g['lat']:.5f}, {g['lon']:.5f}",
            tooltip=f"P{i+1}",
            icon=folium.Icon(color="red", icon="map-pin", prefix="fa"),
        ).add_to(m)

    # Renderizar mapa
    map_data = st_folium(m, width="100%", height=520, key="folium_map",
                         returned_objects=["last_clicked"])

    # Procesar clic en el mapa
    lc = (map_data or {}).get("last_clicked")
    if lc and st.session_state.pending_pixel is not None:
        click_id = f"{lc['lat']:.8f},{lc['lng']:.8f}"
        if click_id != st.session_state.last_map_click:
            px0, py0 = st.session_state.pending_pixel
            gcps.append({
                'px': px0, 'py': py0,
                'lat': lc['lat'], 'lon': lc['lng']
            })
            st.session_state.last_map_click = click_id
            st.session_state.pending_pixel = None
            st.rerun()


# ─── SECCIÓN DE DESCARGA ──────────────────────────────────────────────────────
st.divider()
st.markdown("### 📥 Exportar resultado")

dl_geo, dl_wld, dl_csv = st.columns(3)

with dl_geo:
    if transform_ready:
        gj = build_polygon_geojson(img_w, img_h, c_lon, c_lat, st.session_state.img_name)
        st.download_button(
            label="⬇️ Polígono GeoJSON",
            data=json.dumps(gj, indent=2, ensure_ascii=False),
            file_name=f"{st.session_state.img_name.rsplit('.', 1)[0]}_poly.geojson",
            mime="application/geo+json",
            use_container_width=True,
            type="primary",
        )
        st.caption("Compatible con QGIS, ArcGIS, Google Earth, geojson.io")
    else:
        st.button("⬇️ Polígono GeoJSON", disabled=True, use_container_width=True)
        st.caption(f"Necesitas {3 - n_gcps} punto(s) más")

with dl_wld:
    if transform_ready:
        wld_content = build_world_file(c_lon, c_lat)
        base_name = st.session_state.img_name.rsplit('.', 1)[0]
        st.download_button(
            label="⬇️ World File (.pgw)",
            data=wld_content,
            file_name=f"{base_name}.pgw",
            mime="text/plain",
            use_container_width=True,
        )
        st.caption("Coloca junto a tu PNG para abrirlo georreferenciado en QGIS/ArcGIS")
    else:
        st.button("⬇️ World File (.pgw)", disabled=True, use_container_width=True)
        st.caption("Disponible con ≥3 GCPs")

with dl_csv:
    if n_gcps > 0:
        csv_rows = ["n,px,py,longitud,latitud"]
        for i, g in enumerate(gcps):
            csv_rows.append(
                f"{i+1},{g['px']:.2f},{g['py']:.2f},{g['lon']:.8f},{g['lat']:.8f}"
            )
        st.download_button(
            label="⬇️ GCPs como CSV",
            data="\n".join(csv_rows),
            file_name="puntos_control.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.caption("Para importar en QGIS Georeferencer o ArcMap")
    else:
        st.button("⬇️ GCPs como CSV", disabled=True, use_container_width=True)
        st.caption("Agrega al menos un GCP")


# ─── MÉTRICAS Y RESIDUALES ────────────────────────────────────────────────────
if transform_ready:
    st.divider()
    geos = image_corners_geo(img_w, img_h, c_lon, c_lat)
    lats_g = [g[0] for g in geos]
    lons_g = [g[1] for g in geos]
    lat_mean = np.mean(lats_g)

    lat_span_m = (max(lats_g) - min(lats_g)) * 111_319
    lon_span_m = (max(lons_g) - min(lons_g)) * 111_319 * np.cos(np.radians(lat_mean))

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("GCPs usados", n_gcps)
    m2.metric("Extensión N-S", f"{lat_span_m:,.0f} m")
    m3.metric("Extensión E-O", f"{lon_span_m:,.0f} m")

    if n_gcps > 3:
        rms = compute_rms(gcps, c_lon, c_lat)
        delta = "✓ Bueno" if rms < 5 else ("⚠ Revisar" if rms < 20 else "✗ Alto")
        m4.metric("RMS Residual", f"{rms:.2f} m", delta=delta,
                  delta_color="normal" if rms < 5 else "inverse")
    else:
        m4.metric("RMS Residual", "—", help="Agrega más de 3 GCPs para ver el error")

    st.success(
        f"✅ Imagen georreferenciada con {n_gcps} puntos. "
        "El overlay se muestra en el mapa con la opacidad configurada."
    )
