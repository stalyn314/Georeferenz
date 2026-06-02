# 🗺️ GeoRef Web — Georreferenciador Visual de Imágenes

Aplicación web para georreferenciar imágenes por superposición visual y puntos de control (GCPs), sin depender de ArcMap o QGIS para la tarea básica de asignar coordenadas a una imagen.

## ¿Qué hace?

1. **Subes tu imagen** (PNG, JPG, TIFF — sin coordenadas)
2. **Haces clic** en un punto reconocible de la imagen
3. **Haces clic** en ese mismo punto en el mapa web
4. Repites al menos 3 veces
5. La app **calcula la transformación afín** (mínimos cuadrados) y muestra el overlay transparente en el mapa
6. **Descargas** el polígono de extensión en GeoJSON o un World File (.pgw) para usar en QGIS/ArcGIS

---

## 🚀 Cómo ejecutar localmente

```bash
# 1. Clonar el repositorio
git clone https://github.com/TU_USUARIO/georef-app.git
cd georef-app

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar
streamlit run app.py
```

La app abrirá en `http://localhost:8501`

---

## ☁️ Desplegar en Streamlit Community Cloud (gratis)

1. Sube este repositorio a GitHub (puede ser público o privado)
2. Ve a [share.streamlit.io](https://share.streamlit.io)
3. Conecta tu cuenta de GitHub
4. Elige el repo → `app.py` como archivo principal
5. Haz clic en **Deploy** — listo en ~2 minutos

---

## 📦 Dependencias

| Paquete | Para qué |
|---------|----------|
| `streamlit` | Framework web |
| `streamlit-folium` | Mapa interactivo con Leaflet |
| `streamlit-image-coordinates` | Captura de clics en la imagen |
| `folium` | Generación del mapa |
| `numpy` | Álgebra lineal (transformación afín) |
| `Pillow` | Lectura y conversión de imágenes |
| `opencv-python-headless` | Dibujo de marcadores sobre la imagen |

---

## 🗺️ Flujo de trabajo detallado

```
Subir imagen
     │
     ▼
Clic en imagen ──────────────────────────────────┐
(punto reconocible)                              │
     │                                           │
     ▼                                           │
Clic en mapa                                     │
(mismo punto en la realidad)                     │
     │                                           │
     ▼                                           │
GCP guardado ────────────────── < 3 GCPs ────────┘
     │
     │ ≥ 3 GCPs
     ▼
Transformación afín calculada
(mínimos cuadrados: px,py → lat,lon)
     │
     ▼
Overlay en el mapa (transparente, ajustable)
     │
     ▼
Descargar:
  ├── Polígono GeoJSON (extensión de la imagen)
  ├── World File .pgw (para QGIS/ArcGIS)
  └── GCPs como CSV
```

---

## 📐 Transformación utilizada

Se aplica una **transformación afín 2D por mínimos cuadrados**:

```
lon = a·px + b·py + c
lat = d·px + e·py + f
```

Con ≥ 3 puntos el sistema está determinado; con más puntos se minimiza el error cuadrático (RMS residual visible en la app).

Para transformaciones no lineales (imágenes muy distorsionadas), se recomienda usar la función *Thin Plate Spline* disponible como extensión futura.

---

## 🔧 Outputs explicados

### Polígono GeoJSON
Las 4 esquinas de la imagen transformadas a coordenadas geográficas. Úsalo para:
- Recortar capas en QGIS
- Definir el área de estudio
- Cargar en geojson.io o Google Earth

### World File (.pgw)
Seis parámetros de la transformación afín. Si colocas el `.pgw` junto al PNG original, QGIS y ArcGIS lo abrirán directamente como capa ráster georreferenciada.

### CSV de GCPs
Tabla con los puntos de control para reusar en QGIS Georeferencer o documentación.

---

## ⚙️ Limitaciones conocidas

- La transformación afín asume que la imagen no tiene distorsión de lente severa. Para imágenes muy distorsionadas (fotos aéreas oblicuas), el RMS residual será alto — se recomienda agregar más GCPs o usar ortofotografía.
- El overlay en el mapa es una caja alineada al norte (bounding box rectangular). La imagen rotada se visualiza correctamente pero el overlay de Folium no soporta rotación arbitraria sin una capa custom.
- Tamaño máximo de upload: 200 MB (configurable en `.streamlit/config.toml`).

---

## 📄 Licencia

MIT — úsalo y modifícalo libremente.
