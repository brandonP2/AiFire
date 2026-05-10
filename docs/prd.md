# PRD — PetenFire

Sistema Integrado de Predicción y Detección de Incendios Forestales para el Petén, Guatemala.

---

## Problema

El departamento del Petén concentra la mayor superficie boscosa de Guatemala: la Reserva de Biosfera Maya (~2.1 M ha), el Parque Nacional Tikal y la Sierra del Lacandón. Cada año, especialmente entre **marzo y mayo** (estación seca), sufre pérdidas significativas por incendios forestales.

**Causa raíz del problema técnico:** Las herramientas disponibles para CONRED, INAB y CONAP son **reactivas**. La respuesta comienza cuando el incendio ya es visible desde tierra o cuando llega un reporte. No existe una herramienta operativa que:
- Anticipe dónde es más probable que ocurra un foco de ignición.
- Simule cómo se propagaría un incendio dado el viento, vegetación y topografía actuales.
- Confirme automáticamente focos activos usando imágenes satelitales recientes.

**Por qué ahora:** Los datos satelitales necesarios son completamente abiertos (NASA, Copernicus, USGS) y los modelos de ML necesarios son maduros y bien documentados. El costo de entrada es cero en datos y bajo en cómputo.

---

## Usuarios

**Primarios (operativo):**
- **CONRED** — toma decisiones de despliegue de brigadas y evacuación. Necesita: mapa de riesgo diario, simulador de propagación en tiempo casi real.
- **INAB** — planificación forestal. Necesita: tendencias de riesgo por zona, análisis de temporada.

**Secundarios (planificación y conservación):**
- **MARN** — evaluación de impacto ambiental.
- **CONAP** — manejo de áreas protegidas (Biosfera Maya, Tikal, Laguna del Tigre).
- **Municipalidades del Petén** — alerta temprana local.

**Terciarios (investigación y sociedad civil):**
- **WCS, Defensores de la Naturaleza** — monitoreo de cobertura forestal.
- **Investigadores** — estudios de ecología del fuego en Mesoamérica.

---

## Value Proposition

> CONRED recibe cada mañana un mapa de riesgo de ignición para el Petén, puede simular la propagación de cualquier foco activo en minutos, y confirma detecciones satelitales automáticamente — todo sin pagar por datos o licencias.

---

## Funcionalidades core (MVP)

### M1 — Predicción de Riesgo
- Mapa diario de probabilidad de ignición por celda de 1 km × 1 km para todo el Petén.
- Horizontes: 1, 3 y 7 días.
- Inputs: clima (temperatura, HR, viento, precipitación), NDVI, topografía, distancias a caminos/poblados, índice FWI.
- Modelo: XGBoost / LightGBM con calibración de probabilidades.

### M2 — Simulación de Propagación
- Dado un punto de ignición (lon, lat) y condiciones actuales, simula el avance del incendio a 6h, 12h y 24h.
- Output: raster de probabilidad de quema.
- Base: autómata celular simplificado tipo FARSITE + ML supervisado sobre polígonos históricos MODIS MCD64A1.

### M3 — Detección Visual Temprana
- Clasifica tiles de Sentinel-2 / Landsat-8/9 (bandas SWIR especialmente) como foco activo / no activo.
- Output: clasificación binaria + bounding box del foco.
- Modelo: CNN ligera (ResNet18 fine-tuned) o YOLOv8.
- Etiquetas: cruzadas automáticamente con NASA FIRMS.

### Dashboard integrado
- Aplicación Streamlit con tres tabs: mapa de riesgo M1, simulador M2, visor de detecciones M3.
- Mapa interactivo (folium) con capa de áreas protegidas superpuesta.

---

## Fuera de alcance (v1)

- Cobertura nacional (solo Petén en MVP).
- Predicción operativa en tiempo real con actualización automática horaria.
- Alertas SMS o push a dispositivos móviles.
- Integración directa con sistemas internos de CONRED.
- App móvil.
- Análisis de calidad del aire / humo.
- Modelo de impacto económico o de biodiversidad.

---

## Criterios de éxito

| Componente | Métrica | Meta MVP |
|------------|---------|----------|
| M1 | AUC-ROC | ≥ 0.80 |
| M1 | Recall en zonas de alto riesgo (P > 0.6) | ≥ 0.75 |
| M2 | Error de área quemada predicha vs. real | < 30% en eventos de prueba |
| M3 | Precisión en focos activos | ≥ 0.85 |
| M3 | Recall en focos activos | ≥ 0.80 |
| Producto | Tiempo de generación del mapa diario completo | < 15 min |
| Dashboard | Tiempo de carga del mapa de riesgo | < 5 s |

---

## Requisitos no funcionales

**Rendimiento:** El pipeline completo (descarga incremental + inferencia M1) debe completarse en < 15 minutos para un día de datos del Petén.

**Escala:** MVP para uso interno de 2–10 usuarios simultáneos en el dashboard. No requiere CDN ni escalado horizontal.

**Plataforma:** Web (Streamlit). Accesible desde navegador de escritorio en redes gubernamentales con ancho de banda limitado — minimizar assets pesados.

**Datos:**
- Datos crudos: almacenados localmente o en bucket privado (S3/GCS). Nunca en el repositorio.
- No se almacenan datos personales. Sin PII.
- Datos de entrenamiento: 2018–2024 (~7 años).

**Autenticación:** Sin autenticación en MVP (dashboard interno). Streamlit Sharing o despliegue en servidor con acceso por VPN si se requiere privacidad.

**Presupuesto:**
- Datos: $0 (todas las fuentes son abiertas).
- Cómputo de entrenamiento: instancia spot con GPU (~$2–5/h, 1–3 sesiones) para M3.
- Hosting del dashboard: servidor universitario / institucional o Streamlit Community Cloud (gratuito).
- APIs externas: NASA FIRMS, NASA POWER, USGS → gratuitas. Copernicus CDS → gratuita con registro.

**Reproducibilidad:** Cualquier investigador con acceso a las APIs abiertas debe poder reproducir el dataset y los modelos ejecutando los scripts del repositorio.

---

## Restricciones

- **Equipo:** 1 desarrollador principal + Claude Code como asistente.
- **Timeline:** 12 semanas.
- **Compute:** Sin GPU dedicada en desarrollo. Entrenamiento de M3 en instancia cloud spot o Google Colab Pro.
- **Dato faltante conocido:** FIRMS sub-reporta focos bajo nube densa. Mitigación: combinar MODIS Terra/Aqua + VIIRS Suomi-NPP/NOAA-20 + MCD64A1 como ground truth adicional.
- **Desbalance de clases extremo:** En M1, ~0.5–2% de celdas-día tienen fuego. Requiere estrategia de muestreo y métricas apropiadas (no usar accuracy).

---

## Fuentes de datos

| Dato | Fuente | Acceso | Cobertura temporal |
|------|--------|--------|--------------------|
| Focos activos | NASA FIRMS (MODIS + VIIRS) | API gratuita | 2000–presente |
| Burned area | MODIS MCD64A1 | NASA Earthdata | 2000–presente |
| Clima diario | NASA POWER | API REST gratuita | 1981–presente |
| Clima alta resolución | Copernicus ERA5 | CDS API (registro) | 1940–presente |
| NDVI mensual | MODIS MOD13Q1 | Earthdata / GEE | 2000–presente |
| Topografía | SRTM 30m | USGS | Estático |
| Uso de suelo | ESA WorldCover 2021 | Descarga directa | 2021 (estático) |
| Imágenes ópticas | Sentinel-2 L2A, Landsat-8/9 | Copernicus / GEE / STAC | 2015–presente |
| Caminos / poblados | OpenStreetMap | Geofabrik | Continuo |
| Áreas protegidas | CONAP / WDPA | Descarga directa | Estático |

---

## Cronograma

| Semana | Hito |
|--------|------|
| 1 | Setup, spikes de APIs, grilla base del Petén |
| 2–3 | EDA + construcción del dataset M1 |
| 4–5 | M1 baseline + iteración + calibración |
| 6–7 | M2 autómata celular + validación con MCD64A1 |
| 8–9 | M3 CNN/YOLOv8 + dataset de chips Sentinel-2 |
| 10–11 | Dashboard Streamlit integrado (M1 + M2 + M3) |
| 12 | Documentación, model cards, presentación a stakeholders |

---

## Riesgos y mitigaciones

| Riesgo | Probabilidad | Impacto | Mitigación |
|--------|-------------|---------|-----------|
| Sub-reporte FIRMS bajo nube | Alta | Medio | Combinar MODIS + VIIRS; usar MCD64A1 como ground truth alternativo |
| Desbalance de clases extremo en M1 | Alta | Alto | SMOTE conservador, focal loss, umbral adaptativo por zona |
| Falsos positivos costosos operativamente | Media | Alto | Calibración de probabilidades (Platt / isotónica), doble umbral (alerta / acción) |
| Volumen de imágenes Sentinel-2 | Alta | Medio | Google Earth Engine o STAC + Planetary Computer; chips 64×64 solo sobre zonas de interés |
| Cambio climático sesga el histórico | Media | Medio | Features dinámicas (anomalías respecto a media móvil), reentrenamiento anual |
| APIs externas caídas o con cuota agotada | Baja | Alto | Cache local de datos descargados; fuentes alternativas documentadas |
