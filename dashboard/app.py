"""Dashboard PetenFire — Predicción de Incendios Forestales, Petén Guatemala.

Ejecución:
    uv run streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from folium.plugins import HeatMap
from streamlit_folium import st_folium

sys.path.insert(0, str(Path(__file__).parent.parent))

from dashboard.utils import (
    get_feature_importance,
    load_dataset_dates,
    load_experiment_config,
    load_features_for_date,
    load_firms,
    load_grid,
    load_metrics,
    load_model,
    load_model_feature_cols,
)

# ─── Página ───────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PetenFire — Riesgo de Incendios",
    page_icon="🔥",
    layout="wide",
)

st.title("🔥 PetenFire — Sistema de Predicción de Incendios Forestales")
st.caption("Petén, Guatemala · Modelo M1 LightGBM · 1 km² / celda")

# ─── Tabs principales ─────────────────────────────────────────────────────────
tab_map, tab_metrics, tab_firms = st.tabs(
    ["🗺️ Mapa de Riesgo", "📊 Métricas del Modelo", "🔥 Análisis FIRMS"]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Mapa de Riesgo
# ══════════════════════════════════════════════════════════════════════════════
with tab_map:
    col_ctrl, col_map = st.columns([1, 3])

    with col_ctrl:
        st.subheader("Configuración")

        with st.spinner("Cargando fechas disponibles..."):
            available_dates = load_dataset_dates()

        if not available_dates:
            st.error("No hay fechas disponibles en el dataset.")
            st.stop()

        min_d, max_d = min(available_dates), max(available_dates)

        selected_date = st.date_input(
            "Fecha de predicción",
            value=max_d,
            min_value=min_d,
            max_value=max_d,
        )
        if selected_date not in available_dates:
            # Buscar la más cercana
            selected_date = min(available_dates, key=lambda d: abs((d - selected_date).days))
            st.info(f"Fecha ajustada a la más cercana disponible: {selected_date}")

        show_firms = st.checkbox("Superponer incendios FIRMS", value=True)
        firms_window = st.slider(
            "Ventana FIRMS (±días)",
            min_value=0, max_value=7, value=1,
            disabled=not show_firms,
        )
        risk_threshold = st.slider(
            "Umbral de riesgo alto (resaltar)",
            min_value=0.01, max_value=0.50, value=0.10, step=0.01,
            format="%.2f",
        )

        st.divider()
        st.caption("🌡️ **Leyenda de riesgo**")
        st.markdown(
            """
            <div style='line-height:1.8'>
            🟢 &lt;5% Bajo<br>
            🟡 5–15% Moderado<br>
            🟠 15–30% Alto<br>
            🔴 &gt;30% Muy alto
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_map:
        model = load_model()
        if model is None:
            st.warning("Modelo no encontrado. Ejecuta primero el entrenamiento.")
        else:
            with st.spinner(f"Generando mapa de riesgo para {selected_date}..."):
                from src.models.m1_risk.predict import predict_risk

                FEATURE_COLS = load_model_feature_cols()
                features_df = load_features_for_date(selected_date)
                risk_df = predict_risk(model, features_df, FEATURE_COLS)

                grid = load_grid()
                merged = grid.merge(risk_df, on="cell_id", how="left")
                merged["risk_prob"] = merged["risk_prob"].fillna(0.0)

                # Normalizar a percentiles para que el heatmap sea visible
                # incluso cuando el modelo predice valores muy bajos
                p_max = merged["risk_prob"].quantile(0.99)
                if p_max > 0:
                    merged["risk_norm"] = (merged["risk_prob"] / p_max).clip(0, 1)
                else:
                    merged["risk_norm"] = merged["risk_prob"]

            # Aviso si el modelo tiene baja discriminación
            max_risk = merged["risk_prob"].max()
            if max_risk < 0.05:
                st.info(
                    f"Modelo en etapa inicial (riesgo máx. = {max_risk:.4f}). "
                    "El mapa muestra riesgo relativo (percentil 99 = 100%). "
                    "Las métricas mejorarán cuando el entrenamiento 2018-2021 finalice."
                )

            # Estadísticas rápidas
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Celdas totales", f"{len(merged):,}")
            c2.metric(
                "Celdas riesgo alto",
                f"{(merged['risk_prob'] >= risk_threshold).sum():,}",
            )
            c3.metric("Riesgo mediano", f"{merged['risk_prob'].median():.4f}")
            c4.metric("Riesgo máximo", f"{merged['risk_prob'].max():.4f}")

            # ── Construir mapa Folium ─────────────────────────────────────────
            center_lat = merged["lat_center"].mean() if "lat_center" in merged.columns else 16.8
            center_lon = merged["lon_center"].mean() if "lon_center" in merged.columns else -90.3

            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=8,
                tiles="CartoDB positron",
            )

            # Heatmap usando valores normalizados por percentil
            heat_data = []
            for _, row in merged.iterrows():
                if "lat_center" in row and "lon_center" in row:
                    heat_data.append([row["lat_center"], row["lon_center"], float(row["risk_norm"])])

            if heat_data:
                HeatMap(
                    heat_data,
                    name="Riesgo de ignición (relativo)",
                    min_opacity=0.3,
                    max_zoom=12,
                    radius=12,
                    blur=15,
                    gradient={0.2: "blue", 0.4: "lime", 0.6: "yellow", 0.8: "orange", 1.0: "red"},
                ).add_to(m)

            # Puntos de riesgo alto
            high_risk = merged[merged["risk_prob"] >= risk_threshold]
            if not high_risk.empty and len(high_risk) <= 2000:
                fg_risk = folium.FeatureGroup(name="Celdas riesgo alto", show=True)
                for _, row in high_risk.iterrows():
                    if "lat_center" not in row or "lon_center" not in row:
                        continue
                    folium.CircleMarker(
                        location=[row["lat_center"], row["lon_center"]],
                        radius=4,
                        color="#e74c3c",
                        fill=True,
                        fill_color="#e74c3c",
                        fill_opacity=0.6,
                        popup=f"Riesgo: {row['risk_prob']:.3f}",
                        tooltip=f"{row['cell_id']}: {row['risk_prob']:.3f}",
                    ).add_to(fg_risk)
                fg_risk.add_to(m)

            # Puntos FIRMS
            if show_firms:
                firms = load_firms()
                if not firms.empty:
                    d_min = selected_date - timedelta(days=firms_window)
                    d_max = selected_date + timedelta(days=firms_window)
                    firms_win = firms[
                        (firms["acq_date"] >= d_min) & (firms["acq_date"] <= d_max)
                    ]
                    if not firms_win.empty:
                        fg_firms = folium.FeatureGroup(
                            name=f"FIRMS ±{firms_window}d ({len(firms_win):,})",
                            show=True,
                        )
                        for _, row in firms_win.iterrows():
                            folium.CircleMarker(
                                location=[row["latitude"], row["longitude"]],
                                radius=3,
                                color="black",
                                fill=True,
                                fill_color="black",
                                fill_opacity=0.8,
                                popup=f"FIRMS {row['acq_date']} | FRP={row.get('frp', '?')}",
                            ).add_to(fg_firms)
                        fg_firms.add_to(m)

            folium.LayerControl().add_to(m)
            st_folium(m, width=900, height=550, returned_objects=[])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Métricas del Modelo
# ══════════════════════════════════════════════════════════════════════════════
with tab_metrics:
    metrics = load_metrics()
    cfg = load_experiment_config()

    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.subheader("Métricas de evaluación")
        if metrics:
            for split, vals in metrics.items():
                st.markdown(f"**{split.upper()}**")
                sub_cols = st.columns(len(vals))
                for i, (k, v) in enumerate(vals.items()):
                    sub_cols[i].metric(k, f"{v:.4f}")
        else:
            st.info("No hay métricas guardadas todavía.")

        if cfg:
            st.divider()
            st.subheader("Configuración del experimento")
            st.json(cfg)

    with col_m2:
        st.subheader("Feature Importance")
        imp = get_feature_importance()
        if imp is not None and not imp.empty:
            fig = px.bar(
                imp.head(20),
                x="importance",
                y="feature",
                orientation="h",
                color="importance",
                color_continuous_scale="Oranges",
                title="Top 20 features — LightGBM split importance",
                labels={"importance": "Importancia", "feature": "Feature"},
            )
            fig.update_layout(
                yaxis={"autorange": "reversed"},
                coloraxis_showscale=False,
                height=550,
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Feature importance no disponible (modelo no cargado).")

    # Distribución de probabilidades predichas
    st.divider()
    st.subheader("Distribución de probabilidades predichas")
    model = load_model()
    if model is not None:
        with st.spinner("Calculando distribución sobre muestra del dataset..."):
            from src.models.m1_risk.predict import predict_risk
            FEATURE_COLS = load_model_feature_cols()

            all_dates = load_dataset_dates()
            # Muestra representativa: primer día de cada mes
            sample_dates = sorted({d for d in all_dates if d.day == 1})[:12]

            all_probs = []
            for d in sample_dates:
                try:
                    feat = load_features_for_date(d)
                    risk = predict_risk(model, feat, FEATURE_COLS)
                    all_probs.extend(risk["risk_prob"].tolist())
                except Exception:
                    continue

        if all_probs:
            probs_arr = np.array(all_probs)
            fig2 = go.Figure()
            fig2.add_trace(go.Histogram(
                x=probs_arr,
                nbinsx=100,
                name="P(ignición)",
                marker_color="#e67e22",
                opacity=0.8,
            ))
            fig2.update_layout(
                xaxis_title="Probabilidad predicha",
                yaxis_title="Frecuencia",
                title=f"Distribución sobre {len(sample_dates)} días de muestra ({len(probs_arr):,} celdas-día)",
                height=320,
                margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig2, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Análisis FIRMS
# ══════════════════════════════════════════════════════════════════════════════
with tab_firms:
    firms = load_firms()

    if firms.empty:
        st.warning("No se encontraron archivos FIRMS en data/raw/firms/")
    else:
        st.subheader(f"Incendios detectados — {len(firms):,} registros FIRMS")

        # Filtros
        col_f1, col_f2, col_f3 = st.columns(3)
        all_sources = sorted(firms["source"].unique().tolist()) if "source" in firms.columns else []
        sel_sources = col_f1.multiselect("Fuente", all_sources, default=all_sources)

        firms["year"] = pd.to_datetime(firms["acq_date"]).dt.year
        all_years = sorted(firms["year"].unique().tolist())
        sel_years = col_f2.multiselect("Año", all_years, default=all_years)

        min_frp = col_f3.number_input(
            "FRP mínimo (MW)", min_value=0.0, value=0.0, step=1.0
        ) if "frp" in firms.columns else 0.0

        # Filtrar
        mask = pd.Series([True] * len(firms), index=firms.index)
        if sel_sources and "source" in firms.columns:
            mask &= firms["source"].isin(sel_sources)
        if sel_years:
            mask &= firms["year"].isin(sel_years)
        if "frp" in firms.columns and min_frp > 0:
            mask &= firms["frp"] >= min_frp

        firms_f = firms[mask].copy()
        st.caption(f"{len(firms_f):,} registros tras filtrado")

        col_a, col_b = st.columns(2)

        with col_a:
            # Serie temporal
            firms_f["month"] = pd.to_datetime(firms_f["acq_date"]).dt.to_period("M").astype(str)
            monthly = firms_f.groupby("month").size().reset_index(name="count")
            fig3 = px.bar(
                monthly,
                x="month",
                y="count",
                title="Incendios detectados por mes",
                labels={"month": "Mes", "count": "N° detecciones"},
                color="count",
                color_continuous_scale="YlOrRd",
            )
            fig3.update_layout(
                height=350,
                coloraxis_showscale=False,
                margin=dict(l=10, r=10, t=40, b=10),
                xaxis_tickangle=-45,
            )
            st.plotly_chart(fig3, use_container_width=True)

        with col_b:
            # Por día del año (estacionalidad)
            firms_f["doy"] = pd.to_datetime(firms_f["acq_date"]).dt.dayofyear
            doy = firms_f.groupby("doy").size().reset_index(name="count")
            fig4 = px.line(
                doy,
                x="doy",
                y="count",
                title="Estacionalidad — incendios por día del año",
                labels={"doy": "Día del año", "count": "N° detecciones"},
            )
            fig4.update_traces(line_color="#e74c3c")
            fig4.update_layout(
                height=350,
                margin=dict(l=10, r=10, t=40, b=10),
            )
            # Marcar temporada seca (mar-may ≈ día 60-150)
            fig4.add_vrect(x0=60, x1=150, fillcolor="orange", opacity=0.15, line_width=0,
                           annotation_text="Temporada seca")
            st.plotly_chart(fig4, use_container_width=True)

        # Mapa de puntos FIRMS
        st.subheader("Mapa de detecciones FIRMS")
        if len(firms_f) > 20_000:
            firms_plot = firms_f.sample(20_000, random_state=42)
            st.caption(f"(Mostrando 20,000 de {len(firms_f):,} puntos)")
        else:
            firms_plot = firms_f

        m2 = folium.Map(location=[16.8, -90.3], zoom_start=8, tiles="CartoDB dark_matter")
        heat2 = [
            [r["latitude"], r["longitude"], float(r.get("frp", 1) or 1)]
            for _, r in firms_plot.iterrows()
        ]
        HeatMap(
            heat2,
            name="Densidad incendios",
            min_opacity=0.4,
            radius=8,
            blur=10,
            gradient={0.3: "yellow", 0.6: "orange", 1.0: "red"},
        ).add_to(m2)
        folium.LayerControl().add_to(m2)
        st_folium(m2, width=900, height=450, returned_objects=[])

        # Estadísticas descriptivas FRP
        if "frp" in firms_f.columns:
            st.divider()
            st.subheader("Intensidad de fuego (FRP — Fire Radiative Power, MW)")
            frp_stats = firms_f["frp"].describe().to_frame().T
            st.dataframe(frp_stats.style.format("{:.2f}"), use_container_width=True)

            fig5 = px.histogram(
                firms_f[firms_f["frp"] < firms_f["frp"].quantile(0.99)],
                x="frp",
                nbins=80,
                title="Distribución FRP (excl. top 1%)",
                labels={"frp": "FRP (MW)", "count": "N°"},
                color_discrete_sequence=["#e74c3c"],
            )
            fig5.update_layout(height=300, margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig5, use_container_width=True)
