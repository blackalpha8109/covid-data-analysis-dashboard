"""
dashboard.py
------------
Main Streamlit dashboard for COVID-19 Big Data Analytics.
Run with: streamlit run src/dashboard.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from fetch_data import fetch_covid_data
from analyze import clean_data, get_summary_stats, get_monthly_aggregates, get_wave_periods
from predict import run_prediction_pipeline

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="COVID-19 Big Data Dashboard",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

COUNTRIES = ["India", "United States", "Brazil", "United Kingdom", "Germany", "France"]
COLOR_MAP = {
    "India": "#FF6B35",
    "United States": "#4CC9F0",
    "Brazil": "#7BC67E",
    "United Kingdom": "#F72585",
    "Germany": "#FFD60A",
    "France": "#9B5DE5",
}

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0d1117; }
    
    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #161b22; }
    
    /* Metric cards */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1f2937, #111827);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 16px;
    }
    div[data-testid="metric-container"] label { color: #8b949e !important; font-size: 13px !important; }
    div[data-testid="metric-container"] div { color: #f0f6fc !important; }
    
    /* Headers */
    h1, h2, h3 { color: #f0f6fc !important; }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab"] { color: #8b949e; background-color: #161b22; border-radius: 8px 8px 0 0; }
    .stTabs [aria-selected="true"] { color: #58a6ff !important; border-bottom: 2px solid #58a6ff !important; }
    
    /* Divider */
    hr { border-color: #30363d; }
    
    .big-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #58a6ff, #bc8cff, #ff7b72);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle {
        color: #8b949e;
        font-size: 0.95rem;
        margin-top: 4px;
    }
    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #58a6ff;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        border-left: 3px solid #58a6ff;
        padding-left: 10px;
        margin: 20px 0 12px 0;
    }
</style>
""", unsafe_allow_html=True)


# ── Load Data ─────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    df = fetch_covid_data(use_cache=True)
    df = clean_data(df)
    return df


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🦠 COVID-19 Dashboard")
    st.markdown("**Big Data Analytics — TA-1**")
    st.markdown("*Topic 10: COVID-19 Tracking & Prediction*")
    st.divider()

    st.markdown("**⚙️ Filters**")
    selected_countries = st.multiselect(
        "Select Countries",
        options=COUNTRIES,
        default=COUNTRIES[:4]
    )

    date_range = st.select_slider(
        "Year Range",
        options=["2020", "2021", "2022", "2023"],
        value=("2020", "2022")
    )

    selected_country_ml = st.selectbox(
        "Country for ML Prediction",
        options=selected_countries if selected_countries else COUNTRIES,
        index=0
    )

    st.divider()
    st.markdown("**📚 Data Sources**")
    st.markdown("- [Our World in Data](https://ourworldindata.org)")
    st.markdown("- [WHO Dashboard](https://covid19.who.int)")
    st.markdown("- [Johns Hopkins CSSE](https://coronavirus.jhu.edu)")
    st.divider()
    st.caption("Team: Arjun (28) · Armaan (29) · Arnav (30)")


# ── Load & Filter ─────────────────────────────────────────────────────────────
with st.spinner("🔄 Loading COVID-19 data pipeline..."):
    df = load_data()

if not selected_countries:
    st.warning("⚠️ Please select at least one country from the sidebar.")
    st.stop()

df_filtered = df[
    (df["location"].isin(selected_countries)) &
    (df["date"].dt.year >= int(date_range[0])) &
    (df["date"].dt.year <= int(date_range[1]))
]


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="big-title">🦠 COVID-19 Big Data Analytics Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Big Data Analytics · TA-1 · Topic 10: COVID-19 Tracking & Prediction</div>', unsafe_allow_html=True)
st.divider()


# ── KPI Metrics ───────────────────────────────────────────────────────────────
summary = get_summary_stats(df_filtered)
total_cases = int(summary["total_cases"].sum())
total_deaths = int(summary["total_deaths"].sum())
avg_vax = summary["max_vaccinated_pct"].mean()
avg_cfr = summary["case_fatality_rate_%"].mean()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("🔴 Total Cases", f"{total_cases:,}")
with col2:
    st.metric("💀 Total Deaths", f"{total_deaths:,}")
with col3:
    st.metric("💉 Avg Vaccination %", f"{avg_vax:.1f}%")
with col4:
    st.metric("📊 Avg Case Fatality Rate", f"{avg_cfr:.2f}%")


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Case Trends", "🌍 Country Comparison", "🔥 Heatmap", "🤖 ML Prediction"
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: Case Trends
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Daily New Cases — 7-Day Rolling Average</div>', unsafe_allow_html=True)

    fig_trend = go.Figure()
    for country in selected_countries:
        cdf = df_filtered[df_filtered["location"] == country].copy()
        cdf["rolling"] = cdf["new_cases"].rolling(7).mean()
        fig_trend.add_trace(go.Scatter(
            x=cdf["date"], y=cdf["rolling"],
            name=country,
            mode="lines",
            line=dict(width=2, color=COLOR_MAP.get(country, "#aaa")),
            fill="tozeroy",
            fillcolor=COLOR_MAP.get(country, "#aaa").replace("#", "rgba(") + ",0.07)",
        ))

    fig_trend.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
        xaxis_title="Date",
        yaxis_title="Daily New Cases (7-day avg)",
        height=400,
        margin=dict(l=0, r=0, t=20, b=0)
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    st.markdown('<div class="section-header">Cumulative Total Cases Over Time</div>', unsafe_allow_html=True)
    fig_cum = px.line(
        df_filtered, x="date", y="total_cases",
        color="location", color_discrete_map=COLOR_MAP,
        template="plotly_dark",
        labels={"total_cases": "Total Cases", "date": "Date", "location": "Country"}
    )
    fig_cum.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(bgcolor="rgba(0,0,0,0)"), height=380,
        margin=dict(l=0, r=0, t=20, b=0)
    )
    st.plotly_chart(fig_cum, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: Country Comparison
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">Summary by Country</div>', unsafe_allow_html=True)

    display_summary = summary[["location", "total_cases", "total_deaths",
                                "peak_daily_cases", "max_vaccinated_pct", "case_fatality_rate_%"]]
    display_summary.columns = ["Country", "Total Cases", "Total Deaths",
                                "Peak Daily Cases", "Max Vaccinated %", "CFR %"]
    st.dataframe(display_summary.style.format({
        "Total Cases": "{:,}",
        "Total Deaths": "{:,}",
        "Peak Daily Cases": "{:,}",
        "Max Vaccinated %": "{:.1f}%",
        "CFR %": "{:.2f}%",
    }), use_container_width=True, hide_index=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-header">Total Cases by Country</div>', unsafe_allow_html=True)
        fig_bar = px.bar(
            summary.sort_values("total_cases"),
            x="total_cases", y="location",
            orientation="h",
            color="location", color_discrete_map=COLOR_MAP,
            template="plotly_dark",
            labels={"total_cases": "Total Cases", "location": ""}
        )
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False, height=320, margin=dict(l=0, r=0, t=10, b=0)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">Vaccination Progress</div>', unsafe_allow_html=True)
        fig_vax = px.bar(
            summary.sort_values("max_vaccinated_pct"),
            x="max_vaccinated_pct", y="location",
            orientation="h",
            color="location", color_discrete_map=COLOR_MAP,
            template="plotly_dark",
            labels={"max_vaccinated_pct": "Vaccinated %", "location": ""}
        )
        fig_vax.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            showlegend=False, height=320, margin=dict(l=0, r=0, t=10, b=0)
        )
        st.plotly_chart(fig_vax, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: Heatmap
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">Monthly Case Heatmap</div>', unsafe_allow_html=True)
    st.caption("Each cell = total new cases in that month for that country. Darker = more cases.")

    monthly = get_monthly_aggregates(df_filtered)
    pivot = monthly.pivot_table(index="location", columns="year_month", values="monthly_cases", aggfunc="sum")
    pivot = pivot.fillna(0)

    fig_heat = px.imshow(
        pivot,
        color_continuous_scale="Reds",
        template="plotly_dark",
        aspect="auto",
        labels={"x": "Month", "y": "Country", "color": "Cases"},
    )
    fig_heat.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=380, margin=dict(l=0, r=0, t=20, b=0),
        xaxis=dict(tickangle=-45)
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown('<div class="section-header">Monthly Deaths Heatmap</div>', unsafe_allow_html=True)
    pivot_deaths = monthly.pivot_table(index="location", columns="year_month", values="monthly_deaths", aggfunc="sum")
    pivot_deaths = pivot_deaths.fillna(0)

    fig_heat2 = px.imshow(
        pivot_deaths,
        color_continuous_scale="Oranges",
        template="plotly_dark",
        aspect="auto",
        labels={"x": "Month", "y": "Country", "color": "Deaths"},
    )
    fig_heat2.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        height=380, margin=dict(l=0, r=0, t=20, b=0),
        xaxis=dict(tickangle=-45)
    )
    st.plotly_chart(fig_heat2, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: ML Prediction
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown(f'<div class="section-header">30-Day Case Forecast — {selected_country_ml}</div>', unsafe_allow_html=True)

    st.info("""
    **ML Model Used:** Linear Regression on 7-day rolling average of daily cases.  
    This is a simplified version of the **LSTM / Prophet** models described in the case study.  
    It shows the concept of time-series forecasting using historical COVID data.
    """)

    with st.spinner(f"🤖 Training model on {selected_country_ml} data..."):
        result = run_prediction_pipeline(df[df["location"] == selected_country_ml], selected_country_ml)

    metrics = result["metrics"]
    col_m1, col_m2 = st.columns(2)
    with col_m1:
        st.metric("📉 Mean Absolute Error (MAE)", f"{metrics['MAE']:,.0f} cases")
    with col_m2:
        st.metric("📈 R² Score", f"{metrics['R2_Score']:.4f}")

    cdf = result["country_df"]
    X_train = result["X_train"]
    X_test = result["X_test"]
    y_train = result["y_train"]
    y_test = result["y_test"]
    y_pred = result["y_pred"]
    future_days = result["future_days"]
    future_preds = result["future_preds"]

    # Rebuild dates for future
    last_date = cdf["date"].max()
    future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=30)

    fig_pred = go.Figure()

    # Actual data
    fig_pred.add_trace(go.Scatter(
        x=cdf["date"], y=cdf["rolling_avg"],
        name="Actual (7-day avg)", mode="lines",
        line=dict(color="#58a6ff", width=2)
    ))

    # Test predictions
    test_dates = cdf["date"].iloc[-len(y_test):]
    fig_pred.add_trace(go.Scatter(
        x=test_dates, y=y_pred,
        name="Model Fit (test set)", mode="lines",
        line=dict(color="#ff7b72", width=2, dash="dot")
    ))

    # 30-day forecast
    fig_pred.add_trace(go.Scatter(
        x=list(future_dates) + list(future_dates[::-1]),
        y=list(future_preds * 1.15) + list(future_preds * 0.85),
        fill="toself", fillcolor="rgba(188,140,255,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Confidence Interval"
    ))
    fig_pred.add_trace(go.Scatter(
        x=future_dates, y=future_preds,
        name="30-Day Forecast", mode="lines",
        line=dict(color="#bc8cff", width=3, dash="dash")
    ))

    fig_pred.add_vline(x=str(last_date), line_dash="dot", line_color="#8b949e",
                       annotation_text="Forecast starts here")

    fig_pred.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        xaxis_title="Date", yaxis_title="Daily New Cases",
        height=450, margin=dict(l=0, r=0, t=20, b=0),
        hovermode="x unified"
    )
    st.plotly_chart(fig_pred, use_container_width=True)

    st.markdown(f"""
    > **How to read this chart:**  
    > - 🔵 Blue line = actual reported cases (7-day smoothed)  
    > - 🔴 Dotted red = how well the model fits historical data  
    > - 🟣 Purple dashed = **30-day future prediction**  
    > - Shaded region = uncertainty range (±15%)
    """)
