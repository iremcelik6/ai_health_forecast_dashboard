import streamlit as st
import pandas as pd
import plotly.express as px
import requests
from pathlib import Path

# === SAYFA AYARLARI ===
st.set_page_config(page_title="AI Health Forecast Dashboard", layout="wide")
st.title("🧠 AI Health Forecast Dashboard")
st.caption("Forecasting and simulating how air pollution affects cardiovascular and respiratory mortality rates across U.S. states (2020–2028).")

# === DOSYA YOLLARI ===
pred_path = Path("outputs/predictions_state.csv")
whatif_path = Path("outputs/what_if_results.csv")

# === DOSYA KONTROLÜ ===
if not pred_path.exists() or not whatif_path.exists():
    st.error("🚨 Missing files. Please run 'forecast_next_years.py' and 'what_if_analysis.py' first.")
    st.stop()

# === VERİLERİ YÜKLE ===
df_pred = pd.read_csv(pred_path)
df_whatif = pd.read_csv(whatif_path)

df_pred = df_pred.dropna(subset=["year", "state"])
df_pred["year"] = df_pred["year"].astype(int)

# === KİRLETİCİLERİ NORMALİZE ET (O₃, NO₂, SO₂, CO) ===
def to_base_pollutant(name: str) -> str:
    n = str(name).lower()
    if "o3" in n:
        return "O₃ (Ozone)"
    if "no2" in n:
        return "NO₂ (Nitrogen Dioxide)"
    if "so2" in n:
        return "SO₂ (Sulfur Dioxide)"
    if "co" in n:
        return "CO (Carbon Monoxide)"
    return name  # fallback

df_whatif["BasePollutant"] = df_whatif["pollutant"].apply(to_base_pollutant)

# Aynı kirleticiye ait alt değişkenleri grupla
df_whatif_grp = (
    df_whatif
    .groupby(["BasePollutant", "target"], as_index=False)["avg_delta"]
    .mean()
)

# === SEKME YAPISI ===
tab1, tab2 = st.tabs(["🌍 Forecasted Mortality Map", "🧪 What-If (+10%) Simulation"])

# ===============================================================
# 🌍 TAB 1 — Forecasted Mortality Map
# ===============================================================
with tab1:
    st.markdown("### 🌎 Forecasted Mortality Map (2020–2028)")

    health_option = st.radio(
        "Select health outcome to visualize:",
        ["Respiratory Mortality", "Cardiovascular Mortality"],
        horizontal=True
    )

    color_col = "pred_resp_rate" if health_option == "Respiratory Mortality" else "pred_cardio_rate"

    # === GEOJSON ===
    geojson_url = "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json"
    us_states = requests.get(geojson_url).json()

    fig_map = px.choropleth(
        df_pred,
        geojson=us_states,
        locations="state",
        featureidkey="properties.name",
        color=color_col,
        animation_frame="year",
        color_continuous_scale="RdYlBu_r",
        range_color=(df_pred[color_col].min(), df_pred[color_col].max()),
        hover_name="state",
        labels={color_col: f"Predicted {health_option} (per 100k)"},
    )

    fig_map.update_geos(
        fitbounds="locations",
        visible=True,
        showcountries=True,
        showland=True,
        landcolor="white",
        lakecolor="lightblue",
        coastlinecolor="gray",
        projection_type="albers usa",
    )

    fig_map.update_traces(marker_line_width=0.8, marker_line_color="gray")
    fig_map.update_layout(title=f"Predicted {health_option} by State (2020–2028)", title_x=0.3)

    st.plotly_chart(fig_map, use_container_width=True)

    st.info(f"""
    🧭 **How to read the map:**
    - 🔴 **Red** = higher predicted {health_option.lower()}
    - 🔵 **Blue** = lower predicted {health_option.lower()}
    - Move the **timeline slider** below the map to explore 2020–2028
    - Hover any state to view its predicted mortality (per 100,000 people)
    """)

# ===============================================================
# 🧪 TAB 2 — What-If Analysis (+10%)
# ===============================================================
with tab2:
    st.markdown("### 🧪 What-If Analysis (+10% Pollutant Impact on Mortality)")

    col1, col2 = st.columns(2)
    with col1:
        pollutant = st.selectbox(
            "🌫️ Select pollutant:",
            sorted(df_whatif_grp["BasePollutant"].unique())
        )
    with col2:
        outcome = st.selectbox(
            "🩺 Select health outcome:",
            ["Respiratory Mortality", "Cardiovascular Mortality"]
        )

    label_map = {
        "resp_rate": "Respiratory Mortality",
        "cardio_rate": "Cardiovascular Mortality"
    }

    sel = df_whatif_grp[df_whatif_grp["BasePollutant"] == pollutant].copy()
    sel["Outcome"] = sel["target"].map(label_map)
    sel["Change (%)"] = sel["avg_delta"] * 100
    sel["Abs Change (%)"] = sel["Change (%)"].abs().clip(lower=0.01)
    sel["ycat"] = pollutant  # tek satırda çizim

    fig = px.scatter(
        sel,
        x="Change (%)",
        y="ycat",
        color="Outcome",
        color_discrete_map={
            "Respiratory Mortality": "#2E86AB",
            "Cardiovascular Mortality": "#D7263D",
        },
        size="Abs Change (%)",
        size_max=40,
        text="Outcome",
        title=f"Impact of +10% {pollutant} on Health Outcomes",
        labels={"ycat": "", "Change (%)": "Predicted Change (%)"}
    )

    fig.update_traces(textposition="top center", marker_line_width=0.5, marker_line_color="rgba(0,0,0,.3)")
    fig.update_layout(
        xaxis_zeroline=True,
        xaxis_zerolinewidth=1,
        xaxis_zerolinecolor="#999",
        plot_bgcolor="rgba(0,0,0,0)",
        title_x=0.35
    )

    st.plotly_chart(fig, use_container_width=True)

    # === YORUM ===
    st.markdown("### 💬 AI Interpretation")
    for _, row in sel.iterrows():
        direction = "increase 🔺" if row["Change (%)"] > 0 else "decrease 🔻"
        st.markdown(
            f"- A **10% rise in {pollutant}** is predicted to cause a **{abs(row['Change (%)']):.2f}% {direction}** in **{row['Outcome']}** across the U.S."
        )

    st.info("""
    🧭 **How to interpret the chart:**
    - Each point shows how a +10% pollutant increase impacts mortality rates.
    - Blue = Respiratory Mortality | Red = Cardiovascular Mortality  
    - Larger bubbles = stronger predicted effect  
    - Values are aggregated across model horizon (average effect).
    """)


