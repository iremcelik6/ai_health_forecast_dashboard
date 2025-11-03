import pandas as pd
import plotly.express as px
from pathlib import Path

# === Dosyalar ===
DATA = Path("../data/processed/state_timeseries.csv")
PRED = Path("../outputs/predictions_state.csv")
WHATIF = Path("../outputs/what_if_results.csv")

# === Verileri yükle ===
hist = pd.read_csv(DATA, dtype={'state_fips': str})
pred = pd.read_csv(PRED, dtype={'state_fips': str})
whatif = pd.read_csv(WHATIF)

target = "resp_rate"
if f"pred_{target}" in pred.columns:
    pred.rename(columns={f"pred_{target}": target}, inplace=True)

# === Son yıl ve tahminleri birleştir ===
last_hist = (
    hist.sort_values(["state_fips", "year"])
    .groupby("state_fips")
    .tail(1)[["state_fips", "state_abbr", "state", "year", target]]
)
plot_df = pd.concat(
    [last_hist, pred[["state_fips", "state_abbr", "state", "year", target]]],
    ignore_index=True
)
plot_df["year"] = plot_df["year"].astype(int)

# === Kirletici isimlerini kullanıcı dostu hale getir ===
pollutant_names = {
    "pm2_5_mean": "PM2.5 (Fine Particles)",
    "o3_mean": "O₃ (Ozone)",
    "o3_aqi": "O₃ AQI (Air Quality Index)",
    "no2_mean": "NO₂ (Nitrogen Dioxide)",
    "no2_aqi": "NO₂ AQI",
    "so2_mean": "SO₂ (Sulfur Dioxide)",
    "so2_aqi": "SO₂ AQI",
    "co_mean": "CO (Carbon Monoxide)",
    "co_aqi": "CO AQI",
    "o3_1st_max_hour": "O₃ Max Hour",
    "o3_1st_max_value": "O₃ Max Value",
    "no2_1st_max_hour": "NO₂ Max Hour",
    "no2_1st_max_value": "NO₂ Max Value",
    "so2_1st_max_hour": "SO₂ Max Hour",
    "so2_1st_max_value": "SO₂ Max Value",
    "co_1st_max_hour": "CO Max Hour",
    "co_1st_max_value": "CO Max Value",
}
whatif["pollutant_label"] = whatif["pollutant"].map(pollutant_names).fillna(whatif["pollutant"])

# === Ortalama etki hesapla ===
whatif_avg = (
    whatif.groupby(["pollutant_label", "target"])
    .mean(numeric_only=True)
    .reset_index()
)

# === Dropdown için kirletici listesi ===
pollutants = sorted(whatif_avg["pollutant_label"].unique())
default_pollutant = pollutants[0]

# === Ana forecast haritası ===
fig_forecast = px.choropleth(
    plot_df,
    locations="state_abbr",
    locationmode="USA-states",
    color=target,
    animation_frame="year",
    scope="usa",
    color_continuous_scale="RdYlBu_r",
    labels={target: "Respiratory Death Rate (per 100k)"},
    hover_data={"state": True, "year": True, target: ":.3f"},
    title="🌎 AI Forecast of Respiratory Disease Mortality in the U.S. (2020–2028)"
)
fig_forecast.update_layout(
    geo=dict(bgcolor="rgba(0,0,0,0)"),
    coloraxis_colorbar=dict(title="Death Rate<br>(per 100k)")
)

# === What-If Bar Chart ===
fig_whatif = px.bar(
    whatif_avg,
    x="pollutant_label",
    y="avg_delta",
    color="target",
    barmode="group",
    title=f"⚙️ What-If Scenario: Average Δ Effect per Pollutant (+10%)",
    labels={
        "avg_delta": "Δ Change in Rate (per 100k)",
        "pollutant_label": "Pollutant",
        "target": "Health Outcome"
    },
    color_discrete_sequence=px.colors.qualitative.Set2,
)

fig_whatif.update_layout(
    xaxis=dict(tickangle=-30),
    yaxis_title="Δ Predicted Rate (per 100k)",
    font=dict(size=14),
    updatemenus=[
        {
            "buttons": [
                {
                    "label": pol,
                    "method": "update",
                    "args": [
                        {"visible": [whatif_avg["pollutant_label"] == pol]},
                        {"title": f"⚙️ What-If Scenario: +10% {pol}"}
                    ],
                }
                for pol in pollutants
            ],
            "direction": "down",
            "showactive": True,
            "x": 0.45,
            "xanchor": "left",
            "y": 1.15,
            "yanchor": "top"
        }
    ]
)

# === Dashboard birleştirme ===
OUT = Path("../outputs/dashboard_with_dropdown.html")
with open(OUT, "w") as f:
    f.write("<h2 style='text-align:center;'>🌎 AI Health Forecast & What-If Dashboard</h2>")
    f.write(fig_forecast.to_html(full_html=False, include_plotlyjs='cdn'))
    f.write("<hr><h2 style='text-align:center;'>⚙️ What-If Scenario Explorer</h2>")
    f.write("<p style='text-align:center;color:gray;'>Select a pollutant below to explore how +10% pollution affects health outcomes.</p>")
    f.write(fig_whatif.to_html(full_html=False, include_plotlyjs=False))

print(f"✅ Yeni dashboard oluşturuldu: {OUT}")


