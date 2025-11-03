"""
heatmap_analysis.py
-------------------
Generate correlation and forecast heatmaps for AI Health Forecast Dashboard project.

"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# === PATHS ===
DATA = Path("../data/processed/state_timeseries.csv")
PRED = Path("../outputs/predictions_state.csv")

# === 1️⃣ CORRELATION HEATMAP (Pollutants vs Health Outcomes) ===
print("🔹 Generating correlation heatmap...")

# Veri yükle
df = pd.read_csv(DATA)

# Sadece sayısal kolonları seç
num_df = df.select_dtypes(include=np.number)

# Korelasyon matrisi
corr = num_df.corr()

# Pollutant ve sağlık kolonlarını filtrele
pollutant_cols = [c for c in corr.columns if any(p in c.lower() for p in ["o3", "no2", "so2", "co", "pm"])]
health_cols = [c for c in corr.columns if "resp" in c or "cardio" in c]

corr_subset = corr.loc[pollutant_cols, health_cols]

# Görselleştir
plt.figure(figsize=(8, 5))
sns.heatmap(corr_subset, annot=True, cmap="RdBu_r", center=0, linewidths=0.5)
plt.title("Correlation between Air Pollutants and Health Outcomes")
plt.tight_layout()
plt.savefig("../outputs/heatmap_correlation.png", dpi=300)
plt.close()
print("✅ Correlation heatmap saved → outputs/heatmap_correlation.png")

# === 2️⃣ FORECAST HEATMAP (2024–2028) ===
print("🔹 Generating forecast heatmap...")

df_pred = pd.read_csv(PRED)
df_pred = df_pred.dropna(subset=["year", "state"])
df_pred["year"] = df_pred["year"].astype(int)

for target, col in {
    "Respiratory Mortality": "pred_resp_rate",
    "Cardiovascular Mortality": "pred_cardio_rate"
}.items():
    pivot = df_pred.pivot_table(index="state", columns="year", values=col)
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, cmap="RdYlBu_r", linewidths=0.3)
    plt.title(f"Forecasted {target} (2024–2028)")
    plt.tight_layout()
    plt.savefig(f"../outputs/heatmap_{col}.png", dpi=300)
    plt.close()
    print(f"✅ Saved forecast heatmap for {target} → outputs/heatmap_{col}.png")

print("\n🎯 Heatmap generation complete!")
