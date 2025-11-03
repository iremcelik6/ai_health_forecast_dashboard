import pandas as pd, numpy as np, json
from pathlib import Path
from tensorflow import keras

# === Sabitler ===
DATA = Path("../data/processed/state_timeseries.csv")
OUT  = Path("../outputs"); OUT.mkdir(exist_ok=True, parents=True)

LOOKBACK = 8
HORIZON  = 5

# === Veriyi yükle ===
df = pd.read_csv(DATA, dtype={'state_fips': str}).sort_values(["state_fips", "year"])

# === Modelleri yükle ===
model_resp = keras.models.load_model(OUT / "lstm_state_forecaster.keras")
model_cardio = keras.models.load_model(OUT / "lstm_cardio_forecaster.keras")

# === Ölçekleme parametreleri ===
mean = np.load(OUT / "scaler_mean.npy")
scale = np.load(OUT / "scaler_scale.npy")

with open(OUT / "feature_cols.json") as f:
    feat_cols = json.load(f)

def scale_feats(X):
    return (X - mean) / (scale + 1e-9)

rows = []
for fips, g in df.groupby("state_fips"):
    g = g.sort_values("year").reset_index(drop=True)
    if len(g) < LOOKBACK:
        continue

    X_last = g[feat_cols].iloc[-LOOKBACK:].values.astype(float)
    X_last = scale_feats(X_last).reshape(1, LOOKBACK, -1)

    # İki modelden de tahmin al
    y_resp = model_resp.predict(X_last, verbose=0)[0]
    y_cardio = model_cardio.predict(X_last, verbose=0)[0]

    last_year = int(g["year"].max())
    years = [last_year + i for i in range(1, HORIZON + 1)]

    for yv, pr, pc in zip(years, y_resp, y_cardio):
        rows.append({
            "state_fips": fips,
            "state_abbr": g["state_abbr"].iloc[0],
            "state": g["state"].iloc[0],
            "year": yv,
            "pred_resp_rate": float(pr),
            "pred_cardio_rate": float(pc)
        })

pred = pd.DataFrame(rows)
pred.to_csv(OUT / "predictions_state.csv", index=False)
print(f"💾 Saved: {OUT/'predictions_state.csv'}")
print(pred.head())
