import pandas as pd, numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers
import json

# === Sabitler ===
DATA = Path("../data/processed/state_timeseries.csv")
OUT  = Path("../outputs"); OUT.mkdir(exist_ok=True, parents=True)

TARGET = "cardio_rate"   # 🔹 Cardio hedefi
LOOKBACK = 8
HORIZON  = 5
BATCH    = 64
EPOCHS   = 40
LR       = 1e-3
SEED     = 42
np.random.seed(SEED); tf.random.set_seed(SEED)

# === 1) Veri yükle & sırala ===
df = pd.read_csv(DATA, dtype={'state_fips': str}).sort_values(["state_fips", "year"])

id_cols = {"state_fips", "state_abbr", "state", "year"}
drop_from_feats = {"resp_rate", "cardio_rate"}
num_cols = [c for c in df.columns if c not in id_cols and pd.api.types.is_numeric_dtype(df[c])]

# === 2) NaN hedefleri filtrele ===
before = len(df)
df = df[~df[TARGET].isna()]
print(f"🧹 {TARGET} NaN filtresi: {before} → {len(df)}")

# === 3) Eksik veri doldurma ===
def fill_group(g):
    g[num_cols] = g[num_cols].ffill().bfill()
    return g

df = df.groupby("state_fips", group_keys=False).apply(fill_group)
med = df[num_cols].median(numeric_only=True)
df[num_cols] = df[num_cols].fillna(med)

for c in num_cols:
    v = df[c].values
    v[~np.isfinite(v)] = np.nan
    df[c] = v
df[num_cols] = df[num_cols].fillna(med)

# === 4) Sıfır varyanslı kolonları at ===
std = df[num_cols].std(numeric_only=True)
num_cols = [c for c in num_cols if std[c] > 0]

# === 5) Ölçekleme ===
scaler_X = StandardScaler()
Xall = scaler_X.fit_transform(df[num_cols].values.astype(float))
np.save(OUT / "scaler_X_mean_cardio.npy", scaler_X.mean_)
np.save(OUT / "scaler_X_scale_cardio.npy", scaler_X.scale_)

scaler_y = StandardScaler()
df["y_scaled"] = scaler_y.fit_transform(df[[TARGET]])
np.save(OUT / "scaler_y_mean_cardio.npy", scaler_y.mean_)
np.save(OUT / "scaler_y_scale_cardio.npy", scaler_y.scale_)

df_scaled = df.copy()
df_scaled[num_cols] = Xall
TARGET_SCALED = "y_scaled"

# === 6) Pencere oluşturma ===
def build_windows(g):
    Xs, ys = [], []
    g = g.reset_index(drop=True)
    X_mat = g[num_cols].values.astype(float)
    y_vec = g[TARGET_SCALED].values.astype(float)
    for i in range(len(g) - LOOKBACK - HORIZON + 1):
        Xw = X_mat[i:i+LOOKBACK]
        yw = y_vec[i+LOOKBACK:i+LOOKBACK+HORIZON]
        if np.isnan(Xw).any() or np.isnan(yw).any(): continue
        Xs.append(Xw); ys.append(yw)
    return (np.array(Xs), np.array(ys)) if Xs else (None, None)

Xs, Ys = [], []
for fips, g in df_scaled.groupby("state_fips"):
    Xg, Yg = build_windows(g)
    if Xg is not None:
        Xs.append(Xg); Ys.append(Yg)
X = np.concatenate(Xs, axis=0); y = np.concatenate(Ys, axis=0)
print("✅ Pencere sayısı:", X.shape, y.shape)

# === 7) Üçlü veri bölmesi ===
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=SEED)
split_idx = len(X_temp) // 2
X_val, X_test = X_temp[:split_idx], X_temp[split_idx:]
y_val, y_test = y_temp[:split_idx], y_temp[split_idx:]
print(f"🧠 Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# === 8) Model (Regularized LSTM) ===
inp = keras.Input(shape=(LOOKBACK, X.shape[-1]))
x = keras.layers.Masking()(inp)
x = keras.layers.LSTM(128, return_sequences=True, kernel_regularizer=regularizers.l2(1e-4))(x)
x = keras.layers.Dropout(0.3)(x)
x = keras.layers.LSTM(64, return_sequences=False, kernel_regularizer=regularizers.l2(1e-4))(x)
x = keras.layers.Dropout(0.3)(x)
x = keras.layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
x = keras.layers.Dropout(0.2)(x)
out = keras.layers.Dense(HORIZON)(x)
model = keras.Model(inp, out)

optimizer = keras.optimizers.AdamW(learning_rate=LR, weight_decay=1e-5)
lr_scheduler = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, min_lr=1e-5, verbose=1)
early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1)
checkpoint = keras.callbacks.ModelCheckpoint(OUT / "best_lstm_cardio_forecaster.keras", monitor="val_loss", save_best_only=True, verbose=1)

model.compile(optimizer=optimizer, loss="mae", metrics=["mse"])

# === 9) Eğitim ===
hist = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH,
    verbose=1,
    callbacks=[lr_scheduler, early_stop, checkpoint]
)

# === 10) Kaydet & Loss grafiği ===
model.save(OUT / "lstm_cardio_forecaster.keras")
pd.DataFrame(hist.history).to_csv(OUT / "training_history_cardio.csv", index=False)

hist_df = pd.read_csv(OUT / "training_history_cardio.csv")
plt.figure(figsize=(8,5))
plt.plot(hist_df["loss"], label="Train Loss")
plt.plot(hist_df["val_loss"], label="Val Loss")
plt.xlabel("Epoch"); plt.ylabel("MAE")
plt.legend(); plt.title("Regularized LSTM (Cardiovascular Mortality)")
plt.grid(True)
plt.show()

# === 11) Performans ölçümü ===
print("\n📊 Model Evaluation Metrics")

def eval_set(Xt, yt, name="Set"):
    yp = model.predict(Xt)
    mae = mean_absolute_error(yt, yp)
    rmse = np.sqrt(mean_squared_error(yt, yp))
    r2 = r2_score(yt, yp)
    print(f"{name} → MAE: {mae:.4f} | RMSE: {rmse:.4f} | R²: {r2:.4f}")
    return mae, rmse, r2

val_mae, val_rmse, val_r2 = eval_set(X_val, y_val, "🔹 Validation")
test_mae, test_rmse, test_r2 = eval_set(X_test, y_test, "🧪 Test")

# === 12) 3'lü Performans Grafiği ===
train_mae = hist.history["loss"][-1]
train_rmse = np.sqrt(hist.history["mse"][-1]) if "mse" in hist.history else train_mae

labels = ["Train", "Validation", "Test"]
mae_values = [train_mae, val_mae, test_mae]
rmse_values = [train_rmse, val_rmse, test_rmse]

fig, ax = plt.subplots(1, 2, figsize=(10,5))
ax[0].bar(labels, mae_values, color=["skyblue", "orange", "lightcoral"])
ax[0].set_title("MAE Comparison"); ax[0].set_ylabel("MAE")
for i, v in enumerate(mae_values):
    ax[0].text(i, v + 0.01, f"{v:.3f}", ha="center")

ax[1].bar(labels, rmse_values, color=["skyblue", "orange", "lightcoral"])
ax[1].set_title("RMSE Comparison"); ax[1].set_ylabel("RMSE")
for i, v in enumerate(rmse_values):
    ax[1].text(i, v + 0.01, f"{v:.3f}", ha="center")

plt.suptitle("📊 Train vs Validation vs Test (Cardiovascular Mortality)", fontsize=13)
plt.tight_layout()
plt.show()

print("\n✅ Cardiovascular model başarıyla eğitildi ve kaydedildi: lstm_cardio_forecaster.keras")
