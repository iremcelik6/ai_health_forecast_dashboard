import pandas as pd
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

RAW = Path("../data/raw")
OUT = Path("../data/processed")
OUT.mkdir(parents=True, exist_ok=True)

# === AQI verisini yükle ===
aqi = pd.read_csv(OUT / "aqi_state.csv", dtype={'state_fips': str})
print("✅ AQI yüklendi:", aqi.shape)
# AQI'dan eyalet adı → FIPS map'i (case-insensitive)
state_name_to_fips = (
    aqi[['state_fips', 'state']]
    .dropna()
    .drop_duplicates()
)
state_name_to_fips['state_l'] = state_name_to_fips['state'].astype(str).str.lower()
state_name_to_fips = dict(zip(state_name_to_fips['state_l'], state_name_to_fips['state_fips']))

# === Cardiovascular (temiz versiyon) ===
print("🔄 Cardiovascular verisi okunuyor...")
cardio = pd.read_csv(
    RAW / "Cardiovascular_Disease_Death_Rates_2010_2020.csv",
    sep=";", decimal=",", low_memory=False
)
cardio.columns = [c.lower().strip() for c in cardio.columns]

# FIPS kodu (LocationID'den)
cardio['state_fips'] = cardio['locationid'].astype(str).str.zfill(5).str[:2]

# Yılı düzelt (ör. '2010-2019' → 2010)
cardio['year'] = cardio['year'].astype(str).str.extract(r'(\d{4})')[0].astype(int)

# Data_Value sayısala dönüştür (bozuk veriler NaN olacak)
cardio['death_rate'] = (
    cardio['data_value']
    .astype(str)
    .str.replace(',', '.', regex=False)
    .str.extract(r'(\d+\.?\d*)')[0]
    .astype(float)
)

cardio = (
    cardio.groupby(['state_fips', 'year'], as_index=False)['death_rate']
    .mean(numeric_only=True)
    .rename(columns={'death_rate': 'cardio_rate'})
)
print("✅ Cardiovascular temizlendi:", cardio.shape)

# === Respiratory (solunum hastalıkları) ===
print("🔄 Respiratory verisi okunuyor...")
resp = pd.read_csv(RAW / "respiratory_disease_mortality_rate_usa.csv")
resp.columns = [c.lower().strip() for c in resp.columns]

# FIPS → state_fips (ilk 2 hane)
resp['state_fips'] = resp['fips'].astype(str).str.zfill(2)
resp['year'] = resp['year_id'].astype(int)
resp['death_rate'] = resp['rate'].astype(float)

# Eyalet bazında yıllık ortalama
resp = (
    resp.groupby(['state_fips', 'year'], as_index=False)['death_rate']
    .mean(numeric_only=True)
    .rename(columns={'death_rate': 'resp_rate'})
)
print("✅ Respiratory temizlendi:", resp.shape)

# === Pollution (2000–2023) ===
print("🔄 Pollution verisi okunuyor...")
poll = pd.read_csv(RAW / "pollution_2000_2023.csv", low_memory=False)

# kolon isimlerini normalize et
poll.columns = [c.strip().lower().replace(' ', '_') for c in poll.columns]

# tarih → yıl
if 'date' in poll.columns:
    try:
        poll['year'] = pd.to_datetime(poll['date'], errors='coerce').dt.year
    except Exception:
        # fallback: ilk 4 hane
        poll['year'] = poll['date'].astype(str).str.extract(r'(^\d{4})')[0].astype(float)
else:
    raise ValueError("pollution_2000_2023.csv içinde 'Date' kolonu bulunamadı.")

# state_fips üret (State adından)
if 'state' not in poll.columns:
    raise ValueError("pollution_2000_2023.csv içinde 'State' kolonu yok.")
poll['state_fips'] = poll['state'].astype(str).str.lower().map(state_name_to_fips)

# geçersiz / eksikleri at
poll = poll.dropna(subset=['state_fips', 'year'])
poll['year'] = poll['year'].astype(int)

# sayısal kirletici kolonları seç (o3, co, so2, no2, aqi, varsa pm)
num_cols = poll.select_dtypes(include='number').columns.tolist()
# gruplayıp ortalama al
poll = (
    poll.groupby(['state_fips', 'year'], as_index=False)[num_cols]
        .mean(numeric_only=True)
)

# ana çıktıdan gereksiz id kolonlarını zaten seçmedik; bilgi amaçlı boyut yazdır
print("✅ Pollution temizlendi:", poll.shape)


# === Merge hepsi ===
print("🔄 Veriler birleştiriliyor...")
df = (
    aqi.merge(poll, on=['state_fips', 'year'], how='left')
       .merge(cardio, on=['state_fips', 'year'], how='left')
       .merge(resp, on=['state_fips', 'year'], how='left')
)

# Eksik verileri doldur (her eyalet için ileri-geri doldurma)
df = df.sort_values(['state_fips', 'year'])
df = df.groupby('state_fips').apply(lambda g: g.ffill().bfill()).reset_index(drop=True)
# === Sadece ABD eyaletleri + DC kalsın (state_abbr var ve state_fips != '00') ===
before = len(df)
df = df[df['state_abbr'].notna()]
df = df[df['state_fips'] != '00']

# (İsteğe bağlı) yıl aralığını hizala: cardio 2010–2020, aqi 2010–2023
# Eğer hedefin 2010–2020 ile başlamaksa:
# df = df[(df['year'] >= 2010) & (df['year'] <= 2020)]

after = len(df)
print(f"🧹 Filtre: {before} → {after} satır (ABD eyaletleri + DC)")


# === Kaydet ===
OUT_FILE = OUT / "state_timeseries.csv"
df.to_csv(OUT_FILE, index=False)
print("💾 Kaydedildi:", OUT_FILE)
print("\n📊 İlk 10 satır:")
print(df.head(10))

