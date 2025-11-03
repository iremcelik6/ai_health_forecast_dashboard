import pandas as pd
from pathlib import Path

RAW = Path("../data/raw")
OUT = Path("../data/processed"); OUT.mkdir(parents=True, exist_ok=True)

STATE_MAP = {
    "alabama": ("AL","01"), "alaska": ("AK","02"), "arizona": ("AZ","04"), "arkansas": ("AR","05"),
    "california": ("CA","06"), "colorado": ("CO","08"), "connecticut": ("CT","09"), "delaware": ("DE","10"),
    "district of columbia": ("DC","11"), "florida": ("FL","12"), "georgia": ("GA","13"), "hawaii": ("HI","15"),
    "idaho": ("ID","16"), "illinois": ("IL","17"), "indiana": ("IN","18"), "iowa": ("IA","19"),
    "kansas": ("KS","20"), "kentucky": ("KY","21"), "louisiana": ("LA","22"), "maine": ("ME","23"),
    "maryland": ("MD","24"), "massachusetts": ("MA","25"), "michigan": ("MI","26"), "minnesota": ("MN","27"),
    "mississippi": ("MS","28"), "missouri": ("MO","29"), "montana": ("MT","30"), "nebraska": ("NE","31"),
    "nevada": ("NV","32"), "new hampshire": ("NH","33"), "new jersey": ("NJ","34"), "new mexico": ("NM","35"),
    "new york": ("NY","36"), "north carolina": ("NC","37"), "north dakota": ("ND","38"), "ohio": ("OH","39"),
    "oklahoma": ("OK","40"), "oregon": ("OR","41"), "pennsylvania": ("PA","42"), "rhode island": ("RI","44"),
    "south carolina": ("SC","45"), "south dakota": ("SD","46"), "tennessee": ("TN","47"), "texas": ("TX","48"),
    "utah": ("UT","49"), "vermont": ("VT","50"), "virginia": ("VA","51"), "washington": ("WA","53"),
    "west virginia": ("WV","54"), "wisconsin": ("WI","55"), "wyoming": ("WY","56")
}

aqi_files = sorted(RAW.glob("annual_aqi_by_county_*.csv"))
print(f"📂 AQI dosya sayısı: {len(aqi_files)}")

frames = []
for f in aqi_files:
    df = pd.read_csv(f)
    df.columns = [c.strip().lower() for c in df.columns]
    needed = ["state","year","median aqi","days with aqi"]
    if not all(c in df.columns for c in needed):
        print(f"⚠️ {f.name} beklenen kolonlar eksik, atlanıyor")
        continue

    agg = {"median aqi":"mean","days with aqi":"mean"}
    for c in ["pm2.5","ozone","co","no2"]:
        if c in df.columns:
            agg[c] = "mean"

    g = (df.groupby(["state","year"], as_index=False)
           .agg(agg)
           .rename(columns={"median aqi":"aqi_median","days with aqi":"aqi_days",
                            "pm2.5":"pm25","ozone":"o3"}))

    g["state_abbr"] = g["state"].map(lambda s: STATE_MAP.get(s.lower(), ("NA","00"))[0])
    g["state_fips"] = g["state"].map(lambda s: STATE_MAP.get(s.lower(), ("NA","00"))[1])
    frames.append(g)

aqi_state = pd.concat(frames, ignore_index=True)
cols = ["state_fips","state_abbr","state","year","aqi_median","aqi_days","pm25","o3","co","no2"]
aqi_state = aqi_state[[c for c in cols if c in aqi_state.columns]]

out_file = OUT / "aqi_state.csv"
aqi_state.to_csv(out_file, index=False)
print("💾 Kaydedildi:", out_file)
print(aqi_state.head(10))
