"""
GERCEK VERI KATMANI (Data Layer) - GERCEK VERIYLE
----------------------------------------------------
Aydanur'un temizledigi gercek OMS kesinti veri setini
(gercek_veri_csv/03_TEMIZ_VERI(temiz_kesintiler).csv) okuyup, direk/varlik bazinda
risk skorlama ve karar destek katmanlarinin kullanabilecegi
ozet (aggregate) tablolara cevirir.

Ham veri: 2021-2025 arasi 4047 gercek kesinti kaydi, 60 fider,
2784 varlik (direk + dagitim trafosu), 6 gercek Mersin ilcesi.

Cikti semasi (diger katmanlarin bekledigi standart format):
    assets   -> her varlik (direk/trafo) icin ozet risk oznitelikleri
    feeders  -> fider bazinda ozet
    outages  -> ham kesinti kayitlarinin kendisi (trend analizleri icin)
"""

import numpy as np
import pandas as pd

RAW_FILE = "gercek_veri_csv/03_TEMIZ_VERI(temiz_kesintiler).csv"

# Gercek Mersin ilceleri icin yaklasik merkez koordinatlari (13 ilcenin tamami)
# (Aydanur'un veri setinde GPS koordinati olmadigi icin ilce merkezi kullanildi;
#  tamamen kara ustunde kalacak sekilde kuzeye/ic kesime kaydirildi - sahil ilceleri
#  icin guvenlik payi daha fazla birakildi)
ILCE_COORDS = {
    "Akdeniz":     (36.87, 34.63),
    "Mezitli":     (36.90, 34.50),
    "Toroslar":    (36.91, 34.58),
    "Yenişehir":   (36.89, 34.53),
    "Tarsus":      (36.98, 34.90),
    "Erdemli":     (36.80, 34.32),
    "Silifke":     (36.50, 33.93),
    "Mut":         (36.65, 33.44),
    "Gülnar":      (36.40, 33.75),
    "Aydıncık":    (36.37, 33.33),
    "Bozyazı":     (36.38, 32.97),
    "Anamur":      (36.33, 32.83),
    "Çamlıyayla":  (37.10, 34.56),
}

# Aydanur'un ham veri dosyasindaki ilce isimleri ASCII (Turkce karaktersiz)
# yazilmis olabilir - dogru Turkce yazima ceviriyoruz (hem eslesme hem gorsel icin)
ILCE_NAME_FIX = {
    "Yenisehir": "Yenişehir",
    "Gulnar": "Gülnar",
    "Aydincik": "Aydıncık",
    "Bozyazi": "Bozyazı",
    "Camliyayla": "Çamlıyayla",
}


def _load_raw():
    df = pd.read_csv(RAW_FILE, sep=';', encoding='windows-1254', decimal=',', low_memory=False)
    df["baslangic"] = pd.to_datetime(df["baslangic"])
    df["bitis"] = pd.to_datetime(df["bitis"])
    df["ay"] = df["baslangic"].dt.strftime("%Y-%m")
    df["ilce"] = df["ilce"].replace(ILCE_NAME_FIX)
    return df


def _assign_coords(ilce_series):
    lats, lons = [], []
    rng = np.random.default_rng(42)
    for ilce in ilce_series:
        base_lat, base_lon = ILCE_COORDS.get(ilce, (36.85, 34.6))
        lats.append(base_lat + rng.normal(0, 0.012))
        lons.append(base_lon + rng.normal(0, 0.015))
    return lats, lons


def build_asset_table(df, as_of=None):
    """Her varlik (direk/trafo) icin ozet risk oznitelikleri."""
    if as_of is None:
        as_of = df["baslangic"].max()

    last_12m = df[df["baslangic"] >= as_of - pd.Timedelta(days=365)]
    recent_counts = last_12m.groupby("asset_id").size().rename("recent_outage_count")

    agg = df.groupby("asset_id").agg(
        feeder_id=("feeder_id", "first"),
        ilce=("ilce", "first"),
        bolge_tipi=("bolge_tipi", "first"),
        varlik_tipi=("varlik_tipi", "first"),
        total_outage_count=("outage_id", "count"),
        avg_duration_min=("sure_dk", "mean"),
        total_customers_affected=("etkilenen_abone", "sum"),
        total_critical_customers=("etkilenen_kritik_musteri", "sum"),
        total_energy_loss_kwh=("kayip_enerji_kWh", "sum"),
        last_outage_date=("baslangic", "max"),
        dominant_cause=("ariza_nedeni", lambda x: x.mode().iat[0] if not x.mode().empty else "DIGER"),
    ).reset_index()

    agg = agg.merge(recent_counts, on="asset_id", how="left")
    agg["recent_outage_count"] = agg["recent_outage_count"].fillna(0).astype(int)
    agg["days_since_last_outage"] = (as_of - agg["last_outage_date"]).dt.days

    agg = agg.rename(columns={"asset_id": "pole_id"})
    lats, lons = _assign_coords(agg["ilce"])
    agg["lat"] = lats
    agg["lon"] = lons

    return agg


def build_feeder_table(df, asset_df):
    agg = asset_df.groupby("feeder_id").agg(
        asset_count=("pole_id", "count"),
        total_outage_count=("total_outage_count", "sum"),
        avg_duration_min=("avg_duration_min", "mean"),
        total_customers_affected=("total_customers_affected", "sum"),
    ).reset_index()
    agg["feeder_name"] = agg["feeder_id"]
    return agg


def generate_all_data():
    """app.py'nin bekledigi arayuzle ayni: sozluk icinde standart tablolar."""
    df = _load_raw()
    asset_df = build_asset_table(df)
    feeder_df = build_feeder_table(df, asset_df)
    return {
        "outages_raw": df,
        "assets": asset_df,
        "feeders": feeder_df,
    }


if __name__ == "__main__":
    data = generate_all_data()
    for name, d in data.items():
        print(f"{name}: {d.shape}")
    print()
    print(data["assets"].head(10))
