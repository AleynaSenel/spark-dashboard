"""
OZGUR'UN GERCEK RISK SKORLAMA MODELI - ANALITIK KATMANIN ANA KAYNAGI
-----------------------------------------------------------------------
Ozgur Sakalli'nin uretttigi gercek risk skorlama dosyasini
(gercek_veri_csv/Dagitim_Sebekesi_Risk_Skorlama_Modeli(Risk Analizi).csv) okuyup
dashboard'un bekledigi standart semaya cevirir.

Bu dosya, ekibin kendi urettigi GERCEK risk skorlarini icerir - referans
(placeholder) formul olan real_risk_scoring.py'nin yerini alir. 20.715
varlik icin Kirilganlik_Puan (yas, arıza gecmisi, agac riski, yuklenme
orani) ve Etki_Puan (fider abone sayisi, kritik musteri, enerji kaybi)
ayri ayri hesaplanip Risk_Skoru'nda birlestirilmis.

Koordinat ve bolge tipi (kentsel/kirsal) bu dosyada yok - real_data_layer
ile ayni yontemle (gercek ilce merkezi + guvenli jitter) atanir.
"""

import os
import numpy as np
import pandas as pd

from real_data_layer import ILCE_COORDS, ILCE_NAME_FIX, _load_raw as _load_aydanur_raw

RAW_FILE = "gercek_veri_csv/Dagitim_Sebekesi_Risk_Skorlama_Modeli(Risk Analizi).csv"

# Mersin'in 4 merkez (buyuksehir) ilcesi kentsel kabul edilir, digerleri kirsal
# (Aydanur'un dosyasindaki bolge_tipi alanina paralel, gercekci bir varsayim)
KENTSEL_ILCELER = {"Akdeniz", "Toroslar", "Yenişehir", "Mezitli"}


def _is_real_data():
    return os.path.exists(RAW_FILE)


def _assign_coords(ilce_series):
    lats, lons = [], []
    rng = np.random.default_rng(7)
    for ilce in ilce_series:
        base_lat, base_lon = ILCE_COORDS.get(ilce, (36.85, 34.6))
        lats.append(base_lat + rng.normal(0, 0.012))
        lons.append(base_lon + rng.normal(0, 0.015))
    return lats, lons


def compute_asset_risk():
    """app_gercek.py'nin bekledigi semaya uygun, Ozgur'un GERCEK risk
    skorlarini iceren asset_risk_df dondurur."""
    df = pd.read_csv(RAW_FILE, sep=';', encoding='windows-1254', decimal=',', low_memory=False)
    df["ilce"] = df["ilce"].replace(ILCE_NAME_FIX)
    df["bolge_tipi"] = df["ilce"].apply(lambda i: "kentsel" if i in KENTSEL_ILCELER else "kirsal")

    df = df.rename(columns={
        "asset_id": "pole_id",
        "asset_type": "varlik_tipi",
        "Risk_Skoru": "risk_score",
        "Risk_Kategorisi": "risk_category",
    })

    lats, lons = _assign_coords(df["ilce"])
    df["lat"] = lats
    df["lon"] = lons

    # dashboard'un bazi tablolarda beklediği alanlar icin makul karsiliklar
    df["total_outage_count"] = df["ariza_sayisi_5yil"]
    df["recent_outage_count"] = (df["ariza_sayisi_5yil"] / 2.5).round().astype(int)
    df["avg_duration_min"] = 42.0  # Ozgur'un dosyasinda yok - Aydanur'un genel ortalamasina yakin sabit varsayim
    df["total_customers_affected"] = df["fider_abone_sayisi"]
    df["total_critical_customers"] = df["fider_kritik_musteri"]

    # Aydanur'un gercek kesinti kayitlarindan, her varlik icin en sik gorulen
    # gercek arıza nedenini eslestiriyoruz (asset_id uzerinden)
    try:
        aydanur_df = _load_aydanur_raw()
        dominant = (
            aydanur_df.groupby("asset_id")["ariza_nedeni"]
            .agg(lambda x: x.mode().iat[0] if not x.mode().empty else "DIGER")
        )
        df["dominant_cause"] = df["pole_id"].map(dominant)
        df["dominant_cause"] = df["dominant_cause"].fillna("Gerçek kayıt yok (bu varlıkta geçmiş arıza görülmemiş)")
    except Exception:
        df["dominant_cause"] = "Bilinmiyor (Aydanur'un dosyası okunamadı)"

    df["risk_score"] = df["risk_score"].round(1)
    return df


def compute_feeder_risk(asset_risk_df):
    agg = asset_risk_df.groupby("feeder_id").agg(
        avg_risk=("risk_score", "mean"),
        max_risk=("risk_score", "max"),
        asset_count=("pole_id", "count"),
        critical_assets=("risk_category", lambda x: (x == "Kritik").sum()),
        high_assets=("risk_category", lambda x: (x == "Yüksek").sum()),
        total_customers_affected=("total_customers_affected", "max"),
    ).reset_index()
    agg["avg_risk"] = agg["avg_risk"].round(1)
    agg["feeder_name"] = agg["feeder_id"]
    return agg


if __name__ == "__main__":
    ar = compute_asset_risk()
    print("Toplam varlik:", len(ar))
    print(ar["risk_category"].value_counts())
    print()
    print(ar.sort_values("risk_score", ascending=False)[
        ["pole_id", "feeder_id", "ilce", "bolge_tipi", "risk_score", "risk_category"]
    ].head(10))
