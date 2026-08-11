"""
GERCEK ANALITIK KATMAN (Analytics Layer) - GERCEK VERIYLE
-------------------------------------------------------------
Aydanur'un gercek kesinti veri setinden turetilen varlik bazinda
0-100 risk skoru hesaplar.

Simulasyondan farki: burada "yas" veya "malzeme" gibi statik
oznitelikler degil, GERCEK GECMIS ARIZA DAVRANISI kullaniliyor:
    - Son 12 aydaki arıza sıklığı        (%30)
    - Toplam arıza sayısı (5 yıllık)      (%15)
    - Ortalama kesinti süresi             (%15)
    - Etkilenen abone sayısı              (%15)
    - Kritik müşteri etkisi (hastane vb.) (%15)
    - Son arızadan bu yana geçen süre     (%10, yakın zamanlı = yuksek risk)
"""

import numpy as np
import pandas as pd

WEIGHTS = {
    "recent_freq": 0.30,
    "total_freq": 0.15,
    "duration": 0.15,
    "customers": 0.15,
    "critical": 0.15,
    "recency": 0.10,
}


def _normalize(series):
    lo, hi = series.min(), series.max()
    if hi - lo == 0:
        return series * 0
    return (series - lo) / (hi - lo)


def compute_asset_risk(asset_df):
    df = asset_df.copy()

    recent_score = _normalize(df["recent_outage_count"])
    total_score = _normalize(df["total_outage_count"])
    duration_score = _normalize(df["avg_duration_min"])
    customers_score = _normalize(df["total_customers_affected"])
    critical_score = _normalize(df["total_critical_customers"])
    # Yakinlik: az gun once ariza olmussa risk yuksek (ters orantili)
    recency_score = 1 - _normalize(df["days_since_last_outage"])

    df["risk_score"] = 100 * (
        WEIGHTS["recent_freq"] * recent_score +
        WEIGHTS["total_freq"] * total_score +
        WEIGHTS["duration"] * duration_score +
        WEIGHTS["customers"] * customers_score +
        WEIGHTS["critical"] * critical_score +
        WEIGHTS["recency"] * recency_score
    )
    df["risk_score"] = df["risk_score"].round(1)

    df["risk_category"] = pd.cut(
        df["risk_score"],
        bins=[-1, 25, 45, 65, 101],
        labels=["Düşük", "Orta", "Yüksek", "Kritik"],
    )
    return df


def compute_feeder_risk(asset_risk_df):
    agg = asset_risk_df.groupby("feeder_id").agg(
        avg_risk=("risk_score", "mean"),
        max_risk=("risk_score", "max"),
        asset_count=("pole_id", "count"),
        critical_assets=("risk_category", lambda x: (x == "Kritik").sum()),
        high_assets=("risk_category", lambda x: (x == "Yüksek").sum()),
        total_customers_affected=("total_customers_affected", "sum"),
    ).reset_index()
    agg["avg_risk"] = agg["avg_risk"].round(1)
    agg["feeder_name"] = agg["feeder_id"]
    return agg
