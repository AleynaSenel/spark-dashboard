# -*- coding: utf-8 -*-
"""
ozgur_risk_v4.py
=================
Özgür'ün gerçek risk skorlama modeli (v4-corrected) — 06_VARLIK_TABLOSU(varliklar).csv
üzerinde çalışacak şekilde uyarlanmıştır. Metodoloji, ağırlıklar ve eşikler
Özgür'ün gönderdiği 09_risk_skorlama_v4.py dosyasıyla BİREBİR AYNIDIR;
tek değişiklik, kaynağın .md yerine gerçek .csv varlık tablosu olmasıdır
(Özgür'ün orijinal dosyası .md formatında bir varlık tablosu bekliyordu,
ama gönderdiği gerçek dosya .csv idi).

ÖNEMLİ NOT — EPDK UYUMLULUĞU HAKKINDA (Özgür'ün kendi notu):
EPDK, bireysel varlık (direk/trafo) bazında kamuya açık bir "risk skorlama
formülü" YAYINLAMAMAKTADIR. Bu modelde "EPDK uyumluluğu", EPDK'nin fiilen
izlediği 3 süreklilik göstergesiyle (OKSIK/SAIFI, OKSÜRE/SAIDI, AENS) eşdeğer
varlık-seviyesi büyüklükler kullanılarak UYGULANMIŞTIR — bu EPDK'nin resmi
formülü değil, mühendislik yaklaşımıdır.

METODOLOJİ: Risk_Skoru = Olasılık_Puan × Etki_Puan / 100  (ISO 31000: Risk = P × I)

Olasılık_Puan (Kırılganlık) — Özgür'ün görev tanımındaki 4 ZORUNLU boyut:
    Arıza Geçmişi     ağırlık 0.45
    Yaş               ağırlık 0.10
    Çevresel Riskler  ağırlık 0.20
    Yüklenme Oranı    ağırlık 0.25  (vekil: varlık başına abone)

Etki_Puan — EPDK'nin 3 süreklilik göstergesine hizalı:
    Kullanıcı Etkisi (SAIFI-benzeri)      ağırlık 0.35
    Kesinti Süresi Etkisi (SAIDI-benzeri) ağırlık 0.35
    Enerji Kaybı Etkisi (AENS-benzeri)    ağırlık 0.30

Kategori eşikleri (v4-corrected, gerçek dağılımın yüzdelik dilimlerine göre):
    Risk >= 16      -> Kritik
    10 <= Risk < 16 -> Yüksek
    5  <= Risk < 10 -> Orta
    Risk < 5        -> Düşük
"""

import numpy as np
import pandas as pd

from real_data_layer import ILCE_NAME_FIX

SRC = "gercek_veri_csv/06_VARLIK_TABLOSU(varliklar).csv"


def to_num(s, default=0.0):
    return pd.to_numeric(s, errors="coerce").fillna(default)


def norm(series):
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(50.0, index=series.index)
    return (series - mn) / (mx - mn) * 100.0


def kategori(risk):
    if risk >= 16:
        return "Kritik"
    if risk >= 10:
        return "Yüksek"
    if risk >= 5:
        return "Orta"
    return "Düşük"


def build_model():
    df = pd.read_csv(SRC, sep=';', encoding='windows-1254', decimal=',', low_memory=False)
    df["ilce"] = df["ilce"].replace(ILCE_NAME_FIX)

    df["yas_2025"] = to_num(df["yas_2025"])
    df["ariza_sayisi_5yil"] = to_num(df["ariza_sayisi_5yil"])
    df["ariza_sayisi_son2yil"] = to_num(df["ariza_sayisi_son2yil"])
    df["agac_yogunlugu_skoru"] = to_num(df["agac_yogunlugu_skoru"])
    df["kiyi_mesafesi_km"] = to_num(df["kiyi_mesafesi_km"])
    df["varlik_basina_abone"] = to_num(df["varlik_basina_abone"])
    df["fider_abone_sayisi"] = to_num(df["fider_abone_sayisi"])
    df["fider_kritik_musteri"] = to_num(df["fider_kritik_musteri"])
    df["toplam_kesinti_dk"] = to_num(df["toplam_kesinti_dk"])
    df["toplam_kayip_kWh"] = to_num(df["toplam_kayip_kWh"])

    df["ters_kiyi_mesafesi"] = df["kiyi_mesafesi_km"].max() - df["kiyi_mesafesi_km"]

    ariza_alt = 0.75 * norm(df["ariza_sayisi_son2yil"]) + 0.25 * norm(df["ariza_sayisi_5yil"])
    cevresel_alt = 0.5 * norm(df["agac_yogunlugu_skoru"]) + 0.5 * norm(df["ters_kiyi_mesafesi"])

    w_ariza, w_yas, w_cevresel, w_yuklenme = 0.45, 0.10, 0.20, 0.25

    olasilik = (
        w_ariza * ariza_alt
        + w_yas * norm(df["yas_2025"])
        + w_cevresel * cevresel_alt
        + w_yuklenme * norm(df["varlik_basina_abone"])
    )
    df["Olasilik_Puan"] = olasilik.clip(0, 100)

    etki = (
        0.35 * norm(df["fider_abone_sayisi"])
        + 0.35 * norm(df["toplam_kesinti_dk"])
        + 0.30 * norm(df["toplam_kayip_kWh"])
    )
    df["Etki_Puan"] = etki.clip(0, 100)

    df["Risk_Skoru"] = (df["Olasilik_Puan"] * df["Etki_Puan"] / 100.0).round(4)
    df["Risk_Kategorisi"] = df["Risk_Skoru"].apply(kategori)
    df["Olasilik_Puan"] = df["Olasilik_Puan"].round(3)
    df["Etki_Puan"] = df["Etki_Puan"].round(3)

    df = df.sort_values("Risk_Skoru", ascending=False).reset_index(drop=True)
    return df


def compute_asset_risk():
    """Dashboard'un beklediği standart semaya cevrilmis, Ozgur'un
    GERCEK v4 risk modelinin ciktisi. Gercek GPS koordinatlari
    (enlem/boylam) ve gercek baskin ariza nedeni de dahildir."""
    df = build_model()
    df = df.rename(columns={
        "asset_id": "pole_id",
        "asset_type": "varlik_tipi",
        "enlem": "lat",
        "boylam": "lon",
        "Risk_Skoru": "risk_score",
        "Risk_Kategorisi": "risk_category",
        "ariza_sayisi_5yil": "total_outage_count",
        "ariza_sayisi_son2yil": "recent_outage_count",
        "ort_kesinti_dk": "avg_duration_min",
        "fider_abone_sayisi": "total_customers_affected",
        "fider_kritik_musteri": "total_critical_customers",
        "baskin_ariza_nedeni": "dominant_cause",
    })
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
    agg["avg_risk"] = agg["avg_risk"].round(2)
    agg["feeder_name"] = agg["feeder_id"]
    return agg


if __name__ == "__main__":
    df = build_model()
    print("Toplam varlik:", len(df))
    print(df["Risk_Kategorisi"].value_counts())
    print()
    print(df[["asset_id", "asset_type", "feeder_id", "ilce", "Olasilik_Puan",
              "Etki_Puan", "Risk_Skoru", "Risk_Kategorisi"]].head(15).to_string())
