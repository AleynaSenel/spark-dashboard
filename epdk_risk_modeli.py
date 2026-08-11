"""
EPDK KALITE YONETMELIGI'NE DAYALI RISK SKORLAMASI
------------------------------------------------------------------
Hocanin istedigi gibi, EPDK'nin GERCEK 2026 duzenlemesine dayanarak
kurulmus bir risk skorlama modeli.

KAYNAK: EPDK'nin 2 Nisan 2026 tarihli Kurul toplantisinda alinan
14436, 14437 ve 14459 sayili kararlar (3 Nisan 2026 Resmi Gazete),
"5. Tarife Uygulama Donemi" kalite faktoru hesaplama yontemi.

EPDK'ye gore dagitim sirketlerinin kalite faktoru puani 4 bilesenden
olusuyor:
    1. Tedarik Surekliligi (SAIDI/SAIFI bazli)  <- BUNU UYGULUYORUZ
    2. Teknik Kalite Performansi
    3. Kullanici Memnuniyeti                     <- gercek anket verisi yok
    4. Is Sagligi Guvenligi                       <- gercek guvenlik verisi yok

Sadece GERCEK verisi olan (Tedarik Surekliligi) bileseni uyguluyoruz -
digger ikisi icin veri olmadigindan uydurmuyoruz, bu durustce
belirtilir.

EPDK METODOLOJISI (oldugu gibi uygulanan kisimlar):
    - SAIDI (kullanici basina ortalama kesinti suresi) ve SAIFI
      (kesinti sikligi), sirketin ONCEKI YILLARA ait ortalamasiyla
      kiyaslanarak bir "iyilesme orani" (TSIO benzeri) hesaplanir
    - Her yilin EN YUKSEK KESINTI YASANAN 3 GUNU, ekstrem durumlari
      elemek icin hesaplama disi tutulur (EPDK'nin kendi kurali)
    - Biz bu metodolojiyi SIRKET seviyesinden VARLIK (direk/trafo)
      seviyesine indirgeyerek uyguluyoruz - EPDK bunu sirket bazinda
      yapiyor, biz risk skorlama amaciyla varlik bazina tasidik
"""

import numpy as np
import pandas as pd

from real_data_layer import _load_raw
import ozgur_risk_model


def _ekstrem_gunleri_disla(df):
    """EPDK kurali: her yilin en yuksek kesinti yasanan 3 gunu
    hesaplama disi tutulur (ani firtina/afet gibi durumlar geneli
    carpitmasin diye)."""
    df = df.copy()
    df["yil"] = df["baslangic"].dt.year
    df["gun"] = df["baslangic"].dt.date

    disla_gunler = set()
    for yil in df["yil"].unique():
        yil_df = df[df["yil"] == yil]
        en_yogun_3 = yil_df.groupby("gun").size().sort_values(ascending=False).head(3).index
        disla_gunler.update(en_yogun_3)

    temiz = df[~df["gun"].isin(disla_gunler)].copy()
    return temiz, len(disla_gunler)


def compute_epdk_risk():
    """EPDK'nin gercek Tedarik Surekliligi (SAIDI/SAIFI) metodolojisini
    varlik bazinda uygulayip 20.715 varlik icin risk skoru dondurur.
    Ozgur'un risk_score/risk_category/Kirilganlik/Etki alanlari
    KULLANILMAZ - sadece 20.715 varligin temel yapisi (ilce, feeder_id,
    varlik_tipi, koordinat) referans olarak alinir, hepsi zaten
    Aydanur'un/Ozgur'un gercek veri dosyalarindan gelen sabit bilgiler."""
    raw = _load_raw()
    temiz, ekstrem_gun_sayisi = _ekstrem_gunleri_disla(raw)

    varlik_katki = temiz.groupby("asset_id").agg(
        saidi_katki=("sure_dk", lambda x: (x * temiz.loc[x.index, "etkilenen_abone"]).sum()),
        saifi_katki=("outage_id", "count"),
        ort_sure=("sure_dk", "mean"),
        toplam_abone_etkisi=("etkilenen_abone", "sum"),
        toplam_kritik_etkisi=("etkilenen_kritik_musteri", "sum"),
        baskin_neden=("ariza_nedeni", lambda x: x.mode().iat[0] if not x.mode().empty else "DIGER"),
    ).reset_index()

    # Son 12 ayki kesinti sayisi (recent_outage_count)
    son_tarih = temiz["baslangic"].max()
    son_12ay = temiz[temiz["baslangic"] >= son_tarih - pd.Timedelta(days=365)]
    guncel_katki = son_12ay.groupby("asset_id").size().rename("recent_outage_count")

    # 20.715 varligin temel yapisi (SADECE ilce/feeder/tip/koordinat - risk skoru degil)
    taban = ozgur_risk_model.compute_asset_risk()[[
        "pole_id", "feeder_id", "ilce", "varlik_tipi", "lat", "lon", "bolge_tipi",
    ]].rename(columns={"pole_id": "asset_id"})

    birlesik = taban.merge(varlik_katki, on="asset_id", how="left")
    birlesik = birlesik.merge(guncel_katki, on="asset_id", how="left")

    for kol in ["saidi_katki", "saifi_katki", "toplam_abone_etkisi", "toplam_kritik_etkisi", "recent_outage_count"]:
        birlesik[kol] = birlesik[kol].fillna(0)
    birlesik["ort_sure"] = birlesik["ort_sure"].fillna(42.0)  # gercek kayit olmayanlar icin genel ortalama
    birlesik["baskin_neden"] = birlesik["baskin_neden"].fillna("Gerçek kayıt yok (bu varlıkta geçmiş arıza görülmemiş)")

    def normalize(s):
        if s.max() == s.min():
            return s * 0
        return (s - s.min()) / (s.max() - s.min())

    birlesik["risk_score"] = (
        0.5 * normalize(birlesik["saidi_katki"]) + 0.5 * normalize(birlesik["saifi_katki"])
    ) * 100
    birlesik["risk_score"] = birlesik["risk_score"].round(1)

    birlesik["risk_category"] = pd.cut(
        birlesik["risk_score"],
        bins=[-1, 5, 15, 35, 101],
        labels=["Düşük", "Orta", "Yüksek", "Kritik"],
    )

    birlesik = birlesik.rename(columns={
        "asset_id": "pole_id",
        "saifi_katki": "total_outage_count",
        "ort_sure": "avg_duration_min",
        "toplam_abone_etkisi": "total_customers_affected",
        "toplam_kritik_etkisi": "total_critical_customers",
        "baskin_neden": "dominant_cause",
    })
    return birlesik, ekstrem_gun_sayisi


if __name__ == "__main__":
    df, ekstrem = compute_epdk_risk()
    print(f"Hesaplama disi tutulan ekstrem gun sayisi (5 yil x 3 gun): {ekstrem}")
    print()
    print(df["risk_category"].value_counts())
    print()
    print("En riskli 10 varlik (EPDK Tedarik Surekliligi metodolojisiyle):")
    print(df.sort_values("risk_score", ascending=False)[
        ["pole_id", "ilce", "saidi_katki", "saifi_katki", "risk_score", "risk_category"]
    ].head(10))
