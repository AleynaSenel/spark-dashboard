"""
GERCEK KARAR DESTEK SISTEMI (Decision Support Layer) - GERCEK VERIYLE
--------------------------------------------------------------------
Risk skorlarini varlik tipine (direk/trafo) gore tahmini maliyetle
ve gercek gecmis etkiyle (etkilenen abone, kesinti suresi) birlestirip
bakim onceliklendirme listesi uretir.

MALIYETLER: TEDAS'in resmi "2026 Yili Birim Fiyat Kitabi"ndan alinan
GERCEK fiyatlardir (Aydanur'un sagladigi 5_2026_Yili_TEDAS_BFK.xlsx):

- DIREK (beton direk, temelden egik durumun duzeltilmesi - montaj bedeli):
    Sehir/koy ici AG-OG sebekesi (kentsel): 5.955,26 TL
    Buyuk aralikli hat (kirsal):            9.303,73 TL
  (Kaynak: "TEMELDEN EGIK DURUMDAKI BETON DIREKLERIN DUZELTILMESI" bolumu,
   bolge_tipi alanina gore secilir)

- DAGITIM_TRAFOSU (100 kVA, 6.3/0.4 kV, direk tipi, hermetik, bakir sargili
  transformator - malzeme + montaj bedeli toplami):
    263.554,23 TL (malzeme) + 49.927,02 TL (montaj) = 313.481,25 TL
  (Kaynak: "HERMETIK (TAM KAPALI) DIREK TIPI TRANSFORMATORLER" bolumu;
   gercek veri setinde trafo kVA degeri bulunmadigindan en yaygin
   dagitim trafosu boyutu olan 100 kVA referans alinmistir)
"""

import numpy as np
import pandas as pd

# Beton direk - "temelden egik durumun duzeltilmesi" montaj bedeli (bolge tipine gore)
DIREK_COST_BY_BOLGE = {"kentsel": 5955.26, "kirsal": 9303.73}

# 100 kVA direk tipi hermetik transformator - malzeme + montaj (TL)
TRAFO_COST_TL = 263554.23 + 49927.02


def estimate_cost(row):
    if row["varlik_tipi"] == "DAGITIM_TRAFOSU":
        return TRAFO_COST_TL
    return DIREK_COST_BY_BOLGE.get(row.get("bolge_tipi"), 5955.26)


def build_priority_list(asset_risk_df, top_n=None):
    df = asset_risk_df.copy()
    df["est_cost_tl"] = df.apply(estimate_cost, axis=1)

    # Beklenen fayda: gecmiste bu varlik yuzunden kaybedilen musteri-dakikasi
    # (etkilenen abone x ortalama sure) - bu, onarilirsa onlenecek gelecekteki
    # olasi kayiplarin bir gostergesi olarak kullanilir
    df["gecmis_musteri_dk_kaybi"] = df["total_customers_affected"] * df["avg_duration_min"]
    df["benefit_cost_score"] = (df["gecmis_musteri_dk_kaybi"] / df["est_cost_tl"]).round(2)

    # ONEMLI: risk_category bir metin sutunu oldugu icin alfabetik siralanirsa
    # "Yuksek" harf sirasi geregi "Kritik"ten once gelir - bu yanlis bir
    # onceliklendirmeye yol acar. Gercek onem sirasini elle tanimliyoruz.
    ONEM_SIRASI = {"Kritik": 0, "Yüksek": 1, "Orta": 2, "Düşük": 3}
    df["_onem_sira"] = df["risk_category"].astype(str).map(ONEM_SIRASI).fillna(9).astype(int)

    # DÜZELTME: Sıralama kriterlerine risk_score eklendi (azalan sırada)
    df = df.sort_values(["_onem_sira", "risk_score", "benefit_cost_score"], ascending=[True, False, False])
    df = df.drop(columns=["_onem_sira"])
    df["priority_rank"] = range(1, len(df) + 1)

    cols = [
        "priority_rank", "pole_id", "feeder_id", "ilce", "varlik_tipi",
        "risk_score", "risk_category", "recent_outage_count", "total_outage_count",
        "est_cost_tl", "benefit_cost_score",
    ]
    result = df[cols].reset_index(drop=True)
    if top_n:
        result = result.head(top_n)
    return result


def recommend_action(risk_category):
    mapping = {
        "Kritik": "Acil saha muayenesi + 30 gun icinde mudahale",
        "Yüksek": "90 gun icinde planli bakim",
        "Orta": "Yillik periyodik bakim programina dahil et",
        "Düşük": "Rutin izleme, ek aksiyon gerekmiyor",
    }
    return mapping.get(str(risk_category), "Degerlendirilmedi")