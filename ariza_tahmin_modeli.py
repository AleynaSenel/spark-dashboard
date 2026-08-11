"""
GERCEK ARIZA TAHMIN MODELI (Analitik Katman - Geleceğe Dönük Tahmin)
------------------------------------------------------------------------
Gokalp'in henuz gelmeyen dosyasi yerine, Aydanur'un gercek kesinti
verisinden (2021-2025, 8.387 kayit, 119 fider) DOGRUDAN kendi tahmin
modelimizi kuruyoruz.

Fark onemli: Ozgur'un modeli GECMISE bakip "bu varlik ne kadar riskli"
diyor (teshis). Bu model ise fider bazinda GUNLUK zaman serisi
olusturup "onumuzdeki 7 gun icinde bu fiderde arıza olur mu" sorusuna
cevap arıyor (tahmin) - yani gercekten ileriye donuk.

Yontem:
    1. Her fider icin gunluk (2021-01-01 - 2025-12-31) ariza sayisi
       zaman serisi olusturulur (ariza olmayan gunler 0 ile doldurulur)
    2. Her gun icin geriye donuk ozellikler hesaplanir (son 30/90/365
       gundeki ariza sayisi, son arızadan bu yana gecen gun, mevsim)
    3. Hedef degisken: onumuzdeki 7 gun icinde en az 1 ariza olur mu (0/1)
    4. Egitim: 2021-2024 verisi, Test: 2025 verisi (gercek zaman ayrimi,
       veri sizintisi yok)
    5. RandomForestClassifier ile egitim, AUC ile degerlendirme
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

from real_data_layer import _load_raw

ROLLING_WINDOWS = [30, 90, 365]
HORIZON_DAYS = 7


def _build_daily_feeder_series(df):
    """Her fider x her gun icin ariza sayisi (0 dahil) tam takvim."""
    df = df.copy()
    df["tarih"] = df["baslangic"].dt.normalize()
    gunluk = df.groupby(["feeder_id", "tarih"]).size().rename("ariza_sayisi").reset_index()

    tum_tarihler = pd.date_range(df["tarih"].min(), df["tarih"].max(), freq="D")
    fiderler = df["feeder_id"].unique()
    tam_index = pd.MultiIndex.from_product([fiderler, tum_tarihler], names=["feeder_id", "tarih"])

    tam = gunluk.set_index(["feeder_id", "tarih"]).reindex(tam_index, fill_value=0).reset_index()
    tam = tam.sort_values(["feeder_id", "tarih"]).reset_index(drop=True)
    return tam


def _build_features(tam):
    tam = tam.copy()
    for w in ROLLING_WINDOWS:
        tam[f"son_{w}g_ariza"] = (
            tam.groupby("feeder_id")["ariza_sayisi"]
            .transform(lambda x: x.rolling(w, min_periods=1).sum())
        )

    def gun_sayisi_son_ariza(grup):
        gun = np.full(len(grup), np.nan)
        son_ariza_idx = -1
        for i, deger in enumerate(grup.values):
            if deger > 0:
                son_ariza_idx = i
            gun[i] = i - son_ariza_idx if son_ariza_idx >= 0 else 999
        return gun

    tam["son_arizadan_gun"] = tam.groupby("feeder_id")["ariza_sayisi"].transform(
        lambda x: pd.Series(gun_sayisi_son_ariza(x), index=x.index)
    )
    tam["ay"] = tam["tarih"].dt.month
    tam["mevsim_kis"] = tam["ay"].isin([12, 1, 2]).astype(int)
    tam["mevsim_yaz"] = tam["ay"].isin([6, 7, 8]).astype(int)

    # ileriye donuk hedef: onumuzdeki 7 gunde ariza var mi
    tam["gelecek_7g_ariza"] = (
        tam.groupby("feeder_id")["ariza_sayisi"]
        .transform(lambda x: x.shift(-1).rolling(HORIZON_DAYS, min_periods=1).sum())
    )
    tam["HEDEF"] = (tam["gelecek_7g_ariza"] > 0).astype(int)
    return tam


def train_and_predict():
    df = _load_raw()
    tam = _build_daily_feeder_series(df)
    tam = _build_features(tam)

    # Son HORIZON_DAYS gun icin hedef hesaplanamaz (gelecek veri yok) - cikar
    tam = tam[tam["tarih"] <= tam["tarih"].max() - pd.Timedelta(days=HORIZON_DAYS)]

    feature_cols = [f"son_{w}g_ariza" for w in ROLLING_WINDOWS] + [
        "son_arizadan_gun", "mevsim_kis", "mevsim_yaz",
    ]

    train = tam[tam["tarih"].dt.year <= 2024]
    test = tam[tam["tarih"].dt.year == 2025]

    X_train, y_train = train[feature_cols], train["HEDEF"]
    X_test, y_test = test[feature_cols], test["HEDEF"]

    model = RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=20, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)

    test = test.copy()
    test["tahmin_olasilik"] = y_pred_proba

    # Her fider icin en son (en guncel) tahmini al
    guncel = test.sort_values("tarih").groupby("feeder_id").tail(1)
    guncel = guncel[["feeder_id", "tarih", "tahmin_olasilik"] + feature_cols].reset_index(drop=True)
    guncel["tahmin_olasilik_yuzde"] = (guncel["tahmin_olasilik"] * 100).round(1)
    guncel = guncel.sort_values("tahmin_olasilik_yuzde", ascending=False).reset_index(drop=True)

    onem = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)

    return guncel, round(auc, 3), onem


if __name__ == "__main__":
    guncel, auc, onem = train_and_predict()
    print("AUC:", auc)
    print()
    print("Ozellik onem sirasi:")
    print(onem)
    print()
    print("En yuksek riskli 10 fider (onumuzdeki 7 gun):")
    print(guncel[["feeder_id", "tahmin_olasilik_yuzde"]].head(10))
