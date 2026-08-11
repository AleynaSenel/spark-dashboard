"""
DASHBOARD (GERCEK VERI VERSIYONU)
------------------------------------
Aydanur'un gercek OMS kesinti veri seti (03_TEMIZ_VERI(temiz_kesintiler).csv) ile calisir.
Simule veri degil - 2021-2025 arasi 4047 gercek kesinti kaydi kullanilir.

Calistirmak icin:
    streamlit run app_gercek.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import os
import base64
from datetime import datetime

from real_data_layer import generate_all_data, ILCE_COORDS
from real_decision_support import build_priority_list, recommend_action
from ozgur_risk_v4 import compute_asset_risk, compute_feeder_risk

try:
    from ariza_tahmin_modeli import train_and_predict
    AI_MODEL_VAR = True
except ImportError:
    AI_MODEL_VAR = False

try:
    # pyrefly: ignore [missing-import]
    from gorsel_tespit_loader import load_gorsel_tespit
    GORSEL_TESPIT_VAR = True
except ImportError:
    GORSEL_TESPIT_VAR = False

try:
    # pyrefly: ignore [missing-import]
    from ultralytics import YOLO
    from PIL import Image
    YOLO_VAR = True
except ImportError:
    YOLO_VAR = False

@st.cache_resource
def load_yolo_model():
    if YOLO_VAR:
        try:
            return YOLO("assets/best.pt")
        except Exception as e:
            return None
    return None

st.set_page_config(
    page_title="SPARK 2026 - Mersin Önleyici Bakım",
    page_icon="⚡",
    layout="wide",
)

st.markdown("""
<style>
    .stApp { background-color: #ffffff; }
    section[data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #eee0a8; }
    section[data-testid="stSidebar"] * { color: #1a1a1a !important; }
    section[data-testid="stSidebar"] h1 { color: #1a1a1a !important; }
    div[data-testid="stMetric"] {
        background-color: white; border-radius: 10px; padding: 14px 10px;
        border: 1px solid #eee0a8; border-left: 4px solid #ffc600;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    div[data-testid="stMetric"] label { color: #6b6b6b !important; }
    h1 { color: #1a1a1a !important; font-weight: 700; }
    h2, h3 { color: #1a1a1a !important; }
    .hero-banner {
        background: linear-gradient(90deg, #ffc600 0%, #ffdd55 100%);
        padding: 32px 28px; border-radius: 12px; margin-bottom: 18px;
        display: flex; align-items: center; justify-content: center; gap: 24px;
        text-align: center;
    }
    .hero-logo {
        background: #1a1a1a; color: #ffc600; font-weight: 800; font-size: 20px;
        border-radius: 8px; padding: 8px 14px; white-space: nowrap;
    }
    .hero-banner h1 { color: #1a1a1a !important; margin: 0; font-size: 40px; }
    .hero-banner p { color: #4a3d00; margin: 4px 0 0 0; font-size: 14px; text-align: center; }
    .real-badge {
        display: inline-block; background: #1a1a1a; color: #ffc600; font-size: 12px;
        font-weight: 700; padding: 3px 10px; border-radius: 20px; margin-left: 10px;
    }
    .stTabs [data-baseweb="tab"] { font-weight: 600; }
    .stTabs [aria-selected="true"] { color: #a8880a !important; }

    /* Multiselect etiketleri (chip) - sari zemin, koyu yazi */
    span[data-baseweb="tag"] {
        background-color: #ffc600 !important;
        color: #1a1a1a !important;
        border-radius: 6px !important;
    }
    span[data-baseweb="tag"] svg { fill: #1a1a1a !important; }
    div[data-baseweb="select"] > div {
        border-color: #eee0a8 !important;
    }

    /* =========================================
       RESPONSIVE MEDIA QUERIES (MOBILE-FRIENDLY)
       ========================================= */

    /* Yana Kaymayı (Yatay Scroll) Engelleme */
    .stApp {
        overflow-x: hidden;
    }

    /* Görsellerin ve iframe/canvas'ların ekrandan taşmasını engelle */
    img, iframe, canvas {
        max-width: 100% !important;
        height: auto;
    }

    /* --- TABLET VE DAR EKRANLAR (max-width: 768px) --- */
    @media (max-width: 768px) {
        
        /* Yan yana duran (flex) Hero Banner'ı alt alta diz */
        .hero-banner {
            flex-direction: column !important;
            padding: 20px 16px;
            gap: 16px;
        }
        
        /* Font boyutlarını orantılı küçült */
        .hero-banner h1 {
            font-size: 28px !important;
        }
        
        .hero-logo {
            font-size: 16px;
            padding: 6px 10px;
        }
        
        .real-badge {
            display: block;
            margin: 10px auto 0;
            width: fit-content;
        }
        
        /* Tabloların (Dataframe) ana ekranı taşırmasını engelle, kendi içinde scroll olsun */
        .stDataFrame {
            width: 100% !important;
            max-width: 100%;
            overflow-x: auto;
        }

        /* Hero Banner'daki Logoyu (Enerjisa) mobilde iyice küçült */
        .hero-banner img {
            height: 120px !important;
            width: auto !important;
        }

        /* Streamlit JS'in inline stillerini ve kendi medyasını KESİN OLARAK ezmek için spesifiklik (specificity) artırıldı */
        html body div[data-testid="stHorizontalBlock"],
        html body div[data-testid="stColumns"] {
            display: flex !important;
            flex-direction: row !important;
            flex-wrap: wrap !important;
            overflow-x: hidden !important;
            gap: 10px !important;
        }
        
        html body div[data-testid="stHorizontalBlock"] > div,
        html body div[data-testid="stColumns"] > div {
            width: calc(50% - 5px) !important;
            flex: 0 0 calc(50% - 5px) !important;
            min-width: calc(50% - 5px) !important;
            max-width: calc(50% - 5px) !important;
            margin-bottom: 12px !important;
        }

        /* Metriklerdeki çok uzun sayıların (örn: Tahmini Maliyet) sonunun "..." ile kesilmesini engelle */
        div[data-testid="stMetricValue"], div[data-testid="stMetricValue"] > div {
            font-size: 1.4rem !important;
            white-space: normal !important;
            word-break: break-word !important;
            overflow: visible !important;
        }
    }

    /* --- KÜÇÜK TELEFONLAR (max-width: 480px) --- */
    @media (max-width: 480px) {
        
        .hero-banner h1 {
            font-size: 26px !important;
        }
        
        .hero-banner p {
            font-size: 12px;
        }
        
        .hero-logo {
            font-size: 14px;
        }
        
        /* Metrik kartlarının iç boşluklarını daralt */
        div[data-testid="stMetric"] {
            padding: 10px 8px;
        }
        
        /* Genel başlık boyutlarını mobil için ez */
        h1 { font-size: 1.5rem !important; }
        h2 { font-size: 1.3rem !important; }
        h3 { font-size: 1.1rem !important; }
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    data = generate_all_data()
    asset_risk = compute_asset_risk()
    feeder_risk = compute_feeder_risk(asset_risk)
    priority = build_priority_list(asset_risk)
    return data, asset_risk, feeder_risk, priority


if AI_MODEL_VAR:
    @st.cache_data
    def load_ai_predictions():
        return train_and_predict()


data, asset_risk_df, feeder_risk_df, priority_df = load_data()
outages_raw = data["outages_raw"]

RISK_COLORS = {
    "Kritik": [200, 30, 30],
    "Yüksek": [235, 140, 30],
    "Orta": [235, 200, 30],
    "Düşük": [60, 160, 80],
}

VARLIK_COLORS = {
    "DIREK": [30, 100, 220],           # mavi
    "DAGITIM_TRAFOSU": [10, 30, 90],   # lacivert
}

# ------------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------------
st.sidebar.image("assets/toroslar_logo.png", use_container_width=True)

search_term = st.sidebar.text_input("🔍 Varlık No Ara", placeholder="ör. MRS-F-019-T002")

ilce_options = sorted(asset_risk_df["ilce"].unique())
selected_ilce = st.sidebar.multiselect("İlçe Seç", ilce_options, default=ilce_options)

if selected_ilce:
    ilce_varlik_sayisi_sb = asset_risk_df[asset_risk_df["ilce"].isin(selected_ilce)].groupby("ilce").size().sort_values(ascending=False)
    st.sidebar.markdown("**Seçili İlçelerde Toplam Varlık**")
    for ilce_adi, sayi in ilce_varlik_sayisi_sb.items():
        st.sidebar.markdown(
            f"<div style='display:flex; justify-content:space-between; font-size:14px; padding:2px 0;'>"
            f"<span>{ilce_adi}</span><span style='font-weight:700; color:#a8880a;'>{sayi}</span></div>",
            unsafe_allow_html=True,
        )
    st.sidebar.markdown(
        f"<div style='display:flex; justify-content:space-between; font-size:14px; padding:6px 0; "
        f"border-top:1px solid #eee0a8; margin-top:4px;'><b>Toplam</b>"
        f"<b style='color:#a8880a;'>{ilce_varlik_sayisi_sb.sum()}</b></div>",
        unsafe_allow_html=True,
    )

varlik_options = sorted(asset_risk_df["varlik_tipi"].unique())
selected_varlik = st.sidebar.multiselect("Varlık Tipi", varlik_options, default=varlik_options)

risk_cat_options = ["Kritik", "Yüksek", "Orta", "Düşük"]
selected_cats = st.sidebar.multiselect("Risk Kategorisi", risk_cat_options, default=risk_cat_options)

min_risk = st.sidebar.slider("Minimum Risk Skoru", 0, 100, 0)

sadece_arizali = st.sidebar.checkbox(
    "Sadece gerçek arıza kaydı olan varlıklar", value=False,
    help="İşaretlenirse, Aydanur'un verisinde en az 1 kez arıza kaydı olan 6.278 varlık gösterilir; işaretlenmezse tüm 20.715 varlık gösterilir.",
)

filtered = asset_risk_df[
    asset_risk_df["ilce"].isin(selected_ilce)
    & asset_risk_df["varlik_tipi"].isin(selected_varlik)
    & asset_risk_df["risk_category"].isin(selected_cats)
    & (asset_risk_df["risk_score"] >= min_risk)
]
if sadece_arizali:
    arizali_idler = set(outages_raw["asset_id"].unique())
    filtered = filtered[filtered["pole_id"].isin(arizali_idler)]
if search_term:
    filtered = filtered[filtered["pole_id"].str.contains(search_term.strip(), case=False, na=False)]

# ------------------------------------------------------------------
# BASLIK VE KPI'LAR
# ------------------------------------------------------------------
with open("assets/enerjisa_logo.png", "rb") as _f:
    _logo_b64 = base64.b64encode(_f.read()).decode()

st.markdown(f"""
<div class="hero-banner">
    <img src="data:image/png;base64,{_logo_b64}" style="height:180px; width:auto;" />
    <div>
        <h1>Mersin Dağıtım Bölgesi</h1>
    </div>
</div>
""", unsafe_allow_html=True)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Toplam Varlık", len(filtered))
k2.metric("Kritik Risk", int((filtered["risk_category"] == "Kritik").sum()))
k3.metric("Yüksek Risk", int((filtered["risk_category"] == "Yüksek").sum()))
k4.metric("Ortalama Risk Skoru", f"{filtered['risk_score'].mean():.1f}" if len(filtered) else "-")
total_cost = priority_df[priority_df["pole_id"].isin(filtered["pole_id"])]["est_cost_tl"].sum()
k5.metric("Tahmini Bakım Maliyeti", f"{total_cost:,.0f} TL")

st.markdown("---")

# ------------------------------------------------------------------
# HARITA
# ------------------------------------------------------------------
st.subheader("📍 Bölgesel Risk Haritası")

map_df = filtered.copy()
map_df["epdk_kategori"] = map_df["risk_category"].astype(str)
map_df["color"] = map_df["epdk_kategori"].map(RISK_COLORS).apply(lambda x: x if isinstance(x, list) else [150, 150, 150])

# HARITA KONUMU: Ozgur'un dosyasindaki gercek enlem/boylam bazen kiyi
# cizgisine cok yakin dustugu icin (bazi noktalar denize taşiyormus gibi
# gorunuyordu), harita gorseli icin ONCEKI guvenli yontemi kullaniyoruz:
# ilce merkezi + kucuk, karada kalacak sekilde sinirlandirilmis jitter.
# NOT: Bu SADECE haritadaki gorsel konum icindir - risk skoru, kategori
# ve tum diger degerler Ozgur'un gercek v4 modelinden degismeden geliyor.
_rng = np.random.RandomState(7)
_lat_jitter = _rng.normal(0, 0.012, size=len(map_df))
_lon_jitter = _rng.normal(0, 0.015, size=len(map_df))
_base_coords = map_df["ilce"].apply(lambda i: ILCE_COORDS.get(i, (36.85, 34.6)))
map_df["lat"] = [c[0] for c in _base_coords] + _lat_jitter
map_df["lon"] = [c[1] for c in _base_coords] + _lon_jitter


def _risk_to_color(score):
    if score >= 65:
        return RISK_COLORS["Kritik"]
    if score >= 45:
        return RISK_COLORS["Yüksek"]
    if score >= 25:
        return RISK_COLORS["Orta"]
    return RISK_COLORS["Düşük"]


if "canli_olaylar" not in st.session_state:
    st.session_state.canli_olaylar = []


@st.fragment(run_every="300s")
def harita_paneli():
    if len(map_df):
        point_layer = pdk.Layer(
            "ScatterplotLayer", data=map_df, get_position="[lon, lat]",
            get_fill_color="color", get_radius=25, radius_min_pixels=2, radius_max_pixels=4,
            pickable=True, opacity=0.85,
        )

        # YOGUNLUK KATMANI: nokta yogunlugu (heatmap) dogrudan sol menudeki
        # gercek arıza sayisiyla orantili. Genis yaricap (radius_pixels) kullanilir
        # ki noktalar tek tek "yildiz/cicek" gibi degil, birbirine karisan
        # pürüzsüz bir bulut olarak gorunsun.
        ilce_ariza_sayisi = outages_raw[outages_raw["ilce"].isin(map_df["ilce"].unique())].groupby("ilce").size()
        ilce_varlik_sayisi = map_df.groupby("ilce").size()
        heat_df = map_df.copy()
        heat_df["weight"] = heat_df["ilce"].map(
            lambda i: ilce_ariza_sayisi.get(i, 0) / max(ilce_varlik_sayisi.get(i, 1), 1)
        )
        heat_layer = pdk.Layer(
            "HeatmapLayer", data=heat_df, get_position="[lon, lat]", get_weight="weight",
            radius_pixels=55, intensity=1.0, threshold=0.02, aggregation="MEAN",
            color_range=[
                [237, 248, 233, 40], [186, 228, 179, 90], [116, 196, 118, 140],
                [49, 163, 84, 190], [0, 109, 44, 225], [0, 68, 27, 255],
            ],
        )

        # CANLI VERI: her 5 dakikada bir yeni "simule" arıza olayı üretilir ve
        # haritada ayırt edici (mor halka) bir katman olarak gösterilir
        weights = (asset_risk_df["risk_score"] + 1)
        weights = weights / weights.sum()
        yeni_olay = asset_risk_df.sample(1, weights=weights).iloc[0]
        st.session_state.canli_olaylar.insert(0, {
            "saat": datetime.now().strftime("%H:%M:%S"),
            "pole_id": yeni_olay["pole_id"], "ilce": yeni_olay["ilce"],
            "lat": yeni_olay["lat"], "lon": yeni_olay["lon"],
            "risk_score": yeni_olay["risk_score"],
        })
        st.session_state.canli_olaylar = st.session_state.canli_olaylar[:8]

        canli_df = pd.DataFrame(st.session_state.canli_olaylar)
        canli_layer = pdk.Layer(
            "ScatterplotLayer", data=canli_df, get_position="[lon, lat]",
            get_fill_color=[190, 40, 220], get_line_color=[255, 255, 255],
            get_radius=90, line_width_min_pixels=2, stroked=True,
            pickable=True, opacity=0.55,
        )

        view_state = pdk.ViewState(
            latitude=map_df["lat"].mean(), longitude=map_df["lon"].mean(), zoom=8.5,
        )
        tooltip = {
            "html": "<b>{pole_id}</b><br/>İlçe: {ilce}<br/>Risk: {risk_score} ({risk_category})<br/>Baskın neden: {dominant_cause}",
            "style": {"backgroundColor": "#1a1a1a", "color": "#ffc600"},
        }
        st.pydeck_chart(pdk.Deck(
            layers=[heat_layer, point_layer, canli_layer], initial_view_state=view_state, tooltip=tooltip,
            map_provider="carto", map_style="light",
        ))
        legend_html = ""
        for cat, color in RISK_COLORS.items():
            legend_html += f"<span style='margin-right: 15px;'><span style='color:rgb({color[0]},{color[1]},{color[2]});'>●</span> {cat}</span>"
        legend_html += "<span><span style='color:rgb(190,40,220)'>◉</span> Canlı olay</span>"
        st.markdown(f"<div style='display:flex; flex-wrap:wrap; margin-bottom:10px;'>{legend_html}</div>", unsafe_allow_html=True)

        st.caption(
            f"🟣 Son canlı olay: {st.session_state.canli_olaylar[0]['saat']} — "
            f"{st.session_state.canli_olaylar[0]['pole_id']} ({st.session_state.canli_olaylar[0]['ilce']}) — "
            "gerçek SCADA bağlantısı yerine simüle edilmiştir, 5 dakikada bir yenilenir."
        )

        st.markdown("**🔴 Canlı Arıza Uyarı Akışı**")

        if "canli_fotolar" not in st.session_state:
            st.session_state.canli_fotolar = {}

        for olay in st.session_state.canli_olaylar:
            olay_id = f"{olay['pole_id']}_{olay['saat']}"
            
            with st.container(border=True):
                c1, c2 = st.columns([2, 1])
                
                with c1:
                    st.markdown(f"**Saat:** {olay['saat']} &nbsp; | &nbsp; **İlçe:** {olay['ilce']}")
                    st.markdown(f"**Varlık:** {olay['pole_id']} &nbsp; | &nbsp; **Risk Skoru:** <span style='color:#ffc600;'>{olay['risk_score']}</span>", unsafe_allow_html=True)
                
                with c2:
                    yuklenen = st.file_uploader(
                        "Fotoğraf yükle", type=["jpg", "jpeg", "png"], key=f"foto_{olay_id}",
                        label_visibility="collapsed",
                    )
                
                if yuklenen is not None:
                    yuklenen.seek(0)
                    st.session_state.canli_fotolar[olay_id] = yuklenen

                if olay_id in st.session_state.canli_fotolar:
                    foto = st.session_state.canli_fotolar[olay_id]
                    
                    if YOLO_VAR:
                        model = load_yolo_model()
                        if model is not None:
                            try:
                                foto.seek(0)
                                image = Image.open(foto)
                                results = model(image)
                                res = results[0]
                                res_plotted = res.plot()
                                res_rgb = res_plotted[..., ::-1]
                                
                                st.image(res_rgb, width=350)
                                
                                if len(res.boxes) > 0:
                                    detected_classes = [res.names[int(cls)] for cls in res.boxes.cls]
                                    unique_classes = list(set(detected_classes))
                                    st.markdown("**Tespit Edilen Problemler:**")
                                    for u_cls in unique_classes:
                                        st.error(f"⚠️ {u_cls}")
                                else:
                                    st.success("✅ Model fotoğrafta bir arıza/sorun tespit etmedi.")
                            except Exception as e:
                                st.error("Görüntü işlenemedi.")
                                st.image(foto, width=150)
                        else:
                            st.image(foto, width=150)
                    else:
                        st.image(foto, width=150)
    else:
        st.info("Seçili filtrelerle gösterilecek varlık yok.")


harita_paneli()

st.markdown("---")

# ------------------------------------------------------------------
# TABLAR
# ------------------------------------------------------------------
tab_labels = [
    "📋 Risk Skoru Tablosu", "🚨 Bakım Öncelik Listesi", "🔌 Fider/İlçe Özeti",
    "📈 Gerçek Arıza Trendleri", "🔎 Kesinti Sorgulama", "📜 Risk Skorlama Modeli",
]
if AI_MODEL_VAR:
    tab_labels.append("🤖 Arıza Tahmin Modeli")
if GORSEL_TESPIT_VAR:
    tab_labels.append("📷 Görsel Tespit (Aleyna)")

all_tabs = st.tabs(tab_labels)
tab1, tab2, tab3, tab4, tab_sorgu, tab_epdk = all_tabs[:6]
extra_tabs = all_tabs[6:]
tab5 = extra_tabs[0] if AI_MODEL_VAR else None
tab6 = extra_tabs[1] if (AI_MODEL_VAR and GORSEL_TESPIT_VAR) else (extra_tabs[0] if (GORSEL_TESPIT_VAR and not AI_MODEL_VAR) else None)

with tab1:
    risk_table = filtered[[
        "pole_id", "feeder_id", "ilce", "varlik_tipi", "total_outage_count",
        "recent_outage_count", "avg_duration_min", "total_customers_affected",
        "dominant_cause", "risk_score", "risk_category",
    ]].sort_values("risk_score", ascending=False)
    st.dataframe(risk_table, use_container_width=True, height=420)
    st.download_button("⬇️ Excel/CSV olarak indir", risk_table.to_csv(index=False).encode("utf-8-sig"),
                        file_name="gercek_risk_skoru.csv", mime="text/csv")

with tab2:
    prio_filtered = priority_df[priority_df["pole_id"].isin(filtered["pole_id"])].copy()
    prio_filtered["onerilen_aksiyon"] = prio_filtered["risk_category"].apply(recommend_action)
    st.dataframe(prio_filtered, use_container_width=True, height=420)
    st.caption("Öncelik: risk kategorisi + fayda/maliyet skoru (geçmiş müşteri-dakika kaybı / tahmini maliyet).")
    st.download_button("⬇️ Excel/CSV olarak indir", prio_filtered.to_csv(index=False).encode("utf-8-sig"),
                        file_name="gercek_bakim_oncelik.csv", mime="text/csv")

with tab3:
    fider_ilce = filtered.groupby("feeder_id")["ilce"].agg(lambda x: x.mode().iat[0] if not x.mode().empty else "Bilinmiyor")
    fdr = feeder_risk_df[feeder_risk_df["feeder_id"].isin(filtered["feeder_id"].unique())].copy()
    fdr["ilce"] = fdr["feeder_id"].map(fider_ilce)
    fdr = fdr[["feeder_id", "ilce", "avg_risk", "max_risk", "asset_count", "critical_assets", "high_assets", "total_customers_affected"]]
    fdr = fdr.rename(columns={"ilce": "İlçe"})
    st.dataframe(fdr.sort_values("avg_risk", ascending=False), use_container_width=True, height=300)
    st.bar_chart(fdr.set_index("feeder_id")["avg_risk"], color="#14406e")

    st.markdown("**İlçe bazlı toplam kesinti sayısı (gerçek veri)**")
    by_ilce = outages_raw[outages_raw["ilce"].isin(filtered["ilce"].unique())]["ilce"].value_counts()
    st.bar_chart(by_ilce, color="#eb8c1e")

with tab4:
    scope_ids = filtered["pole_id"].tolist()
    outages_scope = outages_raw[outages_raw["asset_id"].isin(scope_ids)]

    if len(outages_scope):
        st.markdown("**Aylık kesinti sayısı (2021-2025, gerçek veri)**")
        monthly = outages_scope.groupby("ay").size().sort_index()
        st.line_chart(monthly, color="#eb8c1e")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Arıza nedeni dağılımı**")
            st.bar_chart(outages_scope["ariza_nedeni"].value_counts(), color="#0a2540")
        with col_b:
            st.markdown("**Kırsal / kentsel dağılım**")
            st.bar_chart(outages_scope["bolge_tipi"].value_counts(), color="#14406e")
    else:
        st.info("Seçili filtrelerle gösterilecek kesinti kaydı yok.")

with tab_sorgu:
    st.markdown("**Gerçek kesinti kayıtlarında ayrıntılı sorgulama**")
    st.caption("Aydanur'un gerçek OMS veri setindeki (8.387 kayıt) alanlara göre filtreleme.")

    sq_col1, sq_col2, sq_col3 = st.columns(3)
    with sq_col1:
        yil_opts = sorted(outages_raw["baslangic"].dt.year.unique())
        sec_yil = st.multiselect("Yıl", yil_opts, default=yil_opts, key="sq_yil")
    with sq_col2:
        ilce_opts_sq = sorted(outages_raw["ilce"].unique())
        sec_ilce_sq = st.multiselect("İlçe", ilce_opts_sq, default=ilce_opts_sq, key="sq_ilce")
    with sq_col3:
        kaynak_opts = sorted(outages_raw["ihbar_kanali"].unique())
        sec_kaynak = st.multiselect("Kaynak (İhbar Kanalı)", kaynak_opts, default=kaynak_opts, key="sq_kaynak")

    sq_col4, sq_col5 = st.columns(2)
    with sq_col4:
        neden_opts = sorted(outages_raw["ariza_nedeni"].unique())
        sec_neden = st.multiselect("Neden", neden_opts, default=neden_opts, key="sq_neden")
    with sq_col5:
        sure_min, sure_max = int(outages_raw["sure_dk"].min()), int(outages_raw["sure_dk"].max())
        sec_sure = st.slider("Süre (dakika)", sure_min, sure_max, (sure_min, sure_max), key="sq_sure")

    sorgu_sonuc = outages_raw[
        outages_raw["baslangic"].dt.year.isin(sec_yil)
        & outages_raw["ilce"].isin(sec_ilce_sq)
        & outages_raw["ihbar_kanali"].isin(sec_kaynak)
        & outages_raw["ariza_nedeni"].isin(sec_neden)
        & outages_raw["sure_dk"].between(sec_sure[0], sec_sure[1])
    ]

    st.metric("Bulunan Kayıt Sayısı", len(sorgu_sonuc))
    st.dataframe(
        sorgu_sonuc[[
            "baslangic", "feeder_id", "asset_id", "ilce", "varlik_tipi",
            "ariza_nedeni", "ihbar_kanali", "sure_dk", "etkilenen_abone",
        ]].sort_values("baslangic", ascending=False),
        use_container_width=True, height=400,
    )
    st.download_button(
        "⬇️ Excel/CSV olarak indir", sorgu_sonuc.to_csv(index=False).encode("utf-8-sig"),
        file_name="kesinti_sorgu_sonucu.csv", mime="text/csv",
    )

with tab_epdk:
    st.markdown("**Risk Skorlama Modeli**")
    st.caption(
        "ISO 31000 (Risk = Olasılık × Etki) çerçevesi, görev tanımındaki 4 zorunlu boyut "
        "(Yaş, Arıza Geçmişi, Çevresel Riskler, Yüklenme Oranı) ve EPDK'nin fiilen izlediği "
        "3 süreklilik göstergesi (SAIFI/SAIDI/AENS-benzeri) esas alınarak kurulmuştur."
    )

    with st.expander("ℹ️ Yöntem hakkında"):
        st.markdown("""
**Olasılık Puanı (Kırılganlık)** — 4 zorunlu boyut:
- Arıza Geçmişi — ağırlık %45
- Yaş — ağırlık %10
- Çevresel Riskler (ağaç yoğunluğu + kıyı mesafesi) — ağırlık %20
- Yüklenme Oranı (vekil: varlık başına abone) — ağırlık %25

**Etki Puanı** — EPDK'nin 3 süreklilik göstergesine hizalı:
- Kullanıcı Etkisi (SAIFI-benzeri) — ağırlık %35
- Kesinti Süresi Etkisi (SAIDI-benzeri) — ağırlık %35
- Enerji Kaybı Etkisi (AENS-benzeri) — ağırlık %30

**Risk Skoru = Olasılık Puanı × Etki Puanı / 100**

Önemli not: EPDK, bireysel varlık (direk/trafo) bazında kamuya açık bir risk skorlama
formülü yayınlamamaktadır. Buradaki "EPDK uyumluluğu", EPDK'nin şirket bazında izlediği
göstergelerin varlık seviyesine mühendislik yaklaşımıyla taşınmasıdır — EPDK'nin resmi
formülü değildir.
        """)

    try:
        v4_scoped = filtered

        k1e, k2e, k3e = st.columns(3)
        k1e.metric("Toplam Varlık", f"{len(v4_scoped):,}".replace(",", "."))
        k2e.metric("Kritik", int((v4_scoped["risk_category"] == "Kritik").sum()))
        k3e.metric("Yüksek", int((v4_scoped["risk_category"] == "Yüksek").sum()))

        st.dataframe(
            v4_scoped[[
                "pole_id", "feeder_id", "ilce", "varlik_tipi",
                "total_outage_count", "risk_score", "risk_category", "dominant_cause",
            ]].sort_values("risk_score", ascending=False),
            use_container_width=True, height=400,
        )
        st.download_button(
            "⬇️ Excel/CSV olarak indir", v4_scoped.to_csv(index=False).encode("utf-8-sig"),
            file_name="risk_skorlama_v4.csv", mime="text/csv",
        )
    except Exception as e:
        st.error(f"Hesaplama hatası: {e}")

if tab5 is not None:
    with tab5:
        st.markdown("**Gerçek Arıza Tahmin Modeli — fider bazlı, önümüzdeki 7 gün içinde arıza olasılığı**")
        try:
            ai_preds, auc, onem = load_ai_predictions()
            st.caption(f"Model performansı (AUC): **{auc}**")
            ai_preds_scoped = ai_preds[ai_preds["feeder_id"].isin(filtered["feeder_id"].unique())].copy()

            fider_ilce_map = filtered.groupby("feeder_id")["ilce"].agg(
                lambda x: x.mode().iat[0] if not x.mode().empty else "Bilinmiyor"
            )
            ai_preds_scoped["İlçe"] = ai_preds_scoped["feeder_id"].map(fider_ilce_map)

            st.dataframe(
                ai_preds_scoped[["feeder_id", "İlçe", "tarih", "tahmin_olasilik_yuzde", "son_arizadan_gun", "son_30g_ariza"]],
                use_container_width=True, height=380,
            )
            st.bar_chart(ai_preds_scoped.set_index("feeder_id")["tahmin_olasilik_yuzde"], color="#c81e1e")

            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("**Özellik önem sırası**")
                st.bar_chart(onem, color="#0a2540")
            with col_b:
                st.markdown("**Yöntem**")
                st.caption(
                    "Her fider için günlük zaman serisi kurulup, son 30/90/365 gündeki arıza sayısı, "
                    "son arızadan bu yana geçen gün ve mevsim bilgisiyle RandomForestClassifier eğitildi. "
                    "Hedef: önümüzdeki 7 gün içinde en az 1 arıza olur mu (0/1)."
                )
        except FileNotFoundError:
            st.info("Gerekli veri dosyası bulunamadı.")

if tab6 is not None:
    with tab6:
        st.markdown("**Aleyna'nın görüntü işleme tespitleri (manuel giriş)**")
        gorsel_df = load_gorsel_tespit()
        if len(gorsel_df):
            gorsel_scoped = gorsel_df[gorsel_df["direk_no"].isin(filtered["pole_id"])]
            st.dataframe(gorsel_scoped, use_container_width=True, height=380)
            st.caption(f"Toplam {len(gorsel_df)} görsel tespit kaydı yüklendi.")
        else:
            st.info(
                "Henüz görsel tespit verisi girilmemiş. Aleyna, 'gercek_veri/gorsel_tespit_sablon.xlsx' "
                "şablonunu doldurup aynı isimle bu klasöre koyduğunda, tespitler otomatik olarak burada görünecek."
            )
            st.caption("Şablon sütunları: direk_no, tespit_tarihi, hasar_turu, hasar_skoru_0_100, guven_orani_yuzde, aciklama")

st.markdown("---")