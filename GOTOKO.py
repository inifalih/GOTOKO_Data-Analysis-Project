from operator import attrgetter
import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="GOTOKO — User Retention Cohort",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Styling CSS tema cerah (Sky, Mint, Peach & Clean White)
st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(180deg, #f0f9ff 0%, #f8fafc 45%, #ecfdf5 100%);
        }
        .hero-banner {
            background: linear-gradient(120deg, #e0f2fe 0%, #dcfce7 55%, #fef3c7 100%);
            border: 1px solid #bae6fd;
            color: #0f172a;
            padding: 24px 28px;
            border-radius: 18px;
            margin-bottom: 20px;
            box-shadow: 0 6px 20px rgba(14, 165, 233, 0.08);
        }
        .hero-badge {
            display: inline-block;
            background: #ffffff;
            border: 1px solid #93c5fd;
            color: #0369a1;
            font-size: 0.78rem;
            font-weight: 700;
            padding: 4px 12px;
            border-radius: 999px;
            margin-bottom: 10px;
            letter-spacing: 0.04em;
        }
        .hero-title {
            font-size: 1.9rem;
            font-weight: 800;
            margin: 0 0 6px 0;
            color: #0f172a;
        }
        .hero-sub {
            font-size: 0.98rem;
            color: #334155;
            margin: 0;
        }
        .kpi-card {
            background: #ffffff;
            border-radius: 16px;
            padding: 16px 18px;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
            border: 1px solid #e2e8f0;
            height: 100%;
        }
        .kpi-card.sky {
            background: linear-gradient(180deg, #ffffff 0%, #f0f9ff 100%);
            border-top: 4px solid #0ea5e9;
        }
        .kpi-card.mint {
            background: linear-gradient(180deg, #ffffff 0%, #ecfdf5 100%);
            border-top: 4px solid #10b981;
        }
        .kpi-card.peach {
            background: linear-gradient(180deg, #ffffff 0%, #fff7ed 100%);
            border-top: 4px solid #f97316;
        }
        .kpi-card.lavender {
            background: linear-gradient(180deg, #ffffff 0%, #f5f3ff 100%);
            border-top: 4px solid #8b5cf6;
        }
        .kpi-label {
            font-size: 0.8rem;
            font-weight: 700;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 6px;
        }
        .kpi-value {
            font-size: 1.55rem;
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 4px;
        }
        .kpi-sub {
            font-size: 0.82rem;
            color: #475569;
        }
        .panel-card {
            background: #ffffff;
            border-radius: 16px;
            padding: 18px 20px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.04);
            margin-bottom: 16px;
        }
        .insight-pill {
            background: #f0f9ff;
            border: 1px solid #bae6fd;
            border-left: 4px solid #0ea5e9;
            border-radius: 12px;
            padding: 12px 14px;
            margin-bottom: 10px;
            color: #0f172a;
            font-size: 0.9rem;
            line-height: 1.45;
        }
        .insight-pill.mint {
            background: #ecfdf5;
            border-color: #a7f3d0;
            border-left-color: #10b981;
        }
        .insight-pill.peach {
            background: #fff7ed;
            border-color: #fed7aa;
            border-left-color: #f97316;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Banner Header Cerah
st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-badge">✨ MY SKILL • DATA ANALYSIS PROJECT</div>
        <div class="hero-title">GOTOKO — User Retention Cohort Analysis</div>
        <div class="hero-sub">Analisis retensi bulanan pengguna berdasarkan bulan transaksi pertama (First Order Month) dan jarak periode (Month Number).</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- SIDEBAR ---
st.sidebar.header("🎛️ Kontrol & Filter")
default_file = "Salinan Online Retail Data.csv"
uploaded_file = st.sidebar.file_uploader(
    "Upload file CSV (opsional)", type=["csv"]
)

st.sidebar.subheader("🧹 Data Cleansing")
drop_cancelled = st.sidebar.checkbox(
    "Hapus order batal (order_id diawali 'C')", value=True
)
drop_non_positive = st.sidebar.checkbox(
    "Hapus quantity <= 0 atau price <= 0", value=True
)


@st.cache_data
def load_csv(source) -> pd.DataFrame:
  data = pd.read_csv(source, header=0)
  data.columns = [col.strip().lower() for col in data.columns]
  return data


csv_source = (
    uploaded_file
    if uploaded_file is not None
    else (default_file if os.path.exists(default_file) else None)
)

if csv_source is None:
  st.error(
      f"File **`{default_file}`** tidak ditemukan di `{os.getcwd()}`. Silakan"
      " upload CSV di sidebar."
  )
  st.stop()

df = load_csv(csv_source)

required_cols = {"customer_id", "order_date", "order_id"}
if not required_cols.issubset(set(df.columns)):
  st.error(
      f"Kolom wajib tidak lengkap. Ditemukan: {list(df.columns)}. Wajib ada:"
      " `customer_id`, `order_date`, `order_id`."
  )
  st.stop()

# 1. Data cleansing
df_clean = df.dropna(subset=["customer_id", "order_date"]).copy()
if drop_cancelled and "order_id" in df_clean.columns:
  df_clean = df_clean[~df_clean["order_id"].astype(str).str.startswith("C")]
if drop_non_positive and "quantity" in df_clean.columns:
  df_clean = df_clean[df_clean["quantity"] > 0]
if drop_non_positive and "price" in df_clean.columns:
  df_clean = df_clean[df_clean["price"] > 0]

df_clean["order_date"] = pd.to_datetime(df_clean["order_date"], errors="coerce")
df_clean = df_clean.dropna(subset=["order_date"]).copy()
df_clean["customer_id"] = df_clean["customer_id"].astype(str)
df_clean["year_month"] = df_clean["order_date"].dt.to_period("M")

# 2. Agregat transaksi bulanan per pengguna & tentukan cohort
df_user_monthly = df_clean.groupby(
    ["customer_id", "year_month"], as_index=False
).agg(order_cnt=("order_id", "nunique"))

df_user_monthly["cohort"] = (
    df_user_monthly.groupby("customer_id")["year_month"].transform("min")
)

# 3. Hitung jarak bulan (+ 1)
df_user_monthly["period_num"] = (
    df_user_monthly["year_month"] - df_user_monthly["cohort"]
).apply(attrgetter("n")) + 1

# 4. Tabel pivot jumlah pengguna unik
df_cohort_pivot = pd.pivot_table(
    df_user_monthly,
    index="cohort",
    columns="period_num",
    values="customer_id",
    aggfunc=pd.Series.nunique,
).sort_index()

# Filter rentang cohort
all_cohorts = list(df_cohort_pivot.index)
cohort_labels = [str(c) for c in all_cohorts]
selected_range = st.sidebar.select_slider(
    "📅 Rentang First Order Month",
    options=cohort_labels,
    value=(cohort_labels[0], cohort_labels[-1]),
)

start_idx = cohort_labels.index(selected_range[0])
end_idx = cohort_labels.index(selected_range[1]) + 1
selected_cohorts = all_cohorts[start_idx:end_idx]

df_cohort_pivot_view = df_cohort_pivot.loc[selected_cohorts]

# 5. Hitung rasio retensi relatif terhadap bulan ke-1
cohort_size = df_cohort_pivot_view.iloc[:, 0]
df_retention_cohort = df_cohort_pivot_view.divide(cohort_size, axis=0)

# --- KPI CARDS CERAH ---
total_users = int(cohort_size.sum())
top_cohort = cohort_size.idxmax()
top_cohort_val = int(cohort_size.max())
avg_m2 = (
    df_retention_cohort[2].mean() * 100
    if 2 in df_retention_cohort.columns
    else np.nan
)
avg_m2_str = f"{avg_m2:.1f}%" if pd.notna(avg_m2) else "-"

c1, c2, c3, c4 = st.columns(4)
with c1:
  st.markdown(
      f"""
        <div class="kpi-card sky">
            <div class="kpi-label">Total Pengguna (Cohort Terpilih)</div>
            <div class="kpi-value">{total_users:,}</div>
            <div class="kpi-sub">Pelanggan unik terakuisisi</div>
        </div>
        """,
      unsafe_allow_html=True,
  )
with c2:
  st.markdown(
      f"""
        <div class="kpi-card mint">
            <div class="kpi-label">Cohort Terbesar</div>
            <div class="kpi-value">{top_cohort_val:,} ({top_cohort})</div>
            <div class="kpi-sub">Bulan transaksi pertama tertinggi</div>
        </div>
        """,
      unsafe_allow_html=True,
  )
with c3:
  st.markdown(
      f"""
        <div class="kpi-card peach">
            <div class="kpi-label">Rata-rata Retensi Bulan ke-2</div>
            <div class="kpi-value">{avg_m2_str}</div>
            <div class="kpi-sub">Kembali bertransaksi pada Month 2</div>
        </div>
        """,
      unsafe_allow_html=True,
  )
with c4:
  st.markdown(
      f"""
        <div class="kpi-card lavender">
            <div class="kpi-label">Jumlah Cohort</div>
            <div class="kpi-value">{len(selected_cohorts)} bulan</div>
            <div class="kpi-sub">Rentang {selected_range[0]} s.d. {selected_range[1]}</div>
        </div>
        """,
      unsafe_allow_html=True,
  )

st.write("")

# --- VISUALISASI & RINGKASAN ---
left_col, right_col = st.columns([1.5, 1], gap="large")

with left_col:
  st.markdown(
      '<div class="panel-card"><h4 style="margin-top:0;color:#0f172a;">🌈 Heatmap Cohort Size & User Retention Cohort</h4>',
      unsafe_allow_html=True,
  )
  with sns.axes_style("white"):
    fig, ax = plt.subplots(
        1,
        2,
        figsize=(12, max(6, len(selected_cohorts) * 0.55)),
        sharey=True,
        gridspec_kw={"width_ratios": [1.2, 11]},
    )

    sns.heatmap(
        df_retention_cohort,
        annot=True,
        fmt=".0%",
        cmap="RdYlGn",
        vmin=0.0,
        vmax=1.0,
        linewidths=0.6,
        linecolor="#ffffff",
        ax=ax[1],
    )
    ax[1].set_title(
        "User Retention Cohort", fontsize=13, fontweight="bold", pad=10
    )
    ax[1].set(xlabel="Month Number", ylabel="")

    df_cohort_size = pd.DataFrame(cohort_size)
    white_cmap = mcolors.ListedColormap(["#f0f9ff"])
    sns.heatmap(
        df_cohort_size,
        annot=True,
        cbar=False,
        fmt="g",
        cmap=white_cmap,
        linewidths=0.6,
        linecolor="#bae6fd",
        ax=ax[0],
    )
    ax[0].tick_params(bottom=False)
    ax[0].set(
        xlabel="Cohort Size", ylabel="First Order Month", xticklabels=[]
    )

    fig.tight_layout()
    st.pyplot(fig)
  st.markdown("</div>", unsafe_allow_html=True)

with right_col:
  st.markdown(
      '<div class="panel-card"><h4 style="margin-top:0;color:#0f172a;">💡 Insight & Tren Retensi</h4>',
      unsafe_allow_html=True,
  )

  best_m2_text = "Belum ada data Month 2"
  if 2 in df_retention_cohort.columns and df_retention_cohort[2].notna().any():
    best_cohort = df_retention_cohort[2].idxmax()
    best_pct = df_retention_cohort[2].max() * 100
    best_m2_text = (
        f"Cohort <strong>{best_cohort}</strong> unggul di bulan ke-2 dengan"
        f" retensi <strong>{best_pct:.0f}%</strong>."
    )

  st.markdown(
      f"""
        <div class="insight-pill mint">
            <strong>Akuisisi Puncak:</strong> Sebanyak <strong>{top_cohort_val:,}</strong> pengguna baru memulai transaksi pada <strong>{top_cohort}</strong>.
        </div>
        <div class="insight-pill">
            <strong>Performa Bulan ke-2:</strong> {best_m2_text}
        </div>
        <div class="insight-pill peach">
            <strong>Evaluasi Retensi:</strong> Pantau penurunan pada periode lanjut untuk merancang promo repeat order atau reminder pelanggan.
        </div>
        """,
      unsafe_allow_html=True,
  )

  st.markdown("**Rata-Rata Retensi per Month Number (%)**")
  avg_retention = (df_retention_cohort.mean(skipna=True) * 100).round(2)
  st.area_chart(avg_retention, height=200, color="#0ea5e9")
  st.markdown("</div>", unsafe_allow_html=True)

# --- TABEL DATA ---
st.markdown(
    '<div class="panel-card"><h4 style="margin-top:0;color:#0f172a;">📋 Tabel Detail Analisis</h4>',
    unsafe_allow_html=True,
)
t1, t2, t3 = st.tabs([
    "Persentase Retensi (%)",
    "Jumlah Pengguna Unik",
    "Summary User Monthly",
])

with t1:
  pct_df = (df_retention_cohort * 100).round(1)
  pct_df.index = pct_df.index.astype(str)

  # Fungsi pewarnaan sel agar nilai kosong (NaN) tetap putih bersih, bukan hitam
  pastel_cmap = mcolors.LinearSegmentedColormap.from_list(
      "pastel_retention", ["#fff7ed", "#fef9c3", "#dcfce7", "#34d399"]
  )

  def style_retention_table(data: pd.DataFrame):
    styler = data.style.background_gradient(
        cmap=pastel_cmap, vmin=0, vmax=100, axis=None
    )

    def highlight_nan(val):
      if pd.isna(val):
        return "background-color: #ffffff; color: #94a3b8;"
      return "color: #0f172a; font-weight: 500;"

    return styler.map(highlight_nan).format("{:.1f}%", na_rep="-")

  st.dataframe(style_retention_table(pct_df), use_container_width=True)

with t2:
  cnt_df = df_cohort_pivot_view.copy()
  cnt_df.index = cnt_df.index.astype(str)
  st.dataframe(cnt_df, use_container_width=True)

with t3:
  monthly_df = df_user_monthly[
      df_user_monthly["cohort"].isin(selected_cohorts)
  ].copy()
  monthly_df["year_month"] = monthly_df["year_month"].astype(str)
  monthly_df["cohort"] = monthly_df["cohort"].astype(str)
  st.dataframe(monthly_df.head(500), use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)