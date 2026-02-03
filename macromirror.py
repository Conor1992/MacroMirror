# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import requests
import yfinance as yf
import matplotlib.pyplot as plt

from functools import reduce
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw


# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.header("MacroMirror Settings")

fred_api_key = st.sidebar.text_input("FRED API Key", type="password")

analysis_window = st.sidebar.number_input(
    "Analysis Window (days)",
    min_value=60,
    max_value=2000,
    value=252
)

similarity_model = st.sidebar.selectbox(
    "Similarity Model",
    [
        "Z‑Score Distance",
        "Correlation Pattern",
        "PCA Distance",
        "K‑Means Cluster Match",
        "Dynamic Time Warping"
    ]
)

chart_type = st.sidebar.selectbox(
    "Chart Type",
    ["Static (Matplotlib)", "Interactive (Streamlit)"]
)

run_button = st.sidebar.button("Run Analysis")


# ---------------------------------------------------
# FEATURE GROUPS
# ---------------------------------------------------

FEATURE_GROUPS = {
    "Rates": ["FedFunds", "SOFR", "TBill_3m", "TBill_6m"],
    "Benchmark Yields": ["Yield_2y", "Yield_5y", "Yield_10y", "Yield_30y"],
    "Curve Slopes": ["Slope_2s10s", "Slope_3m10y"],
    "Real Yields": ["Real_5y", "Real_10y", "Real_30y"],
    "Corporate Credit": ["IG_Spread", "HY_Spread", "BBB_Spread"],
    "Dollar Index": ["DXY"],
    "Major FX Pairs": ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCNY", "USDMXN", "USDBRL"],
    "Major Indices": ["SPX", "NDX", "DJIA", "RUT"],
    "Volatility": ["VIX"]
}

selected_groups = st.sidebar.multiselect(
    "Select Feature Groups for Similarity",
    list(FEATURE_GROUPS.keys()),
    default=["Rates", "Benchmark Yields", "Major Indices"]
)


# ---------------------------------------------------
# MAIN TITLE
# ---------------------------------------------------

st.title("📊 MacroMirror — US Macro Analogue Finder")
st.write("A macro‑analytics dashboard for detecting historical similarity using ML and visualisation.")


# ---------------------------------------------------
# RUN BUTTON LOGIC
# ---------------------------------------------------

if not run_button:
    st.info("Set your API key and options in the sidebar, then click **Run Analysis**.")
    st.stop()

if not fred_api_key:
    st.error("Please enter your FRED API key.")
    st.stop()


# ---------------------------------------------------
# DATA INGESTION — FRED
# ---------------------------------------------------

FRED_SERIES = {
    "FedFunds": "FEDFUNDS",
    "SOFR": "SOFR",
    "TBill_3m": "DTB3",
    "TBill_6m": "DTB6",
    "Yield_2y": "DGS2",
    "Yield_5y": "DGS5",
    "Yield_10y": "DGS10",
    "Yield_30y": "DGS30",
    "Slope_2s10s": "T10Y2Y",
    "Slope_3m10y": "T10Y3M",
    "Real_5y": "DFII5",
    "Real_10y": "DFII10",
    "Real_30y": "DFII30",
    "IG_Spread": "BAMLC0A0CM",
    "HY_Spread": "BAMLH0A0HYM2",
    "BBB_Spread": "BAMLC0A4CBBB",
}

def fred_series(series_id):
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {"series_id": series_id, "api_key": fred_api_key, "file_type": "json"}
    r = requests.get(url, params=params)
    data = r.json()

    if "observations" not in data:
        return pd.DataFrame(columns=["date", series_id])

    df = pd.DataFrame(data["observations"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    return df[["date", "value"]].rename(columns={"value": series_id})


fred_dfs = []
for name, sid in FRED_SERIES.items():
    df = fred_series(sid)
    df.rename(columns={sid: name}, inplace=True)
    fred_dfs.append(df)

macro_fred = reduce(lambda l, r: pd.merge(l, r, on="date", how="outer"), fred_dfs)


# ---------------------------------------------------
# DATA INGESTION — YAHOO FINANCE
# ---------------------------------------------------

YAHOO_TICKERS = {
    "DXY": "DX-Y.NYB",
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "JPY=X",
    "USDCHF": "CHF=X",
    "USDCNY": "CNY=X",
    "USDMXN": "MXN=X",
    "USDBRL": "BRL=X",
    "SPX": "^GSPC",
    "NDX": "^NDX",
    "DJIA": "^DJI",
    "RUT": "^RUT",
    "VIX": "^VIX",
}

def yahoo_series(ticker, col_name):
    df = yf.download(ticker, start="2000-01-01", progress=False, auto_adjust=True)
    if df.empty:
        return pd.DataFrame(columns=["date", col_name])

    series = df["Adj Close"] if "Adj Close" in df.columns else df["Close"]
    series = series.reset_index().rename(columns={"Date": "date", "Adj Close": col_name, "Close": col_name})
    return series

yahoo_dfs = []
for name, ticker in YAHOO_TICKERS.items():
    yahoo_dfs.append(yahoo_series(ticker, name))

yahoo_data = reduce(lambda l, r: pd.merge(l, r, on="date", how="outer"), yahoo_dfs)


# ---------------------------------------------------
# MERGE + PREPROCESSING
# ---------------------------------------------------

macro_full = pd.merge(macro_fred, yahoo_data, on="date", how="outer")
macro_full.sort_values("date", inplace=True)
macro_full.set_index("date", inplace=True)

# PREPROCESSING PIPELINE
# 1. Drop columns with >50% NaNs
macro_full = macro_full.loc[:, macro_full.isna().mean() < 0.5]

# 2. Forward-fill then back-fill
macro_full = macro_full.ffill().bfill()

# 3. Drop any remaining all-NaN rows
macro_full = macro_full.dropna(how="all")

# 4. Ensure numeric dtype
macro_full = macro_full.apply(pd.to_numeric, errors="coerce")


# ---------------------------------------------------
# BUILD FEATURE SET
# ---------------------------------------------------

selected_features = []
for g in selected_groups:
    selected_features.extend(FEATURE_GROUPS[g])

selected_features = [f for f in selected_features if f in macro_full.columns]

if len(selected_features) == 0:
    st.error("No valid features selected.")
    st.stop()


# ---------------------------------------------------
# CATEGORY CHARTS
# ---------------------------------------------------

def plot_category_static(df, cols, title):
    st.subheader(f"📈 {title}")
    fig, ax = plt.subplots(figsize=(10, 4))
    df[cols].plot(ax=ax)
    ax.set_title(title)
    ax.grid(True)
    st.pyplot(fig)

def plot_category_interactive(df, cols, title):
    st.subheader(f"📈 {title}")
    st.line_chart(df[cols])


for cat, cols in FEATURE_GROUPS.items():
    available = [c for c in cols if c in macro_full.columns]
    if not available:
        continue

    subset = macro_full[available].dropna().tail(1000)
    if subset.empty:
        st.warning(f"No data available for {cat}")
        continue

    if chart_type == "Static (Matplotlib)":
        plot_category_static(subset, available, cat)
    else:
        plot_category_interactive(subset, available, cat)


# ---------------------------------------------------
# SIMILARITY ENGINE
# ---------------------------------------------------

st.subheader("🔍 Most Similar Historical Periods")

def compute_similarity(current_df, hist_df, model):

    common = current_df.columns.intersection(hist_df.columns)
    a = current_df[common].values
    b = hist_df[common].values

    if model == "Z‑Score Distance":
        return 1 / (1 + np.linalg.norm(a - b))

    if model == "Correlation Pattern":
        corr_a = np.corrcoef(a.T)
        corr_b = np.corrcoef(b.T)
        return 1 / (1 + np.linalg.norm(corr_a - corr_b))

    if model == "PCA Distance":
        pca = PCA(n_components=3)
        pca.fit(np.vstack([a, b]))
        a_p = pca.transform(a)[-1]
        b_p = pca.transform(b)[-1]
        return 1 / (1 + np.linalg.norm(a_p - b_p))

    if model == "K‑Means Cluster Match":
        kmeans = KMeans(n_clusters=5, n_init=10)
        kmeans.fit(np.vstack([a, b]))
        labels = kmeans.labels_
        return 1.0 if labels[-1] == labels[0] else 0.1

    if model == "Dynamic Time Warping":
        distance, _ = fastdtw(a, b, dist=euclidean)
        return 1 / (1 + distance)

    return 0


# ---------------------------------------------------
# FIND MOST SIMILAR PERIOD
# ---------------------------------------------------

current_period = macro_full[selected_features].iloc[-analysis_window:]

scores = []
for i in range(len(macro_full) - analysis_window):
    window = macro_full[selected_features].iloc[i:i+analysis_window]
    score = compute_similarity(current_period, window, similarity_model)
    scores.append((macro_full.index[i], score))

scores = sorted(scores, key=lambda x: x[1], reverse=True)
best_date, best_score = scores[0]

st.metric(
    label="Most Similar Period",
    value=str(best_date.date()),
    delta=f"{best_score*100:.1f}% match"
)
