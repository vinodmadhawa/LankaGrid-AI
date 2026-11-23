from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from train import FEATURES, features


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
DATA = ROOT / "data"
LOGO = ROOT / "assets" / "lankagrid-logo.svg"
PRIMARY = "#5B8FF9"
CYAN = "#35C3F3"
TEAL = "#35D0BA"
AMBER = "#F7B267"
NAVY = "#0B1220"


st.set_page_config(
    page_title="LankaGrid AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)


if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "Dark"

with st.sidebar:
    st.radio(
        "Theme",
        ["Dark", "Light"],
        horizontal=True,
        key="theme_mode",
    )


THEMES = {
    "Dark": {
        "bg": "#09111F",
        "sidebar": "linear-gradient(180deg, #0D1728 0%, #09111F 100%)",
        "panel": "#111B2E",
        "panel_2": "#151F33",
        "border": "rgba(148, 163, 184, 0.16)",
        "text": "#EAF2FF",
        "muted": "#98A8C2",
        "card": "linear-gradient(145deg, rgba(21,31,51,.98), rgba(16,27,47,.98))",
        "card_card": "linear-gradient(145deg, rgba(21,31,51,.92), rgba(13,23,40,.92))",
        "plot_panel": "rgba(17,27,46,.58)",
        "plot_bg": "rgba(0,0,0,0)",
        "grid": "rgba(148,163,184,.10)",
        "font": "#C9D6EA",
        "plot_template": "plotly_dark",
    },
    "Light": {
        "bg": "#F4F7FB",
        "sidebar": "linear-gradient(180deg, #FFFFFF 0%, #EDF3FA 100%)",
        "panel": "#FFFFFF",
        "panel_2": "#EEF4FA",
        "border": "rgba(51, 65, 85, 0.16)",
        "text": "#172033",
        "muted": "#5F6B7A",
        "card": "linear-gradient(145deg, #FFFFFF, #F3F7FC)",
        "card_card": "linear-gradient(145deg, #FFFFFF, #EEF4FA)",
        "plot_panel": "rgba(255,255,255,.82)",
        "plot_bg": "rgba(255,255,255,0)",
        "grid": "rgba(51,65,85,.12)",
        "font": "#40516A",
        "plot_template": "plotly_white",
    },
}
theme = THEMES[st.session_state.theme_mode]


theme_css = """
    <style>
    :root {
        --lg-bg: __LG_BG__;
        --lg-panel: __LG_PANEL__;
        --lg-panel-2: __LG_PANEL_2__;
        --lg-border: __LG_BORDER__;
        --lg-text: __LG_TEXT__;
        --lg-muted: __LG_MUTED__;
    }
    .stApp { background: var(--lg-bg); color: var(--lg-text); }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] {
        background: __LG_SIDEBAR__;
        border-right: 1px solid var(--lg-border);
    }
    [data-testid="stSidebar"] * { color: var(--lg-text); }
    [data-testid="stMetric"], .kpi-card {
        background: __LG_CARD__;
        border: 1px solid var(--lg-border);
        border-radius: 16px;
        padding: 18px;
        box-shadow: 0 12px 34px __LG_SHADOW__;
    }
    [data-testid="stMetricLabel"] { color: var(--lg-muted); }
    [data-testid="stMetricValue"] { color: var(--lg-text); }
    [data-testid="stAppViewContainer"] { background: var(--lg-bg); }
    [data-testid="stMainBlockContainer"] { color: var(--lg-text); }
    [data-testid="stMarkdownContainer"], [data-testid="stCaptionContainer"] { color: var(--lg-text); }
    [data-testid="stDataFrame"] { background: var(--lg-panel); border: 1px solid var(--lg-border); }
    [data-testid="stDataFrame"] * { color: var(--lg-text); }
    [data-baseweb="slider"] div { color: var(--lg-text); }
    .lg-title {
        font-size: 2rem; font-weight: 760; letter-spacing: -.03em;
        margin: 0.15rem 0 0.15rem;
    }
    .lg-subtitle { color: var(--lg-muted); margin-bottom: 1.2rem; }
    .lg-section {
        background: rgba(17,27,46,.72); border: 1px solid var(--lg-border);
        border-radius: 18px; padding: 18px; margin-bottom: 18px;
    }
    .lg-card {
        background: __LG_CARD_CARD__;
        border: 1px solid var(--lg-border); border-radius: 16px;
        padding: 16px; min-height: 112px;
        margin: 0.25rem 0.25rem 0.75rem;
    }
    .lg-card-label { color: var(--lg-muted); font-size: .78rem; text-transform: uppercase;
        letter-spacing: .08em; }
    .lg-card-value { color: var(--lg-text); font-size: 1.65rem; font-weight: 760; margin-top: 5px; }
    .lg-card-note { color: var(--lg-muted); font-size: .82rem; margin-top: 5px; }
    .best-row { background: rgba(53,195,243,.08) !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] { background: transparent; border-radius: 10px; }
    .stTabs [aria-selected="true"] { background: rgba(91,143,249,.15); }
    div[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
    .block-container { max-width: 1480px; padding-top: 1.25rem; }
    hr { border-color: var(--lg-border); }
    .lg-brand-row {
        display: flex; align-items: center; gap: 14px; min-height: 72px;
        margin-bottom: 4px;
    }
    .lg-brand-logo {
        width: 58px; height: 58px; flex: 0 0 58px;
        border-radius: 15px;
        box-shadow: 0 10px 26px rgba(53,195,243,.12);
    }
    [data-testid="stHorizontalBlock"] {
        flex-wrap: wrap;
        column-gap: 1rem;
        row-gap: 1rem;
        margin-bottom: 0.75rem;
    }
    [data-testid="column"] {
        min-width: 220px;
        flex: 1 1 220px;
        padding: 0 0.375rem;
    }
    [data-testid="stMetric"] {
        min-height: 126px;
        overflow: hidden;
        margin: 0 0.25rem 0.75rem;
        width: calc(100% - 0.5rem);
    }
    [data-testid="stMetricValue"] {
        font-size: clamp(1.25rem, 2vw, 1.85rem);
        overflow-wrap: anywhere;
    }
    [data-testid="stPlotlyChart"] {
        border: 1px solid var(--lg-border);
        border-radius: 16px;
        padding: 6px;
        background: __LG_PLOT_PANEL__;
        overflow: hidden;
    }
    [data-testid="stDataFrame"] {
        border: 1px solid var(--lg-border);
    }
    [data-testid="stRadio"] div[role="radiogroup"] {
        gap: 4px;
    }
    [data-testid="stRadio"] label {
        min-height: 42px;
        display: flex;
        align-items: center;
        border-radius: 10px;
        padding: 7px 8px;
        margin: 1px 0;
    }
    [data-testid="stRadio"] label:hover {
        background: rgba(91,143,249,.10);
    }
    [data-testid="stDataFrame"] * {
        white-space: nowrap;
    }
    @media (max-width: 900px) {
        .block-container { padding-left: 0.75rem; padding-right: 0.75rem; }
        .lg-title { font-size: 1.55rem; }
        .lg-subtitle { line-height: 1.45; }
        [data-testid="column"] { min-width: 100%; }
        [data-testid="stMetric"] { min-height: 112px; }
        [data-testid="stSidebar"] { min-width: 17rem; }
    }
    @media (max-width: 620px) {
        .lg-brand-row { align-items: flex-start; }
        .lg-brand-logo { width: 48px; height: 48px; flex-basis: 48px; }
        [data-testid="stSidebar"] { width: 15rem !important; }
        [data-testid="stPlotlyChart"] { padding: 2px; border-radius: 12px; }
        .lg-title { font-size: 1.25rem; }
        .lg-subtitle { font-size: 0.82rem; }
        .lg-card { min-height: 96px; padding: 12px; }
        .lg-card-value { font-size: 1.25rem; }
        [data-testid="stMetric"] { min-height: 102px; }
        [data-testid="stTabs"] [role="tab"] { font-size: 0.78rem; padding: 8px 6px; }
        [data-testid="stHorizontalBlock"] { flex-direction: column; }
    }
    @media (max-width: 420px) {
        .block-container { padding-left: 0.45rem; padding-right: 0.45rem; }
        [data-testid="stSidebar"] { width: 13.5rem !important; }
        [data-testid="stMetricValue"] { font-size: 1.05rem; }
        [data-testid="stDataFrame"] { overflow-x: auto; }
    }
    </style>
    """

theme_css = (
    theme_css.replace("__LG_BG__", theme["bg"])
    .replace("__LG_PANEL__", theme["panel"])
    .replace("__LG_PANEL_2__", theme["panel_2"])
    .replace("__LG_BORDER__", theme["border"])
    .replace("__LG_TEXT__", theme["text"])
    .replace("__LG_MUTED__", theme["muted"])
    .replace("__LG_SIDEBAR__", theme["sidebar"])
    .replace("__LG_CARD__", theme["card"])
    .replace("__LG_CARD_CARD__", theme["card_card"])
    .replace("__LG_PLOT_PANEL__", theme["plot_panel"])
    .replace("__LG_SHADOW__", "rgba(2, 8, 23, .10)" if st.session_state.theme_mode == "Light" else "rgba(2, 8, 23, .22)")
)

st.markdown(theme_css, unsafe_allow_html=True)


def chart_layout(title: str, height: int = 430):
    return dict(
        template=theme["plot_template"],
        title=dict(text=title, x=0.02, xanchor="left", font=dict(size=18)),
        height=height,
        paper_bgcolor=theme["plot_bg"],
        plot_bgcolor=theme["plot_bg"],
        font=dict(color=theme["font"]),
        margin=dict(l=12, r=12, t=58, b=12),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )


def kpi_card(label, value, note=""):
    st.markdown(
        f'<div class="lg-card"><div class="lg-card-label">{label}</div>'
        f'<div class="lg-card-value">{value}</div>'
        f'<div class="lg-card-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_artifacts():
    result = {"available": True, "errors": []}
    required = [
        OUTPUTS / "metrics.csv",
        OUTPUTS / "predictions.csv",
        OUTPUTS / "history.joblib",
        OUTPUTS / "metadata.json",
    ]
    missing = [str(p.relative_to(ROOT)) for p in required if not p.exists()]
    if missing:
        result["available"] = False
        result["errors"].append("Missing artifacts: " + ", ".join(missing))
        return result

    try:
        result["metrics"] = pd.read_csv(OUTPUTS / "metrics.csv")
        result["predictions"] = pd.read_csv(
            OUTPUTS / "predictions.csv", parse_dates=["date"]
        )
        history_payload = joblib.load(OUTPUTS / "history.joblib")
        result["history"] = history_payload["history"].copy()
        result["history"].index = pd.to_datetime(result["history"].index)
        result["features"] = history_payload.get("features", FEATURES)
        result["metadata"] = json.loads((OUTPUTS / "metadata.json").read_text())
    except Exception as exc:
        result["available"] = False
        result["errors"].append(f"Could not read model artifacts: {exc}")
    return result


@st.cache_data(show_spinner=False)
def load_dataset():
    data_path = DATA / "merged-generation-profile-2017-2026.csv"
    if not data_path.exists():
        return pd.DataFrame(columns=["date", "total_gwh"])
    df = pd.read_csv(data_path)
    date_col = df.columns[0]
    df["date"] = pd.to_datetime(df[date_col], errors="coerce")
    numeric_columns = [c for c in df.columns if c not in [date_col, "date"]]
    values = df[numeric_columns].replace("Data N/A", np.nan).apply(pd.to_numeric, errors="coerce")
    df["total_gwh"] = values.sum(axis=1).where(values.notna().any(axis=1))
    df = df[["date", "total_gwh"]]
    df = df.sort_values("date").drop_duplicates("date")
    df["total_gwh"] = pd.to_numeric(df["total_gwh"], errors="coerce")
    return df


@st.cache_resource(show_spinner=False)
def load_model(model_name):
    path = OUTPUTS / f"{model_name.lower().replace(' ', '_')}.joblib"
    return joblib.load(path) if path.exists() else None


artifacts = load_artifacts()

with st.sidebar:
    logo_col, title_col = st.columns([0.38, 1], gap="small")
    with logo_col:
        if LOGO.exists():
            st.image(str(LOGO), width=54)
    with title_col:
        st.markdown("### LankaGrid AI")
    st.caption("Sri Lanka National Grid Analytics")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        [
            "Executive Overview",
            "Electricity Data Explorer",
            "Ensemble Learning Lab",
            "Model Performance Comparison",
            "Historical Backtesting",
            "Next-Day Forecast Studio",
            "Explainable AI & Insights",
            "Dataset & Methodology",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("Data scope")
    if artifacts["available"]:
        meta = artifacts["metadata"]
        st.write(f"**Test:** {meta.get('test_start')} – {meta.get('test_end')}")
        st.write("**Unit:** GWh/day")
    else:
        st.warning("Artifacts unavailable")


if not artifacts["available"]:
    st.error("The dashboard could not load the trained model artifacts.")
    for error in artifacts["errors"]:
        st.write(error)
    st.info('Run: `python train.py --csv data/merged-generation-profile-2017-2026.csv --date-col "Date (GMT+5:30)" --target-col total_gwh`')
    st.stop()


metrics = artifacts["metrics"].copy()
predictions = artifacts["predictions"].copy()
history = artifacts["history"].copy()
meta = artifacts["metadata"]
dataset = load_dataset()

if dataset.empty:
    dataset = history.rename("total_gwh").reset_index().rename(columns={"index": "date"})

history = history.sort_index()
history_df = history.rename("total_gwh").reset_index().rename(columns={"index": "date"})
model_columns = [c for c in predictions.columns if c not in ["date", "actual"]]


def date_filtered(df):
    min_date = df["date"].min().date()
    max_date = df["date"].max().date()
    start, end = st.slider(
        "Date range",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
        format="YYYY-MM-DD",
        key=f"date-range-{page}",
    )
    return df[(df["date"].dt.date >= start) & (df["date"].dt.date <= end)].copy()


def line_chart(df, x, y, title, y_title="Generation (GWh/day)", color=None):
    fig = px.line(df, x=x, y=y, color=color)
    fig.update_layout(**chart_layout(title))
    fig.update_xaxes(title="Date", gridcolor=theme["grid"])
    fig.update_yaxes(title=y_title, gridcolor=theme["grid"])
    fig.update_traces(line_width=2)
    return fig


def derived_metrics(prediction_frame):
    rows = []
    actual = prediction_frame["actual"]
    for name in model_columns:
        pred = prediction_frame[name]
        rows.append(
            {
                "model": name,
                "MAE": mean_absolute_error(actual, pred),
                "RMSE": np.sqrt(mean_squared_error(actual, pred)),
                "R2": r2_score(actual, pred),
            }
        )
    return pd.DataFrame(rows).sort_values("MAE")


derived = derived_metrics(predictions)
best_row = metrics.sort_values("MAE").iloc[0]
best_model = best_row["model"]


header_logo, header_text = st.columns([0.09, 1.0], gap="small", vertical_alignment="center")
with header_logo:
    if LOGO.exists():
        st.image(str(LOGO), width=58)
with header_text:
    st.markdown(
        f'<div class="lg-title">{"LankaGrid AI" if page == "Executive Overview" else page}</div>'
        '<div class="lg-subtitle">Sri Lankan national-grid electricity generation analytics · GWh/day · '
        'Generation is an energy-supply proxy, not direct consumer demand</div>',
        unsafe_allow_html=True,
    )


if page == "Executive Overview":
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Daily observations", f"{len(history):,}", f"{history.index.min().date()} – {history.index.max().date()}")
    with c2:
        kpi_card("Models evaluated", str(len(metrics)), "Held-out chronological test")
    with c3:
        kpi_card("Best model", best_model, f"Lowest saved MAE")
    with c4:
        kpi_card("Lowest test MAE", f"{best_row['MAE']:.2f}", "GWh/day")

    c1, c2, c3 = st.columns(3)
    with c1:
        kpi_card("Test RMSE", f"{best_row['RMSE']:.2f}", "GWh/day")
    with c2:
        kpi_card("Test R²", f"{best_row['R2']:.3f}", "Held-out data")
    with c3:
        kpi_card("Latest generation", f"{history.iloc[-1]:.2f}", str(history.index[-1].date()))

    filtered = date_filtered(history_df)
    filtered = filtered.set_index("date")
    filtered["7-day average"] = filtered["total_gwh"].rolling(7, min_periods=1).mean()
    filtered["30-day average"] = filtered["total_gwh"].rolling(30, min_periods=1).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=filtered.index, y=filtered["total_gwh"], mode="lines",
                             name="Daily", line=dict(color=PRIMARY, width=1), opacity=.58))
    fig.add_trace(go.Scatter(x=filtered.index, y=filtered["7-day average"], mode="lines",
                             name="7-day average", line=dict(color=CYAN, width=2)))
    fig.add_trace(go.Scatter(x=filtered.index, y=filtered["30-day average"], mode="lines",
                             name="30-day average", line=dict(color=AMBER, width=2)))
    fig.update_layout(**chart_layout("Daily electricity generation"))
    fig.update_xaxes(title="Date")
    fig.update_yaxes(title="Generation (GWh/day)")
    st.plotly_chart(fig, width="stretch")

    left, right = st.columns(2)
    monthly = history_df.set_index("date").resample("MS")["total_gwh"].sum().reset_index()
    monthly["label"] = monthly["date"].dt.strftime("%Y-%m")
    with left:
        fig = px.bar(monthly, x="date", y="total_gwh", color_discrete_sequence=[CYAN])
        fig.update_layout(**chart_layout("Monthly generation trend", 360))
        fig.update_xaxes(title="Month")
        fig.update_yaxes(title="GWh/month")
        st.plotly_chart(fig, width="stretch")

    temp = history_df.copy()
    temp["weekday"] = temp["date"].dt.day_name()
    temp["period"] = np.where(temp["date"].dt.dayofweek >= 5, "Weekend", "Weekday")
    pattern = temp.groupby("period", as_index=False)["total_gwh"].mean()
    with right:
        fig = px.bar(pattern, x="period", y="total_gwh", color="period",
                     color_discrete_map={"Weekday": PRIMARY, "Weekend": TEAL}, text_auto=".2f")
        fig.update_layout(**chart_layout("Weekday vs weekend average", 360), showlegend=False)
        fig.update_xaxes(title="")
        fig.update_yaxes(title="Average GWh/day")
        st.plotly_chart(fig, width="stretch")

    left, right = st.columns([2, 1])
    annual = history_df.set_index("date").resample("YE")["total_gwh"].mean().reset_index()
    annual["year"] = annual["date"].dt.year.astype(str)
    with left:
        fig = px.area(annual, x="year", y="total_gwh", markers=True,
                      color_discrete_sequence=[TEAL])
        fig.update_layout(**chart_layout("Annual average generation", 330))
        fig.update_xaxes(title="Year")
        fig.update_yaxes(title="Average GWh/day")
        st.plotly_chart(fig, width="stretch")
    with right:
        mean_value = history.mean()
        peak_date = history.idxmax()
        peak_value = history.max()
        st.markdown("### Key findings")
        st.markdown(
            f"- Mean generation: **{mean_value:.2f} GWh/day**\n"
            f"- Peak observed day: **{peak_value:.2f} GWh** on **{peak_date.date()}**\n"
            f"- Lowest-MAE model: **{best_model}**\n"
            f"- Test range: **{meta.get('test_start')}** to **{meta.get('test_end')}**"
        )


elif page == "Electricity Data Explorer":
    c1, c2, c3 = st.columns(3)
    with c1:
        year = st.selectbox("Year", ["All"] + sorted(dataset["date"].dt.year.unique().astype(str).tolist(), reverse=True))
    with c2:
        month_names = ["All", "January", "February", "March", "April", "May", "June",
                       "July", "August", "September", "October", "November", "December"]
        month = st.selectbox("Month", month_names)
    with c3:
        st.write("")
        st.write("")
        st.metric("Available records", f"{len(dataset):,}")

    filtered = date_filtered(dataset)
    if year != "All":
        filtered = filtered[filtered["date"].dt.year == int(year)]
    if month != "All":
        filtered = filtered[filtered["date"].dt.month == month_names.index(month)]

    st.plotly_chart(line_chart(filtered, "date", "total_gwh", "Daily generation explorer"),
                    width="stretch")

    left, right = st.columns(2)
    agg = filtered.set_index("date")
    monthly = agg["total_gwh"].resample("MS").sum().reset_index(name="generation")
    annual = agg["total_gwh"].resample("YE").sum().reset_index(name="generation")
    with left:
        fig = px.bar(monthly, x="date", y="generation", color_discrete_sequence=[PRIMARY])
        fig.update_layout(**chart_layout("Monthly aggregation", 350))
        fig.update_yaxes(title="GWh/month")
        st.plotly_chart(fig, width="stretch")
    with right:
        fig = px.bar(annual, x="date", y="generation", color_discrete_sequence=[TEAL])
        fig.update_layout(**chart_layout("Annual aggregation", 350))
        fig.update_xaxes(tickformat="%Y")
        fig.update_yaxes(title="GWh/year")
        st.plotly_chart(fig, width="stretch")

    left, right = st.columns([2, 1])
    heat = filtered.copy()
    heat["week"] = ((heat["date"] - heat["date"].min()).dt.days // 7)
    heat["weekday"] = heat["date"].dt.day_name()
    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pivot = heat.pivot_table(index="weekday", columns=heat["date"].dt.to_period("M").astype(str),
                             values="total_gwh", aggfunc="mean")
    pivot = pivot.reindex(weekday_order)
    with left:
        fig = px.imshow(pivot, aspect="auto", origin="upper", color_continuous_scale="Turbo",
                        labels={"x": "Month", "y": "Weekday", "color": "GWh/day"})
        fig.update_layout(**chart_layout("Calendar heatmap · monthly weekday mean", 390))
        st.plotly_chart(fig, width="stretch")

    with right:
        st.markdown("### Data quality")
        duplicate_count = int(filtered["date"].duplicated().sum())
        st.metric("Missing total values", int(filtered["total_gwh"].isna().sum()))
        st.metric("Duplicate dates", duplicate_count)
        st.metric("Consecutive-date gaps", max(0, len(pd.date_range(filtered["date"].min(), filtered["date"].max(), freq="D").difference(filtered["date"]))))

    left, right = st.columns(2)
    with left:
        fig = px.histogram(filtered, x="total_gwh", nbins=40, color_discrete_sequence=[CYAN])
        fig.update_layout(**chart_layout("Generation distribution", 350))
        fig.update_xaxes(title="GWh/day")
        fig.update_yaxes(title="Frequency")
        st.plotly_chart(fig, width="stretch")
    with right:
        fig = px.box(filtered, y="total_gwh", points="outliers", color_discrete_sequence=[AMBER])
        fig.update_layout(**chart_layout("Generation box plot", 350))
        fig.update_yaxes(title="GWh/day")
        st.plotly_chart(fig, width="stretch")

    st.markdown("### Dataset preview")
    st.dataframe(filtered.sort_values("date", ascending=False).head(300), width="stretch", hide_index=True)
    st.download_button(
        "Download filtered CSV",
        filtered.to_csv(index=False).encode("utf-8"),
        file_name="lankagrid_filtered_generation.csv",
        mime="text/csv",
    )


elif page == "Ensemble Learning Lab":
    st.markdown("### Ensemble techniques")
    c1, c2, c3, c4 = st.columns(4)
    categories = {
        "Bagging": ("Parallel variance reduction", "Decision Tree · Bagging", PRIMARY),
        "Boosting": ("Sequential error correction", "Gradient Boosting · Histogram Boosting", CYAN),
        "Stacking": ("Learned meta-model combination", "Stacking", TEAL),
        "Hybrid": ("Fixed 50/50 model blends", "Hybrid RF + GB · Hybrid XGB + GB", AMBER),
    }
    for col, (name, details) in zip([c1, c2, c3, c4], categories.items()):
        title, models, color = details
        with col:
            st.markdown(
                f'<div class="lg-card"><div class="lg-card-label">{name}</div>'
                f'<div class="lg-card-value" style="font-size:1.15rem">{title}</div>'
                f'<div class="lg-card-note">{models}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("### Saved model inventory")
    config = [
        ("Decision Tree", "Baseline", "Bagging family", "max_depth=7, min_samples_leaf=4, random_state=42"),
        ("Bagging", "Bagging", "Bagging", "70 estimators; tree max_depth=8, min_samples_leaf=3"),
        ("Random Forest", "Bagging", "Bagging", "100 estimators; min_samples_leaf=3"),
        ("Gradient Boosting", "Boosting", "Boosting", "120 estimators; learning_rate=0.05; max_depth=3"),
        ("Histogram Boosting", "Boosting", "Boosting", "max_iter=120; learning_rate=0.05"),
        ("AdaBoost", "Boosting", "Boosting", "100 estimators; Decision Tree base_depth=4; learning_rate=0.05"),
        ("XGBoost", "Boosting", "Boosting", "200 estimators; learning_rate=0.05; max_depth=4"),
        ("Stacking", "Stacking", "Stacking", "RF 70 + GB 90 base estimators; Ridge alpha=10; cv=3"),
        ("Hybrid RF + GB", "Hybrid", "Hybrid", "50% Random Forest + 50% Gradient Boosting"),
        ("Hybrid XGB + GB", "Hybrid", "Hybrid", "50% XGBoost + 50% Gradient Boosting"),
    ]
    inventory = pd.DataFrame(config, columns=["Model", "Category", "Technique", "Configuration"])
    st.dataframe(inventory, width="stretch", hide_index=True)

    st.markdown("### Training methodology")
    c1, c2 = st.columns(2)
    with c1:
        st.info(
            f"Chronological 80/20 split: {meta.get('train_rows')} training rows and "
            f"{meta.get('test_rows')} held-out rows."
        )
        st.write(
            "Lag features: 1, 2, 3, 7, 14 and 28 days. Rolling mean features use 7 and 14 days; "
            "the 7-day rolling standard deviation and calendar features are also included."
        )
    with c2:
        st.info("No test data is used to fit the component models.")
        st.write(
            "The hybrid is a predefined 50/50 blend. Stacking uses the saved implementation's "
            "ordinary 3-fold cross-validation; it is not rolling-origin stacking."
        )

    selected = st.selectbox("Model selection for saved configuration", metrics["model"].tolist())
    st.success(f"Selected **{selected}**. Model retraining is not triggered by this control.")


elif page == "Model Performance Comparison":
    metric = st.selectbox("Primary metric", ["MAE", "RMSE", "R2"], format_func=lambda x: {"R2": "R²"}.get(x, x))
    ascending = metric != "R2"
    table = metrics.sort_values(metric, ascending=ascending).copy()
    table["Difference from baseline"] = table[metric] - metrics[metric].iloc[-1]

    left, right = st.columns([1.25, 1])
    with left:
        fig = px.bar(table, x="model", y=metric, color="model",
                     color_discrete_sequence=[PRIMARY, CYAN, TEAL, AMBER, "#7A8CA8", "#A67FF2", "#EF6F6C"],
                     text_auto=".3f")
        fig.update_layout(**chart_layout(f"{'R²' if metric == 'R2' else metric} by model", 430), showlegend=False)
        fig.update_xaxes(title="")
        fig.update_yaxes(title={"R2": "R²", "MAE": "GWh/day", "RMSE": "GWh/day"}[metric])
        st.plotly_chart(fig, width="stretch")
    with right:
        comparison = derived.sort_values("MAE")
        st.markdown("### Ranking")
        st.dataframe(
            table.style.format({"MAE": "{:.3f}", "RMSE": "{:.3f}", "R2": "{:.3f}",
                                "Difference from baseline": "{:.3f}"})
                 .apply(lambda row: ["background-color: rgba(53,195,243,.10)" if row["model"] == best_model else "" for _ in row.index], axis=1),
            width="stretch",
            hide_index=True,
        )
        st.caption("Difference from baseline is calculated relative to the final row of the saved metrics table.")

    c1, c2 = st.columns(2)
    category_map = {
        "Decision Tree": "Baseline",
        "Bagging": "Bagging",
        "Random Forest": "Bagging",
        "Gradient Boosting": "Boosting",
        "Histogram Boosting": "Boosting",
        "AdaBoost": "Boosting",
        "XGBoost": "Boosting",
        "Stacking": "Stacking",
        "Hybrid RF + GB": "Hybrid",
        "Hybrid XGB + GB": "Hybrid",
    }
    category_table = metrics.copy()
    category_table["Category"] = category_table["model"].map(category_map)
    category_summary = category_table.groupby("Category", as_index=False)[["MAE", "RMSE", "R2"]].mean()
    with c1:
        fig = px.bar(category_summary, x="Category", y="MAE", color="Category",
                     color_discrete_sequence=[PRIMARY, CYAN, TEAL, AMBER], text_auto=".2f")
        fig.update_layout(**chart_layout("Average MAE by ensemble family", 350), showlegend=False)
        fig.update_yaxes(title="GWh/day")
        st.plotly_chart(fig, width="stretch")
    with c2:
        bubble_metrics = metrics.copy()
        bubble_metrics["bubble_size"] = bubble_metrics["R2"] - bubble_metrics["R2"].min() + 0.2
        fig = px.scatter(bubble_metrics, x="MAE", y="RMSE", size="bubble_size", color="model",
                         hover_data=["R2"], color_discrete_sequence=px.colors.qualitative.Plotly)
        fig.update_layout(**chart_layout("MAE vs RMSE", 350))
        fig.update_xaxes(title="MAE (GWh/day)")
        fig.update_yaxes(title="RMSE (GWh/day)")
        st.plotly_chart(fig, width="stretch")

    validation = derived.round(10).merge(metrics.round(10), on="model", suffixes=("_derived", "_saved"))
    checks = all(
        np.allclose(validation[f"{name}_derived"], validation[f"{name}_saved"])
        for name in ["MAE", "RMSE", "R2"]
    )
    if checks:
        st.success("Saved metrics match metrics recalculated from aligned test predictions.")
    else:
        st.warning("A mismatch exists between saved and recalculated metrics. Saved metrics are retained as the recorded result.")


elif page == "Historical Backtesting":
    selected_model = st.selectbox("Backtest model", model_columns,
                                  index=model_columns.index(best_model) if best_model in model_columns else 0)
    start, end = st.slider(
        "Test date range",
        min_value=predictions["date"].min().date(),
        max_value=predictions["date"].max().date(),
        value=(predictions["date"].min().date(), predictions["date"].max().date()),
        format="YYYY-MM-DD",
        key="backtest-range",
    )
    backtest = predictions[
        (predictions["date"].dt.date >= start) & (predictions["date"].dt.date <= end)
    ].copy()
    backtest["residual"] = backtest["actual"] - backtest[selected_model]
    backtest["absolute_error"] = backtest["residual"].abs()
    backtest["weekday"] = backtest["date"].dt.day_name()
    backtest["month"] = backtest["date"].dt.strftime("%B")

    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"{mean_absolute_error(backtest['actual'], backtest[selected_model]):.3f} GWh")
    c2.metric("RMSE", f"{np.sqrt(mean_squared_error(backtest['actual'], backtest[selected_model])):.3f} GWh")
    c3.metric("R²", f"{r2_score(backtest['actual'], backtest[selected_model]):.3f}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=backtest["date"], y=backtest["actual"], name="Actual",
                             line=dict(color=PRIMARY, width=2)))
    fig.add_trace(go.Scatter(x=backtest["date"], y=backtest[selected_model], name=selected_model,
                             line=dict(color=CYAN, width=2)))
    fig.update_layout(**chart_layout(f"Actual vs {selected_model}", 440))
    fig.update_yaxes(title="GWh/day")
    st.plotly_chart(fig, width="stretch")

    left, right = st.columns(2)
    with left:
        fig = px.line(backtest, x="date", y="residual", color_discrete_sequence=[AMBER])
        fig.add_hline(y=0, line_dash="dash", line_color="#64748B")
        fig.update_layout(**chart_layout("Residual time series · actual minus predicted", 350))
        fig.update_yaxes(title="Residual (GWh)")
        st.plotly_chart(fig, width="stretch")
    with right:
        fig = px.histogram(backtest, x="absolute_error", nbins=35, color_discrete_sequence=[TEAL])
        fig.update_layout(**chart_layout("Absolute error distribution", 350))
        fig.update_xaxes(title="Absolute error (GWh)")
        st.plotly_chart(fig, width="stretch")

    left, right = st.columns(2)
    with left:
        fig = px.scatter(backtest, x="actual", y=selected_model, opacity=.58,
                         color_discrete_sequence=[CYAN])
        fig.update_layout(**chart_layout("Actual vs predicted scatter", 350))
        fig.update_xaxes(title="Actual (GWh/day)")
        fig.update_yaxes(title="Predicted (GWh/day)")
        st.plotly_chart(fig, width="stretch")
    with right:
        weekday_error = backtest.groupby("weekday", as_index=False)["absolute_error"].mean()
        weekday_error["weekday"] = pd.Categorical(
            weekday_error["weekday"],
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
            ordered=True,
        )
        weekday_error = weekday_error.sort_values("weekday")
        fig = px.bar(weekday_error, x="weekday", y="absolute_error", color_discrete_sequence=[PRIMARY],
                     text_auto=".2f")
        fig.update_layout(**chart_layout("Mean absolute error by weekday", 350))
        fig.update_xaxes(title="")
        fig.update_yaxes(title="MAE (GWh)")
        st.plotly_chart(fig, width="stretch")

    month_error = backtest.groupby("month", as_index=False)["absolute_error"].mean()
    fig = px.bar(month_error, x="month", y="absolute_error", color_discrete_sequence=[AMBER],
                 text_auto=".2f")
    fig.update_layout(**chart_layout("Mean absolute error by month", 350))
    fig.update_xaxes(title="Month")
    fig.update_yaxes(title="MAE (GWh)")
    st.plotly_chart(fig, width="stretch")

    st.caption("This is a single chronological held-out backtest, not rolling cross-validation.")


elif page == "Next-Day Forecast Studio":
    st.info(
        "The studio uses the saved models to predict only the day immediately after the latest observed "
        "generation date. It does not provide arbitrary future dates or recursive weather-aware forecasts."
    )
    selected_model = st.selectbox("Saved model", model_columns)

    ext = pd.concat([history, pd.Series([np.nan], index=[history.index[-1] + pd.DateOffset(days=1)])])
    feature_frame = features(ext.fillna(-999))
    next_date = history.index[-1] + pd.DateOffset(days=1)
    next_row = feature_frame.loc[[next_date], FEATURES]

    forecasts = {}
    for name in model_columns:
        if name == "Hybrid RF + GB":
            rf = load_model("Random Forest")
            gb = load_model("Gradient Boosting")
            if rf is not None and gb is not None:
                forecasts[name] = 0.5 * (rf.predict(next_row)[0] + gb.predict(next_row)[0])
        elif name == "Hybrid XGB + GB":
            xgb = load_model("XGBoost")
            gb = load_model("Gradient Boosting")
            if xgb is not None and gb is not None:
                forecasts[name] = 0.5 * (xgb.predict(next_row)[0] + gb.predict(next_row)[0])
        else:
            model = load_model(name)
            if model is not None:
                forecasts[name] = model.predict(next_row)[0]

    if not forecasts:
        st.error("No saved model artifacts were available for next-day prediction.")
    else:
        chosen_forecast = forecasts[selected_model]
        c1, c2, c3 = st.columns(3)
        c1.metric("Forecast date", str(next_date.date()))
        c2.metric("Latest observation", f"{history.iloc[-1]:.2f} GWh")
        c3.metric("Selected forecast", f"{chosen_forecast:.2f} GWh")

        forecast_table = pd.DataFrame(
            {"Model": list(forecasts), "Forecast (GWh/day)": list(forecasts.values())}
        ).sort_values("Forecast (GWh/day)")
        fig = px.bar(forecast_table, x="Model", y="Forecast (GWh/day)",
                     color="Model", text_auto=".2f",
                     color_discrete_sequence=px.colors.qualitative.Plotly)
        fig.update_layout(**chart_layout("Saved-model next-day forecast comparison", 400), showlegend=False)
        fig.update_xaxes(title="")
        st.plotly_chart(fig, width="stretch")
        st.dataframe(forecast_table, width="stretch", hide_index=True)

    st.markdown("### Prediction inputs")
    st.write(
        "The next-day row uses the most recent observed generation to calculate lag-1/2/3/7/14/28, "
        "7-day and 14-day shifted rolling means, 7-day shifted standard deviation, and calendar values "
        "for the forecast date."
    )
    st.caption(
        "No uncertainty interval is shown because the saved artifacts do not provide a statistically "
        "validated probabilistic interval."
    )


elif page == "Explainable AI & Insights":
    st.markdown("### Permutation importance on held-out data")
    perm_model_name = st.selectbox(
        "Model",
        [m for m in model_columns if not m.startswith("Hybrid ")],
    )
    perm_model = load_model(perm_model_name)

    d = features(history)
    split = int(len(d) * 0.8)
    test_df = d.iloc[split:]
    X_test = test_df[FEATURES]
    y_test = test_df["energy_gwh"]

    if perm_model is not None:
        result = permutation_importance(
            perm_model, X_test, y_test, scoring="neg_mean_absolute_error",
            n_repeats=10, random_state=42, n_jobs=1,
        )
        importance = pd.DataFrame({
            "Feature": FEATURES,
            "Importance": result.importances_mean,
            "Std": result.importances_std,
        }).sort_values("Importance", ascending=False)
        fig = px.bar(importance, x="Importance", y="Feature", orientation="h",
                     error_x="Std", color="Importance", color_continuous_scale="Teal")
        fig.update_layout(**chart_layout("Permutation importance · MAE decrease", 430),
                          coloraxis_showscale=False)
        fig.update_xaxes(title="Increase in MAE when permuted")
        fig.update_yaxes(title="", categoryorder="total ascending")
        st.plotly_chart(fig, width="stretch")
    else:
        st.warning("The saved model needed for permutation importance is unavailable.")

    st.markdown("### Model interpretation")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(
            "**Strengths**\n"
            "- Random Forest can capture nonlinear relationships and interactions.\n"
            "- Bagging can reduce variance from individual trees.\n"
            "- Boosting can refine difficult residual patterns sequentially.\n"
            "- Lag and calendar features provide a consistent leakage-safe baseline."
        )
        st.markdown(
            "**Lag and rolling features**\n"
            "- Lag features use only previous generation observations.\n"
            "- Rolling means use shifted history, not the target day.\n"
            "- Calendar fields capture weekday and seasonal structure."
        )
    with c2:
        st.markdown(
            "**Limitations**\n"
            "- A single chronological split may be sensitive to period-specific conditions.\n"
            "- Ordinary stacking cross-validation is not fully time-series-aware.\n"
            "- Permutation importance shows predictive reliance, not causal effect.\n"
            "- Daily generation is a proxy for supply, not direct end-use demand."
        )
        st.markdown(
            "**Unavailable explanation tools**\n"
            "- SHAP is hidden unless a compatible package and feasible model attribution are available.\n"
            "- No causal weather, fuel, outage or demand variables are in the saved feature set."
        )


elif page == "Dataset & Methodology":
    st.markdown("### Dataset")
    c1, c2, c3 = st.columns(3)
    c1.metric("Observations", f"{len(dataset):,}")
    c2.metric("Date coverage", f"{dataset['date'].min().date()} – {dataset['date'].max().date()}")
    c3.metric("Target unit", "GWh/day")

    st.markdown(
        "The dashboard uses the real Sri Lankan daily generation dataset assembled from the "
        "verified CEB/PUCSL-derived source. `data/merged-generation-profile-2017-2026.csv` "
        "is aggregated by date for training and visualization."
    )

    st.markdown("### Saved training metadata")
    st.json(meta)

    st.markdown("### Feature engineering")
    st.dataframe(
        pd.DataFrame({
            "Feature": FEATURES,
            "Description": [
                "Generation 1 day before the target",
                "Generation 2 days before the target",
                "Generation 3 days before the target",
                "Generation 7 days before the target",
                "Generation 14 days before the target",
                "Generation 28 days before the target",
                "Shifted 7-day rolling mean",
                "Shifted 14-day rolling mean",
                "Shifted 7-day rolling standard deviation",
                "Day of week",
                "Month",
                "Day of year",
                "Weekend indicator",
            ],
        }),
        width="stretch",
        hide_index=True,
    )

    st.markdown("### Integrity notes")
    st.warning(
        "Daily generation is a proxy for energy supplied, not direct consumer demand. "
        "It is not suitable for operational dispatch decisions."
    )
    st.info(
        "The split is chronological. All lag and rolling features are shifted or lagged to use past "
        "observations only. Test records are not used to fit the component models."
    )
    st.error(
        "The saved stacking model uses ordinary 3-fold cross-validation internally. "
        "This is not a strict time-series-aware stacking implementation."
    )
