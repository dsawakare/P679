import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings("ignore")

try:
    import holidays as hol_lib
    HAS_HOLIDAYS = True
except ImportError:
    HAS_HOLIDAYS = False

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="PJM Energy Forecast", page_icon="⚡", layout="wide")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding-top: 1.6rem; padding-bottom: 2rem; }
    [data-testid="stSidebar"] {
        background: #0d0d1a;
        border-right: 1px solid #1e1e30;
    }
    .kpi-wrap {
        background: #12122a;
        border: 1px solid #1e1e38;
        border-radius: 14px;
        padding: 16px 14px 12px;
        text-align: center;
        height: 100%;
    }
    .kpi-val   { font-size: 1.75rem; font-weight: 700; color: #00d4aa; line-height: 1.1; }
    .kpi-label { font-size: 0.72rem; color: #666; margin-top: 5px; letter-spacing: .04em; text-transform: uppercase; }
    .kpi-sub   { font-size: 0.78rem; color: #999; margin-top: 2px; }
    .sec {
        font-size: 0.95rem; font-weight: 600;
        border-left: 3px solid #00d4aa;
        padding-left: 10px;
        margin: 28px 0 12px;
        color: #d0d0e0;
        letter-spacing: .02em;
    }
    .tag {
        display: inline-block;
        background: #1a2a1a;
        color: #00d4aa;
        border: 1px solid #00d4aa44;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.72rem;
        margin-right: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ── Palette ───────────────────────────────────────────────────────────────────
BG, CARD, GRID = "#0a0a18", "#12122a", "#1e1e38"
TEAL, CORAL, AMBER, BLUE, PURPLE = "#00d4aa", "#ff6b6b", "#ffd166", "#4ecdc4", "#9b72cf"

def sty(ax, fig=None, tight=True):
    if fig: fig.patch.set_facecolor(BG)
    ax.set_facecolor(CARD)
    ax.tick_params(colors="#777", labelsize=8)
    ax.xaxis.label.set_color("#888"); ax.yaxis.label.set_color("#888")
    ax.title.set_color("#d8d8e8")
    for sp in ax.spines.values(): sp.set_edgecolor(GRID)
    ax.grid(color=GRID, lw=0.4, alpha=0.6)
    if tight: plt.tight_layout()

def get_season(m):
    return "Winter" if m in [12,1,2] else "Spring" if m in [3,4,5] else "Summer" if m in [6,7,8] else "Autumn"

# ── Loaders ───────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    return joblib.load("xgb_model.pkl"), joblib.load("le_day.pkl"), joblib.load("le_season.pkl")

@st.cache_data
def load_data():
    df = pd.read_excel("PJMW_MW_Hourly.xlsx")
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.sort_values("Datetime").drop_duplicates("Datetime").reset_index(drop=True)
    if HAS_HOLIDAYS:
        uh = hol_lib.US()
        df["IsHoliday"] = df["Datetime"].dt.date.apply(lambda d: 1 if d in uh else 0)
    else:
        df["IsHoliday"] = 0
    df["Hour"]      = df["Datetime"].dt.hour
    df["Day"]       = df["Datetime"].dt.day
    df["Month"]     = df["Datetime"].dt.month
    df["Year"]      = df["Datetime"].dt.year
    df["Quarter"]   = df["Datetime"].dt.quarter
    df["DayOfWeek"] = df["Datetime"].dt.day_name()
    df["IsWeekend"] = (df["Datetime"].dt.dayofweek >= 5).astype(int)
    df["Season"]    = df["Month"].apply(get_season)
    df["Rolling_7_Day"] = df["PJMW_MW"].rolling(window=168).mean()
    return df.dropna()

@st.cache_data
def get_test(_df, _xgb, _ld, _ls):
    df = _df.copy()
    df["de"] = _ld.transform(df["DayOfWeek"])
    df["se"] = _ls.transform(df["Season"])
    ts   = int(len(df) * 0.8)
    test = df.iloc[ts:].copy()
    X    = pd.DataFrame({"Hour":test["Hour"],"Day":test["Day"],"Month":test["Month"],
                          "Year":test["Year"],"Quarter":test["Quarter"],
                          "DayOfWeek":test["de"],"IsWeekend":test["IsWeekend"],
                          "Season":test["se"],"IsHoliday":test["IsHoliday"],
                          "Rolling_7_Day":test["Rolling_7_Day"]})
    test = test.copy()
    test["Pred"]     = _xgb.predict(X)
    test["Residual"] = test["PJMW_MW"] - test["Pred"]
    return test

@st.cache_data
def build_forecast(_df, _xgb, _ld, _ls, n_days):
    n_hrs     = n_days * 24
    last_date = _df["Datetime"].max()
    idx       = pd.date_range(start=last_date + pd.Timedelta(hours=1), periods=n_hrs, freq="h")
    if HAS_HOLIDAYS:
        uh = hol_lib.US()
        ih = [1 if d.date() in uh else 0 for d in idx]
    else:
        ih = [0]*n_hrs
    roll = _df["Rolling_7_Day"].iloc[-1]
    f = pd.DataFrame({"Datetime":idx,"Hour":idx.hour,"Day":idx.day,"Month":idx.month,
                       "Year":idx.year,"Quarter":idx.quarter,"DayOfWeek":idx.day_name(),
                       "IsWeekend":(idx.dayofweek>=5).astype(int),
                       "Season":[get_season(m) for m in idx.month],
                       "IsHoliday":ih,"Rolling_7_Day":roll})
    f["de"] = _ld.transform(f["DayOfWeek"])
    f["se"] = _ls.transform(f["Season"])
    X = pd.DataFrame({"Hour":f["Hour"],"Day":f["Day"],"Month":f["Month"],"Year":f["Year"],
                       "Quarter":f["Quarter"],"DayOfWeek":f["de"],"IsWeekend":f["IsWeekend"],
                       "Season":f["se"],"IsHoliday":f["IsHoliday"],"Rolling_7_Day":f["Rolling_7_Day"]})
    f["Forecast_MW"] = _xgb.predict(X)
    f["Date"] = f["Datetime"].dt.date
    daily = (f.groupby("Date")["Forecast_MW"].agg(["mean","min","max"]).reset_index()
              .rename(columns={"mean":"Avg MW","min":"Min MW","max":"Max MW"}))
    return f, daily

# ── Bootstrap ─────────────────────────────────────────────────────────────────
try:
    xgb_model, le_day, le_season = load_models()
    df       = load_data()
    test_df  = get_test(df, xgb_model, le_day, le_season)
except Exception as e:
    st.error(f"Load error: {e}")
    st.stop()

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
yt  = test_df["PJMW_MW"].values
yp  = test_df["Pred"].values
MAE  = mean_absolute_error(yt, yp)
RMSE = np.sqrt(mean_squared_error(yt, yp))
R2   = r2_score(yt, yp)
MAPE = np.mean(np.abs((yt-yp)/yt))*100

# ════════════════════════════════════════════════════════════════════════════════
#  SIDEBAR — small, forecast slider only
# ════════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        "<div style='font-size:0.95rem;font-weight:700;color:#00d4aa;"
        "letter-spacing:.03em;margin-bottom:4px'>⚡ Forecast</div>",
        unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:0.7rem;color:#555;margin-bottom:14px'>"
        "PJM Energy · P679</div>",
        unsafe_allow_html=True)
    n_days = st.slider("Forecast days", min_value=1, max_value=30, value=7, step=1)
    st.markdown(
        f"<div style='text-align:center;font-size:2.4rem;font-weight:700;"
        f"color:#00d4aa;line-height:1'>{n_days}</div>"
        f"<div style='text-align:center;font-size:0.68rem;color:#555;"
        f"margin-top:2px'>{'day' if n_days==1 else 'days'} selected</div>",
        unsafe_allow_html=True)
    st.markdown("<hr style='border-color:#1e1e38;margin:16px 0'>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:0.65rem;color:#444;line-height:1.7'>"
        "Model: XGBoost<br>Dataset: PJM Hourly MW<br>"
        "Submitted by: Gajender Singh</div>",
        unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
#  HEADER
# ════════════════════════════════════════════════════════════════════════════════
st.markdown("# ⚡ PJM Hourly Energy Consumption Forecast")
st.markdown(
    '<span class="tag">XGBoost</span>'
    '<span class="tag">Time-Series</span>'
    '<span class="tag">P679</span>'
    '<span class="tag">Gajender Singh</span>',
    unsafe_allow_html=True
)
st.markdown("<br>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — KPI CARDS
# ════════════════════════════════════════════════════════════════════════════════
k1, k2, k3, k4 = st.columns(4)
for col, val, lbl, sub in [
    (k1, f"{MAE:,.0f} MW",  "MAE",    "Mean Absolute Error"),
    (k2, f"{RMSE:,.0f} MW", "RMSE",   "Root Mean Sq. Error"),
    (k3, f"{R2:.3f}",        "R²",     "Variance Explained"),
    (k4, f"{MAPE:.2f}%",    "MAPE",   "Mean Abs. % Error"),
]:
    col.markdown(f"""<div class="kpi-wrap">
        <div class="kpi-val">{val}</div>
        <div class="kpi-sub">{lbl}</div>
        <div class="kpi-label">{sub}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — FORECAST CHART (first chart, right below KPIs)
# ════════════════════════════════════════════════════════════════════════════════
st.markdown(f'<div class="sec">Energy Forecast — Next {n_days} Day{"s" if n_days>1 else ""}</div>',
            unsafe_allow_html=True)

fut_df, daily = build_forecast(df, xgb_model, le_day, le_season, n_days)

fig_fc, ax_fc = plt.subplots(figsize=(13, 4.2))
ax_fc.plot(fut_df["Datetime"], fut_df["Forecast_MW"],
           color=AMBER, lw=1.1, alpha=0.95, zorder=3)
ax_fc.fill_between(fut_df["Datetime"], fut_df["Forecast_MW"],
                   alpha=0.13, color=AMBER, zorder=2)
for _, row in daily.iterrows():
    dt = pd.Timestamp(row["Date"])
    if dt.dayofweek >= 5:
        ax_fc.axvspan(dt, dt + pd.Timedelta(hours=23),
                      alpha=0.06, color=PURPLE, zorder=1)
ax_fc.set_xlabel("Date", fontsize=9)
ax_fc.set_ylabel("Forecast (MW)", fontsize=9)
ax_fc.set_title(f"{n_days}-Day Hourly Energy Consumption Forecast  "
                f"(avg {fut_df['Forecast_MW'].mean():,.0f} MW · "
                f"peak {fut_df['Forecast_MW'].max():,.0f} MW)", fontsize=10)
sty(ax_fc, fig_fc)
st.pyplot(fig_fc); plt.close()

fa, fb = st.columns([1, 2])
with fa:
    st.markdown(f"**{n_days}-day summary**")
    d2 = daily.copy()
    d2.index = range(1, len(d2)+1); d2.index.name = "Day"
    d2["Avg MW"] = d2["Avg MW"].round(1)
    d2["Min MW"] = d2["Min MW"].round(1)
    d2["Max MW"] = d2["Max MW"].round(1)
    st.dataframe(d2, use_container_width=True, height=min(35*n_days+40, 420))
    st.download_button(f"Download {n_days}-day CSV", d2.to_csv(),
                       f"pjm_{n_days}day_forecast.csv", "text/csv")
with fb:
    fig_bar, ax_bar = plt.subplots(figsize=(8, max(2.5, n_days*0.28)))
    fig_bar.patch.set_facecolor(BG)
    bc = [TEAL if pd.Timestamp(d).dayofweek < 5 else PURPLE for d in daily["Date"]]
    ax_bar.barh(range(len(daily)), daily["Avg MW"], color=bc, edgecolor="none", height=0.6)
    ax_bar.set_yticks(range(len(daily)))
    ax_bar.set_yticklabels([str(d) for d in daily["Date"]], fontsize=7)
    ax_bar.set_xlabel("Avg MW", fontsize=9)
    ax_bar.set_title("Daily average  (purple = weekend)", fontsize=9)
    ax_bar.invert_yaxis()
    sty(ax_bar, fig_bar)
    st.pyplot(fig_bar); plt.close()

# ════════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — ACTUAL VS PREDICTED
# ════════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec">Actual vs Predicted</div>', unsafe_allow_html=True)

n_h = st.slider("Hours to display", 100, 2000, 500, 100, key="avp")
s   = test_df.iloc[:n_h]

fig_avp, ax_avp = plt.subplots(figsize=(13, 4))
ax_avp.plot(s["Datetime"], s["PJMW_MW"],
            color=TEAL, lw=0.9, label="Actual", alpha=0.9)
ax_avp.plot(s["Datetime"], s["Pred"],
            color=CORAL, lw=0.9, label="Predicted", alpha=0.85, ls="--")
ax_avp.set_xlabel("Date", fontsize=9); ax_avp.set_ylabel("Energy (MW)", fontsize=9)
ax_avp.set_title(f"Actual vs Predicted — first {n_h} hours of test set", fontsize=10)
ax_avp.legend(facecolor=CARD, edgecolor=GRID, labelcolor="#ccc", fontsize=9)
sty(ax_avp, fig_avp)
st.pyplot(fig_avp); plt.close()

# ════════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — RESIDUAL ANALYSIS
# ════════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec">Residual Analysis</div>', unsafe_allow_html=True)

fig_res, ax_res = plt.subplots(figsize=(13, 3.8))
ax_res.hist(test_df["Residual"], bins=90, color=PURPLE, edgecolor="none", alpha=0.85)
ax_res.axvline(0, color=CORAL, lw=1.8, ls="--", label="Zero error line")
ax_res.axvline(test_df["Residual"].mean(), color=AMBER, lw=1.4, ls=":",
               label=f"Mean error  {test_df['Residual'].mean():.1f} MW")
ax_res.set_xlabel("Prediction Error (MW)", fontsize=9)
ax_res.set_ylabel("Frequency", fontsize=9)
ax_res.set_title("Prediction Error Distribution — errors cluster near zero = model is reliable", fontsize=10)
ax_res.legend(facecolor=CARD, edgecolor=GRID, labelcolor="#ccc", fontsize=9)
sty(ax_res, fig_res)
st.pyplot(fig_res); plt.close()

re1, re2, re3 = st.columns(3)
re1.metric("Mean Error",           f"{test_df['Residual'].mean():.1f} MW")
re2.metric("Std Deviation",        f"{test_df['Residual'].std():.1f} MW")
re3.metric("Errors within ±500 MW",f"{(np.abs(test_df['Residual'])<=500).mean()*100:.1f}%")

# ════════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — MODEL EVALUATION
# ════════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec">Model Evaluation</div>', unsafe_allow_html=True)

me1, me2 = st.columns(2)
with me1:
    fig_mc, ax_mc = plt.subplots(figsize=(6.5, 3.5))
    fig_mc.patch.set_facecolor(BG)
    mnames = ["Holt-Winters","ARIMA","Random Forest","XGBoost"]
    rvals  = [1136.68, 1050.0, 620.0, round(RMSE,1)]
    mcols  = [GRID, GRID, BLUE, TEAL]
    b = ax_mc.barh(mnames, rvals, color=mcols, edgecolor="none", height=0.5)
    ax_mc.bar_label(b, fmt="%.0f", padding=4, color="#ccc", fontsize=9)
    ax_mc.set_xlabel("RMSE (MW)", fontsize=9)
    ax_mc.set_title("Model Comparison — RMSE (lower is better)", fontsize=10)
    ax_mc.invert_yaxis()
    sty(ax_mc, fig_mc)
    st.pyplot(fig_mc); plt.close()

with me2:
    fig_sc, ax_sc = plt.subplots(figsize=(6.5, 3.5))
    fig_sc.patch.set_facecolor(BG)
    s2 = test_df.sample(min(2500, len(test_df)), random_state=42)
    ax_sc.scatter(s2["PJMW_MW"], s2["Pred"], alpha=0.18, s=4, color=TEAL)
    mn, mx = min(yt.min(),yp.min()), max(yt.max(),yp.max())
    ax_sc.plot([mn,mx],[mn,mx], color=CORAL, lw=1.5, ls="--", label="Perfect fit")
    ax_sc.set_xlabel("Actual (MW)", fontsize=9)
    ax_sc.set_ylabel("Predicted (MW)", fontsize=9)
    ax_sc.set_title("Actual vs Predicted Scatter", fontsize=10)
    ax_sc.legend(facecolor=CARD, edgecolor=GRID, labelcolor="#ccc", fontsize=9)
    sty(ax_sc, fig_sc)
    st.pyplot(fig_sc); plt.close()

# ════════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — FEATURE IMPORTANCE
# ════════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec">Feature Importance</div>', unsafe_allow_html=True)

fnames = ["Hour","Day","Month","Year","Quarter",
          "DayOfWeek","IsWeekend","Season","IsHoliday","Rolling_7_Day"]
fvals  = xgb_model.feature_importances_
fi     = pd.DataFrame({"Feature":fnames,"Importance":fvals}).sort_values("Importance")

fig_fi, ax_fi = plt.subplots(figsize=(13, 3.8))
fc = [TEAL if v >= np.median(fvals) else BLUE for v in fi["Importance"]]
bfi = ax_fi.barh(fi["Feature"], fi["Importance"], color=fc, edgecolor="none", height=0.55)
ax_fi.bar_label(bfi, fmt="%.3f", padding=4, color="#ccc", fontsize=9)
ax_fi.set_xlabel("Importance Score", fontsize=9)
ax_fi.set_title("XGBoost Feature Importance — teal bars are above-median drivers", fontsize=10)
sty(ax_fi, fig_fi)
st.pyplot(fig_fi); plt.close()

st.markdown("---")
st.caption("P679 · PJM Hourly Energy Consumption Forecast · Gajender Singh · XGBoost Regressor")
