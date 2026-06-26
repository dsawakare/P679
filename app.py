import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

try:
    import holidays as hol_lib
    HAS_HOLIDAYS = True
except ImportError:
    HAS_HOLIDAYS = False

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PJM Energy Forecast",
    page_icon="⚡",
    layout="wide"
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    #MainMenu, footer, header { visibility: hidden; }
    [data-testid="collapsedControl"] { display: none; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1100px; }
    .kpi-card {
        background: #1e1e2e;
        border: 1px solid #2d2d44;
        border-radius: 12px;
        padding: 18px 20px;
        text-align: center;
    }
    .kpi-val   { font-size: 1.9rem; font-weight: 700; color: #00d4aa; }
    .kpi-label { font-size: 0.82rem; color: #888; margin-top: 4px; }
    .sec-header {
        font-size: 1.05rem; font-weight: 600;
        border-left: 3px solid #00d4aa;
        padding-left: 10px;
        margin: 32px 0 14px 0;
        color: #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# ── Colour palette ───────────────────────────────────────────────────────────
BG     = "#0f0f1a"
CARD   = "#1e1e2e"
GRID   = "#2a2a3e"
TEAL   = "#00d4aa"
CORAL  = "#ff6b6b"
AMBER  = "#ffd166"
BLUE   = "#4ecdc4"
PURPLE = "#9b72cf"

def sty(ax, fig=None):
    if fig: fig.patch.set_facecolor(BG)
    ax.set_facecolor(CARD)
    ax.tick_params(colors="#aaa", labelsize=8)
    ax.xaxis.label.set_color("#aaa")
    ax.yaxis.label.set_color("#aaa")
    ax.title.set_color("#e0e0e0")
    for sp in ax.spines.values(): sp.set_edgecolor(GRID)
    ax.grid(color=GRID, linewidth=0.4, alpha=0.5)

def get_season(m):
    if m in [12,1,2]:  return "Winter"
    if m in [3,4,5]:   return "Spring"
    if m in [6,7,8]:   return "Summer"
    return "Autumn"

# ── Load resources ───────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    return (joblib.load("xgb_model.pkl"),
            joblib.load("le_day.pkl"),
            joblib.load("le_season.pkl"))

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
def get_test_preds(_df, _xgb, _le_d, _le_s):
    df = _df.copy()
    df["dow_enc"] = _le_d.transform(df["DayOfWeek"])
    df["sea_enc"] = _le_s.transform(df["Season"])
    ts = int(len(df) * 0.8)
    test = df.iloc[ts:].copy()
    X = pd.DataFrame({
        "Hour": test["Hour"], "Day": test["Day"], "Month": test["Month"],
        "Year": test["Year"], "Quarter": test["Quarter"],
        "DayOfWeek": test["dow_enc"], "IsWeekend": test["IsWeekend"],
        "Season": test["sea_enc"], "IsHoliday": test["IsHoliday"],
        "Rolling_7_Day": test["Rolling_7_Day"]
    })
    test = test.copy()
    test["Predicted"] = _xgb.predict(X)
    test["Residual"]  = test["PJMW_MW"] - test["Predicted"]
    return test

# ── Init ─────────────────────────────────────────────────────────────────────
try:
    xgb_model, le_day, le_season = load_models()
    df = load_data()
    test_df = get_test_preds(df, xgb_model, le_day, le_season)
except Exception as e:
    st.error(f"Error: {e}")
    st.info("Make sure xgb_model.pkl, le_day.pkl, le_season.pkl and PJMW_MW_Hourly.xlsx are in the same folder.")
    st.stop()

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
yt, yp = test_df["PJMW_MW"].values, test_df["Predicted"].values
MAE  = mean_absolute_error(yt, yp)
RMSE = np.sqrt(mean_squared_error(yt, yp))
R2   = r2_score(yt, yp)
MAPE = np.mean(np.abs((yt - yp) / yt)) * 100

# ════════════════════════════════════════════════════════════════════════════
#  HEADER
# ════════════════════════════════════════════════════════════════════════════
st.markdown("# ⚡ PJM Hourly Energy Consumption Forecast")
st.markdown("**Project P679 · Submitted by Gajender Singh · Model: XGBoost Regressor**")
st.markdown("---")

# ════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — KPI CARDS
# ════════════════════════════════════════════════════════════════════════════
c1, c2, c3, c4 = st.columns(4)
for col, val, lbl in [
    (c1, f"{MAE:,.0f} MW", "MAE — Mean Absolute Error"),
    (c2, f"{RMSE:,.0f} MW", "RMSE — Root Mean Sq. Error"),
    (c3, f"{R2:.3f}", "R² Score"),
    (c4, f"{MAPE:.2f}%", "MAPE"),
]:
    col.markdown(f"""<div class="kpi-card">
        <div class="kpi-val">{val}</div>
        <div class="kpi-label">{lbl}</div>
    </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — ACTUAL VS PREDICTED
# ════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-header">Actual vs Predicted</div>', unsafe_allow_html=True)

n_hours = st.slider("Select number of hours to view", 100, 2000, 500, 100,
                    key="avp_slider")
samp = test_df.iloc[:n_hours]

fig, ax = plt.subplots(figsize=(13, 4))
ax.plot(samp["Datetime"], samp["PJMW_MW"],
        color=TEAL, lw=0.9, label="Actual", alpha=0.9)
ax.plot(samp["Datetime"], samp["Predicted"],
        color=CORAL, lw=0.9, label="Predicted", alpha=0.85, ls="--")
ax.set_xlabel("Date"); ax.set_ylabel("Energy (MW)")
ax.set_title(f"Actual vs Predicted — first {n_hours} hours of test set", fontsize=10)
ax.legend(facecolor=CARD, edgecolor=GRID, labelcolor="#ccc", fontsize=9)
sty(ax, fig); plt.tight_layout()
st.pyplot(fig); plt.close()

# ════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — 30-DAY FORECAST (slider 1–30)
# ════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-header">Energy Forecast</div>', unsafe_allow_html=True)

n_days = st.slider("Select forecast days", 1, 30, 7, 1, key="fc_slider")
n_hrs  = n_days * 24

last_date = df["Datetime"].max()
fut_idx   = pd.date_range(start=last_date + pd.Timedelta(hours=1),
                           periods=n_hrs, freq="h")

if HAS_HOLIDAYS:
    uh2 = hol_lib.US()
    is_hol = [1 if d.date() in uh2 else 0 for d in fut_idx]
else:
    is_hol = [0] * n_hrs

last_roll = df["Rolling_7_Day"].iloc[-1]

fut = pd.DataFrame({
    "Datetime":     fut_idx,
    "Hour":         fut_idx.hour,
    "Day":          fut_idx.day,
    "Month":        fut_idx.month,
    "Year":         fut_idx.year,
    "Quarter":      fut_idx.quarter,
    "DayOfWeek":    fut_idx.day_name(),
    "IsWeekend":    (fut_idx.dayofweek >= 5).astype(int),
    "Season":       [get_season(m) for m in fut_idx.month],
    "IsHoliday":    is_hol,
    "Rolling_7_Day": last_roll,
})
fut["dow_enc"] = le_day.transform(fut["DayOfWeek"])
fut["sea_enc"] = le_season.transform(fut["Season"])

X_fut = pd.DataFrame({
    "Hour": fut["Hour"], "Day": fut["Day"], "Month": fut["Month"],
    "Year": fut["Year"], "Quarter": fut["Quarter"],
    "DayOfWeek": fut["dow_enc"], "IsWeekend": fut["IsWeekend"],
    "Season": fut["sea_enc"], "IsHoliday": fut["IsHoliday"],
    "Rolling_7_Day": fut["Rolling_7_Day"]
})
fut["Forecast_MW"] = xgb_model.predict(X_fut)
fut["Date"] = fut["Datetime"].dt.date

daily = (fut.groupby("Date")["Forecast_MW"]
           .agg(["mean","min","max"])
           .reset_index()
           .rename(columns={"mean":"Avg MW","min":"Min MW","max":"Max MW"}))

# Forecast line chart
fig2, ax2 = plt.subplots(figsize=(13, 4))
ax2.plot(fut["Datetime"], fut["Forecast_MW"],
         color=AMBER, lw=0.9, alpha=0.9)
ax2.fill_between(fut["Datetime"], fut["Forecast_MW"],
                 alpha=0.12, color=AMBER)
ax2.set_xlabel("Date"); ax2.set_ylabel("Forecast (MW)")
ax2.set_title(f"{n_days}-Day Hourly Energy Consumption Forecast", fontsize=10)
sty(ax2, fig2); plt.tight_layout()
st.pyplot(fig2); plt.close()

# Daily summary table
st.markdown(f"**Daily summary — {n_days} day(s)**")
daily_show = daily.copy()
daily_show.index = range(1, len(daily_show)+1)
daily_show.index.name = "Day"
daily_show["Avg MW"] = daily_show["Avg MW"].round(1)
daily_show["Min MW"] = daily_show["Min MW"].round(1)
daily_show["Max MW"] = daily_show["Max MW"].round(1)
st.dataframe(daily_show, use_container_width=True)

csv_data = daily_show.to_csv()
st.download_button(f"Download {n_days}-day forecast CSV",
                   csv_data, f"pjm_{n_days}day_forecast.csv", "text/csv")

# ════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — RESIDUAL ANALYSIS (best single chart)
# ════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-header">Residual Analysis</div>', unsafe_allow_html=True)

fig3, ax3 = plt.subplots(figsize=(13, 4))
ax3.hist(test_df["Residual"], bins=80, color=PURPLE, edgecolor="none", alpha=0.85)
ax3.axvline(0, color=CORAL, lw=1.8, ls="--", label="Zero error")
ax3.axvline(test_df["Residual"].mean(), color=AMBER, lw=1.4, ls=":",
            label=f"Mean error ({test_df['Residual'].mean():.1f} MW)")
ax3.set_xlabel("Prediction Error (MW)")
ax3.set_ylabel("Frequency")
ax3.set_title("Prediction Error Distribution — how far off are the forecasts?", fontsize=10)
ax3.legend(facecolor=CARD, edgecolor=GRID, labelcolor="#ccc", fontsize=9)
sty(ax3, fig3); plt.tight_layout()
st.pyplot(fig3); plt.close()

r1, r2, r3 = st.columns(3)
r1.metric("Mean Error",      f"{test_df['Residual'].mean():.1f} MW")
r2.metric("Std Deviation",   f"{test_df['Residual'].std():.1f} MW")
r3.metric("Error within ±500 MW",
          f"{(np.abs(test_df['Residual']) <= 500).mean()*100:.1f}%")

# ════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — MODEL EVALUATION
# ════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-header">Model Evaluation</div>', unsafe_allow_html=True)

col_a, col_b = st.columns([1, 1])

with col_a:
    # Model comparison bar
    fig4, ax4 = plt.subplots(figsize=(6, 3.5))
    fig4.patch.set_facecolor(BG)
    models_list = ["Holt-Winters", "ARIMA", "Random Forest", "XGBoost"]
    rmse_vals   = [1136.68, 1050.0, 620.0, round(RMSE, 1)]
    bar_cols    = [GRID, GRID, BLUE, TEAL]
    bars = ax4.barh(models_list, rmse_vals, color=bar_cols, edgecolor="none", height=0.5)
    ax4.bar_label(bars, fmt="%.0f", padding=4, color="#ccc", fontsize=9)
    ax4.set_xlabel("RMSE (MW)")
    ax4.set_title("Model Comparison — RMSE", fontsize=10)
    ax4.invert_yaxis()
    sty(ax4, fig4); plt.tight_layout()
    st.pyplot(fig4); plt.close()

with col_b:
    # Actual vs predicted scatter
    fig5, ax5 = plt.subplots(figsize=(6, 3.5))
    fig5.patch.set_facecolor(BG)
    samp2 = test_df.sample(min(2000, len(test_df)), random_state=42)
    ax5.scatter(samp2["PJMW_MW"], samp2["Predicted"],
                alpha=0.2, s=4, color=TEAL)
    mn, mx = min(yt.min(), yp.min()), max(yt.max(), yp.max())
    ax5.plot([mn, mx], [mn, mx], color=CORAL, lw=1.5, ls="--")
    ax5.set_xlabel("Actual (MW)")
    ax5.set_ylabel("Predicted (MW)")
    ax5.set_title("Actual vs Predicted Scatter", fontsize=10)
    sty(ax5, fig5); plt.tight_layout()
    st.pyplot(fig5); plt.close()

# ════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — FEATURE IMPORTANCE
# ════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="sec-header">Feature Importance</div>', unsafe_allow_html=True)

feat_names = ["Hour","Day","Month","Year","Quarter",
              "DayOfWeek","IsWeekend","Season","IsHoliday","Rolling_7_Day"]
fi_vals = xgb_model.feature_importances_
fi_df   = pd.DataFrame({"Feature": feat_names, "Importance": fi_vals})
fi_df   = fi_df.sort_values("Importance", ascending=True)

fig6, ax6 = plt.subplots(figsize=(13, 4))
bar_c = [TEAL if v >= np.median(fi_vals) else BLUE for v in fi_df["Importance"]]
bars6 = ax6.barh(fi_df["Feature"], fi_df["Importance"],
                 color=bar_c, edgecolor="none", height=0.55)
ax6.bar_label(bars6, fmt="%.3f", padding=4, color="#ccc", fontsize=9)
ax6.set_xlabel("Importance Score")
ax6.set_title("XGBoost Feature Importance — which features drive the forecast most?", fontsize=10)
sty(ax6, fig6); plt.tight_layout()
st.pyplot(fig6); plt.close()

st.markdown("---")
st.caption("P679 · PJM Hourly Energy Consumption Forecast · Gajender Singh")
