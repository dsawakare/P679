import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

try:
    import holidays as hol_lib
    HAS_HOLIDAYS = True
except ImportError:
    HAS_HOLIDAYS = False

# ── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PJM Energy Forecast",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1e1e2e 0%, #2d2d44 100%);
        border: 1px solid #3d3d5c;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 5px;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #00d4aa; }
    .metric-label { font-size: 0.85rem; color: #aaa; margin-top: 4px; }
    .metric-sub   { font-size: 0.75rem; color: #666; margin-top: 2px; }
    .section-header {
        font-size: 1.2rem; font-weight: 600; color: #e0e0e0;
        padding: 8px 0; border-bottom: 2px solid #00d4aa;
        margin: 20px 0 15px 0;
    }
    .stDataFrame { border-radius: 10px; }
    [data-testid="stSidebar"] { background-color: #0f0f1a; }
</style>
""", unsafe_allow_html=True)

# ── Helpers ──────────────────────────────────────────────────────────────────
PLT_BG   = "#0f0f1a"
PLT_CARD = "#1e1e2e"
PLT_GRID = "#2a2a3e"
C_TEAL   = "#00d4aa"
C_CORAL  = "#ff6b6b"
C_AMBER  = "#ffd166"
C_PURPLE = "#9b72cf"
C_BLUE   = "#4ecdc4"

def set_style(ax, fig=None):
    if fig:
        fig.patch.set_facecolor(PLT_BG)
    ax.set_facecolor(PLT_CARD)
    ax.tick_params(colors="#aaa", labelsize=9)
    ax.xaxis.label.set_color("#aaa")
    ax.yaxis.label.set_color("#aaa")
    ax.title.set_color("#e0e0e0")
    for spine in ax.spines.values():
        spine.set_edgecolor(PLT_GRID)
    ax.grid(color=PLT_GRID, linewidth=0.5, alpha=0.6)

def get_season(month):
    if month in [12, 1, 2]: return "Winter"
    elif month in [3, 4, 5]: return "Spring"
    elif month in [6, 7, 8]: return "Summer"
    else: return "Autumn"

# ── Load Resources ───────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    xgb  = joblib.load("xgb_model.pkl")
    le_d = joblib.load("le_day.pkl")
    le_s = joblib.load("le_season.pkl")
    return xgb, le_d, le_s

@st.cache_data
def load_data():
    df = pd.read_excel("PJMW_MW_Hourly.xlsx")
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.sort_values("Datetime").drop_duplicates("Datetime").reset_index(drop=True)

    if HAS_HOLIDAYS:
        us_hol = hol_lib.US()
        df["IsHoliday"] = df["Datetime"].dt.date.apply(lambda d: 1 if d in us_hol else 0)
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
    df["Rolling_7_Day"] = df["PJMW_MW"].rolling(window=24 * 7).mean()
    df = df.dropna()
    return df

@st.cache_data
def prepare_test(_df, _xgb, _le_d, _le_s):
    df = _df.copy()
    df["DayOfWeek_enc"] = _le_d.transform(df["DayOfWeek"])
    df["Season_enc"]    = _le_s.transform(df["Season"])

    train_size = int(len(df) * 0.8)
    test = df.iloc[train_size:].copy()

    X_test = pd.DataFrame({
        "Hour": test["Hour"], "Day": test["Day"],
        "Month": test["Month"], "Year": test["Year"],
        "Quarter": test["Quarter"], "DayOfWeek": test["DayOfWeek_enc"],
        "IsWeekend": test["IsWeekend"], "Season": test["Season_enc"],
        "IsHoliday": test["IsHoliday"], "Rolling_7_Day": test["Rolling_7_Day"],
    })

    y_pred = _xgb.predict(X_test)
    test   = test.copy()
    test["Predicted"] = y_pred
    test["Residual"]  = test["PJMW_MW"] - y_pred
    return test

# ── Load ─────────────────────────────────────────────────────────────────────
try:
    xgb_model, le_day, le_season = load_models()
    df = load_data()
    test_df = prepare_test(df, xgb_model, le_day, le_season)
except Exception as e:
    st.error(f"Error loading files: {e}")
    st.info("Make sure xgb_model.pkl, le_day.pkl, le_season.pkl and PJMW_MW_Hourly.xlsx are in the same folder.")
    st.stop()

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
y_true = test_df["PJMW_MW"].values
y_pred_all = test_df["Predicted"].values
MAE  = mean_absolute_error(y_true, y_pred_all)
RMSE = np.sqrt(mean_squared_error(y_true, y_pred_all))
R2   = r2_score(y_true, y_pred_all)
MAPE = np.mean(np.abs((y_true - y_pred_all) / y_true)) * 100

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚡ PJM Energy Forecast")
    st.markdown("---")
    st.markdown("### Project Info")
    st.markdown(f"**Model:** XGBoost Regressor")
    st.markdown(f"**Dataset:** PJM Hourly Energy Consumption")
    st.markdown("**Target Variable:** PJMW_MW")
    st.markdown("**Features:** Hour, Season, IsWeekend, Rolling_7_Day, etc.")
    st.markdown("---")
    st.markdown("### Navigation")
    page = st.radio("Go to", [
        "Overview & Metrics",
        "Actual vs Predicted",
        "30-Day Forecast",
        "Residual Analysis",
        "Feature Importance",
        "EDA Insights"
    ])
    st.markdown("---")
    st.markdown("**Submitted by:** Gajender Singh")
    st.markdown("**Project:** P679")

# ═══════════════════════════════════════════════════════════════════════════
#  PAGE 1 — OVERVIEW & METRICS
# ═══════════════════════════════════════════════════════════════════════════
if page == "Overview & Metrics":
    st.markdown("# ⚡ PJM Hourly Energy Consumption Forecast")
    st.markdown("**XGBoost model trained on historical hourly MW data to forecast next 30 days.**")
    st.markdown("---")

    # KPI row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{MAE:,.0f}</div>
            <div class="metric-label">MAE (MW)</div>
            <div class="metric-sub">Mean Absolute Error</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{RMSE:,.0f}</div>
            <div class="metric-label">RMSE (MW)</div>
            <div class="metric-sub">Root Mean Sq. Error</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{R2:.3f}</div>
            <div class="metric-label">R² Score</div>
            <div class="metric-sub">Variance Explained</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div class="metric-card">
            <div class="metric-value">{MAPE:.1f}%</div>
            <div class="metric-label">MAPE</div>
            <div class="metric-sub">Mean Abs. % Error</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Dataset stats
    st.markdown('<div class="section-header">Dataset Summary</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        stats = {
            "Total Records": f"{len(df):,}",
            "Date Range": f"{df['Datetime'].min().date()} → {df['Datetime'].max().date()}",
            "Train Records": f"{int(len(df)*0.8):,}",
            "Test Records":  f"{len(df) - int(len(df)*0.8):,}",
            "Avg Consumption": f"{df['PJMW_MW'].mean():,.1f} MW",
            "Peak Consumption": f"{df['PJMW_MW'].max():,.1f} MW",
        }
        st.dataframe(pd.DataFrame(stats.items(), columns=["Metric", "Value"]),
                     use_container_width=True, hide_index=True)
    with col2:
        # Model comparison bar
        fig, ax = plt.subplots(figsize=(6, 3.5))
        fig.patch.set_facecolor(PLT_BG)
        ax.set_facecolor(PLT_CARD)
        models = ["Holt-Winters", "ARIMA", "Random Forest", "XGBoost"]
        rmses  = [1136.68, 1050.0, 620.0, RMSE]
        colors = [PLT_GRID, PLT_GRID, C_BLUE, C_TEAL]
        bars   = ax.barh(models, rmses, color=colors, edgecolor="none", height=0.55)
        ax.bar_label(bars, fmt="%.0f", padding=4, color="#ccc", fontsize=9)
        ax.set_xlabel("RMSE (MW)", color="#aaa", fontsize=9)
        ax.set_title("Model Comparison (RMSE)", color="#e0e0e0", fontsize=10, pad=8)
        set_style(ax, fig)
        ax.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

# ═══════════════════════════════════════════════════════════════════════════
#  PAGE 2 — ACTUAL VS PREDICTED
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Actual vs Predicted":
    st.markdown("## Actual vs Predicted")
    n_hours = st.slider("Hours to display", 200, 2000, 500, 100)
    sample  = test_df.iloc[:n_hours]

    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.plot(sample["Datetime"], sample["PJMW_MW"],
            color=C_TEAL, linewidth=0.9, label="Actual", alpha=0.9)
    ax.plot(sample["Datetime"], sample["Predicted"],
            color=C_CORAL, linewidth=0.9, label="Predicted", alpha=0.85, linestyle="--")
    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("Energy (MW)", fontsize=10)
    ax.set_title(f"Actual vs Predicted — First {n_hours} Hours of Test Set", fontsize=11)
    ax.legend(facecolor=PLT_CARD, edgecolor=PLT_GRID, labelcolor="#ccc", fontsize=9)
    set_style(ax, fig)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown('<div class="section-header">Prediction Error Distribution</div>',
                unsafe_allow_html=True)
    fig2, axes = plt.subplots(1, 2, figsize=(14, 4))
    fig2.patch.set_facecolor(PLT_BG)

    # Histogram
    axes[0].hist(test_df["Residual"], bins=60, color=C_PURPLE,
                 edgecolor="none", alpha=0.85)
    axes[0].axvline(0, color=C_CORAL, linewidth=1.5, linestyle="--")
    axes[0].set_xlabel("Prediction Error (MW)", fontsize=9)
    axes[0].set_ylabel("Frequency", fontsize=9)
    axes[0].set_title("Prediction Error Distribution", fontsize=10)
    set_style(axes[0], fig2)

    # Scatter actual vs predicted
    sample2 = test_df.sample(min(3000, len(test_df)), random_state=42)
    axes[1].scatter(sample2["PJMW_MW"], sample2["Predicted"],
                    alpha=0.25, s=4, color=C_TEAL)
    mn = min(y_true.min(), y_pred_all.min())
    mx = max(y_true.max(), y_pred_all.max())
    axes[1].plot([mn, mx], [mn, mx], color=C_CORAL, linewidth=1.5, linestyle="--")
    axes[1].set_xlabel("Actual (MW)", fontsize=9)
    axes[1].set_ylabel("Predicted (MW)", fontsize=9)
    axes[1].set_title("Actual vs Predicted Scatter", fontsize=10)
    set_style(axes[1], fig2)

    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

# ═══════════════════════════════════════════════════════════════════════════
#  PAGE 3 — 30-DAY FORECAST
# ═══════════════════════════════════════════════════════════════════════════
elif page == "30-Day Forecast":
    st.markdown("## 30-Day Energy Forecast")
    st.markdown("Forecast generated from the **last known date** in the dataset using XGBoost.")

    last_date   = df["Datetime"].max()
    future_idx  = pd.date_range(start=last_date + pd.Timedelta(hours=1),
                                periods=720, freq="h")

    if HAS_HOLIDAYS:
        us_hol = hol_lib.US()
        is_holiday = [1 if d.date() in us_hol else 0 for d in future_idx]
    else:
        is_holiday = [0] * 720

    last_rolling = df["Rolling_7_Day"].iloc[-1]

    future_df = pd.DataFrame({
        "Datetime":     future_idx,
        "Hour":         future_idx.hour,
        "Day":          future_idx.day,
        "Month":        future_idx.month,
        "Year":         future_idx.year,
        "Quarter":      future_idx.quarter,
        "DayOfWeek":    future_idx.day_name(),
        "IsWeekend":    (future_idx.dayofweek >= 5).astype(int),
        "Season":       [get_season(m) for m in future_idx.month],
        "IsHoliday":    is_holiday,
        "Rolling_7_Day": last_rolling,
    })

    future_df["DayOfWeek_enc"] = le_day.transform(future_df["DayOfWeek"])
    future_df["Season_enc"]    = le_season.transform(future_df["Season"])

    X_future = pd.DataFrame({
        "Hour": future_df["Hour"], "Day": future_df["Day"],
        "Month": future_df["Month"], "Year": future_df["Year"],
        "Quarter": future_df["Quarter"], "DayOfWeek": future_df["DayOfWeek_enc"],
        "IsWeekend": future_df["IsWeekend"], "Season": future_df["Season_enc"],
        "IsHoliday": future_df["IsHoliday"], "Rolling_7_Day": future_df["Rolling_7_Day"],
    })

    forecast = xgb_model.predict(X_future)
    future_df["Forecast_MW"] = forecast

    # Daily summary
    future_df["Date"] = future_df["Datetime"].dt.date
    daily = future_df.groupby("Date")["Forecast_MW"].agg(["mean","min","max"]).reset_index()
    daily.columns = ["Date","Avg MW","Min MW","Max MW"]

    # Line chart
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.plot(future_df["Datetime"], future_df["Forecast_MW"],
            color=C_AMBER, linewidth=0.9, alpha=0.9)
    ax.fill_between(future_df["Datetime"], future_df["Forecast_MW"],
                    alpha=0.15, color=C_AMBER)
    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("Forecasted Energy (MW)", fontsize=10)
    ax.set_title("30-Day Hourly Energy Consumption Forecast", fontsize=11)
    set_style(ax, fig)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Daily bar chart
    st.markdown('<div class="section-header">Daily Average Forecast</div>',
                unsafe_allow_html=True)
    fig2, ax2 = plt.subplots(figsize=(14, 4))
    fig2.patch.set_facecolor(PLT_BG)
    bar_colors = [C_TEAL if i % 2 == 0 else C_BLUE for i in range(len(daily))]
    ax2.bar(range(len(daily)), daily["Avg MW"], color=bar_colors,
            edgecolor="none", width=0.7)
    ax2.set_xticks(range(0, len(daily), 3))
    ax2.set_xticklabels([str(d) for d in daily["Date"].iloc[::3]],
                        rotation=30, fontsize=8)
    ax2.set_ylabel("Avg MW", fontsize=9)
    ax2.set_title("Daily Average Forecasted Consumption (30 Days)", fontsize=10)
    set_style(ax2, fig2)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    # 30-day daily table only
    st.markdown('<div class="section-header">30-Day Forecast Summary (Daily)</div>',
                unsafe_allow_html=True)
    daily_display = daily.copy()
    daily_display.index = range(1, len(daily_display) + 1)
    daily_display.index.name = "Day"
    daily_display["Avg MW"]   = daily_display["Avg MW"].round(1)
    daily_display["Min MW"]   = daily_display["Min MW"].round(1)
    daily_display["Max MW"]   = daily_display["Max MW"].round(1)
    daily_display["Range MW"] = (daily_display["Max MW"] - daily_display["Min MW"]).round(1)
    st.dataframe(daily_display, use_container_width=True, height=740)

    csv = daily_display.to_csv()
    st.download_button("Download 30-Day Forecast CSV", csv,
                       "pjm_30day_forecast.csv", "text/csv")

# ═══════════════════════════════════════════════════════════════════════════
#  PAGE 4 — RESIDUAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Residual Analysis":
    st.markdown("## Residual Analysis")

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.patch.set_facecolor(PLT_BG)
    sample = test_df.sample(min(3000, len(test_df)), random_state=42)

    # 1. Residuals over time
    axes[0,0].plot(test_df["Datetime"].iloc[::10],
                   test_df["Residual"].iloc[::10],
                   color=C_TEAL, linewidth=0.6, alpha=0.7)
    axes[0,0].axhline(0, color=C_CORAL, linewidth=1.2, linestyle="--")
    axes[0,0].set_title("Residuals Over Time", fontsize=10)
    axes[0,0].set_xlabel("Date", fontsize=9)
    axes[0,0].set_ylabel("Residual (MW)", fontsize=9)
    set_style(axes[0,0])

    # 2. Residual histogram
    axes[0,1].hist(test_df["Residual"], bins=70, color=C_PURPLE,
                   edgecolor="none", alpha=0.85)
    axes[0,1].axvline(0, color=C_CORAL, linewidth=1.5, linestyle="--")
    axes[0,1].axvline(test_df["Residual"].mean(), color=C_AMBER,
                      linewidth=1.2, linestyle=":")
    axes[0,1].set_title("Residual Distribution", fontsize=10)
    axes[0,1].set_xlabel("Error (MW)", fontsize=9)
    axes[0,1].set_ylabel("Frequency", fontsize=9)
    set_style(axes[0,1])

    # 3. Residuals by hour
    hourly_res = test_df.groupby("Hour")["Residual"].mean()
    colors_hr  = [C_CORAL if v < 0 else C_TEAL for v in hourly_res]
    axes[1,0].bar(hourly_res.index, hourly_res.values,
                  color=colors_hr, edgecolor="none", width=0.7)
    axes[1,0].axhline(0, color="#aaa", linewidth=0.8, linestyle="--")
    axes[1,0].set_title("Mean Residual by Hour of Day", fontsize=10)
    axes[1,0].set_xlabel("Hour", fontsize=9)
    axes[1,0].set_ylabel("Mean Error (MW)", fontsize=9)
    set_style(axes[1,0])

    # 4. Residuals by month
    monthly_res = test_df.groupby("Month")["Residual"].mean()
    colors_mo   = [C_CORAL if v < 0 else C_TEAL for v in monthly_res]
    axes[1,1].bar(monthly_res.index, monthly_res.values,
                  color=colors_mo, edgecolor="none", width=0.7)
    axes[1,1].axhline(0, color="#aaa", linewidth=0.8, linestyle="--")
    axes[1,1].set_xticks(range(1, 13))
    axes[1,1].set_xticklabels(
        ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
        fontsize=8)
    axes[1,1].set_title("Mean Residual by Month", fontsize=10)
    axes[1,1].set_xlabel("Month", fontsize=9)
    axes[1,1].set_ylabel("Mean Error (MW)", fontsize=9)
    set_style(axes[1,1])

    plt.suptitle("Residual Analysis", color="#e0e0e0", fontsize=12, y=1.01)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Stats
    st.markdown('<div class="section-header">Residual Statistics</div>',
                unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Mean Error", f"{test_df['Residual'].mean():.2f} MW")
    c2.metric("Std Dev",    f"{test_df['Residual'].std():.2f} MW")
    c3.metric("Max Overestimate", f"{test_df['Residual'].min():.0f} MW")
    c4.metric("Max Underestimate", f"{test_df['Residual'].max():.0f} MW")

# ═══════════════════════════════════════════════════════════════════════════
#  PAGE 5 — FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Feature Importance":
    st.markdown("## Feature Importance")

    feat_names = ["Hour","Day","Month","Year","Quarter",
                  "DayOfWeek","IsWeekend","Season","IsHoliday","Rolling_7_Day"]
    importances = xgb_model.feature_importances_
    fi_df = pd.DataFrame({"Feature": feat_names, "Importance": importances})
    fi_df = fi_df.sort_values("Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(PLT_BG)
    bar_colors = [C_TEAL if v >= fi_df["Importance"].median() else C_BLUE
                  for v in fi_df["Importance"]]
    bars = ax.barh(fi_df["Feature"], fi_df["Importance"],
                   color=bar_colors, edgecolor="none", height=0.6)
    ax.bar_label(bars, fmt="%.3f", padding=4, color="#ccc", fontsize=9)
    ax.set_xlabel("Importance Score", fontsize=10)
    ax.set_title("XGBoost Feature Importance", fontsize=11)
    set_style(ax, fig)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown('<div class="section-header">Top Important Features</div>',
                unsafe_allow_html=True)
    fi_sorted = fi_df.sort_values("Importance", ascending=False).reset_index(drop=True)
    fi_sorted.index += 1
    fi_sorted["Importance"] = fi_sorted["Importance"].round(4)
    fi_sorted["Cumulative %"] = (fi_sorted["Importance"].cumsum() /
                                  fi_sorted["Importance"].sum() * 100).round(1)
    st.dataframe(fi_sorted, use_container_width=True)

    # Insight cards
    st.markdown('<div class="section-header">Key Insights</div>',
                unsafe_allow_html=True)
    top3 = fi_sorted["Feature"].iloc[:3].tolist()
    insights = {
        "IsWeekend": "Weekend vs weekday is the strongest signal — demand drops significantly on weekends.",
        "Hour": "Hour of day drives consumption peaks (morning & evening rush hours).",
        "Rolling_7_Day": "The 7-day rolling average captures weekly demand trends effectively.",
        "Season": "Seasonal patterns reflect summer AC loads and winter heating cycles.",
        "IsHoliday": "Holidays show notable demand reduction similar to weekends.",
    }
    cols = st.columns(min(3, len(top3)))
    for i, feat in enumerate(top3):
        with cols[i]:
            st.info(f"**{feat}**\n\n{insights.get(feat, 'Important predictor for energy demand.')}")

# ═══════════════════════════════════════════════════════════════════════════
#  PAGE 6 — EDA INSIGHTS
# ═══════════════════════════════════════════════════════════════════════════
elif page == "EDA Insights":
    st.markdown("## EDA Insights")

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.patch.set_facecolor(PLT_BG)

    # 1. Avg consumption by hour
    hourly = df.groupby("Hour")["PJMW_MW"].mean()
    axes[0,0].plot(hourly.index, hourly.values, color=C_TEAL,
                   linewidth=2, marker="o", markersize=4)
    axes[0,0].fill_between(hourly.index, hourly.values, alpha=0.15, color=C_TEAL)
    axes[0,0].set_title("Avg Consumption by Hour", fontsize=10)
    axes[0,0].set_xlabel("Hour of Day", fontsize=9)
    axes[0,0].set_ylabel("Avg MW", fontsize=9)
    set_style(axes[0,0])

    # 2. Avg by month
    monthly = df.groupby("Month")["PJMW_MW"].mean()
    bar_cols = [C_AMBER if v > monthly.mean() else C_BLUE for v in monthly]
    axes[0,1].bar(monthly.index, monthly.values, color=bar_cols,
                  edgecolor="none", width=0.7)
    axes[0,1].set_xticks(range(1, 13))
    axes[0,1].set_xticklabels(
        ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],
        fontsize=8)
    axes[0,1].set_title("Avg Consumption by Month", fontsize=10)
    axes[0,1].set_xlabel("Month", fontsize=9)
    axes[0,1].set_ylabel("Avg MW", fontsize=9)
    set_style(axes[0,1])

    # 3. Weekday vs weekend
    wd_labels = ["Weekday", "Weekend"]
    wd_means  = [df[df["IsWeekend"]==0]["PJMW_MW"].mean(),
                 df[df["IsWeekend"]==1]["PJMW_MW"].mean()]
    axes[1,0].bar(wd_labels, wd_means, color=[C_TEAL, C_CORAL],
                  edgecolor="none", width=0.45)
    for i, v in enumerate(wd_means):
        axes[1,0].text(i, v + 10, f"{v:,.0f}", ha="center",
                       fontsize=9, color="#ccc")
    axes[1,0].set_title("Weekday vs Weekend Avg Consumption", fontsize=10)
    axes[1,0].set_ylabel("Avg MW", fontsize=9)
    set_style(axes[1,0])

    # 4. Season boxplot
    season_order = ["Spring", "Summer", "Autumn", "Winter"]
    season_colors = [C_TEAL, C_AMBER, C_CORAL, C_PURPLE]
    data_by_season = [df[df["Season"]==s]["PJMW_MW"].values for s in season_order]
    bp = axes[1,1].boxplot(data_by_season, labels=season_order, patch_artist=True,
                            medianprops=dict(color="white", linewidth=1.5),
                            whiskerprops=dict(color="#aaa"),
                            capprops=dict(color="#aaa"),
                            flierprops=dict(marker=".", color="#555", markersize=2))
    for patch, color in zip(bp["boxes"], season_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    axes[1,1].set_title("Consumption by Season", fontsize=10)
    axes[1,1].set_ylabel("MW", fontsize=9)
    set_style(axes[1,1])

    plt.suptitle("Exploratory Data Analysis", color="#e0e0e0", fontsize=12, y=1.01)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # Yearly trend
    st.markdown('<div class="section-header">Yearly Consumption Trend</div>',
                unsafe_allow_html=True)
    yearly = df.groupby("Year")["PJMW_MW"].mean().reset_index()
    fig2, ax2 = plt.subplots(figsize=(10, 3.5))
    fig2.patch.set_facecolor(PLT_BG)
    ax2.plot(yearly["Year"], yearly["PJMW_MW"], color=C_TEAL,
             linewidth=2, marker="o", markersize=6)
    ax2.fill_between(yearly["Year"], yearly["PJMW_MW"], alpha=0.15, color=C_TEAL)
    for _, row in yearly.iterrows():
        ax2.annotate(f"{row['PJMW_MW']:,.0f}", (row["Year"], row["PJMW_MW"]),
                     textcoords="offset points", xytext=(0, 8),
                     ha="center", fontsize=8, color="#ccc")
    ax2.set_title("Yearly Average Energy Consumption", fontsize=10)
    ax2.set_xlabel("Year", fontsize=9)
    ax2.set_ylabel("Avg MW", fontsize=9)
    set_style(ax2, fig2)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()
