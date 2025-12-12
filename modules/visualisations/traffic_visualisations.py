# traffic_visualisations.py
"""
Traffic visualisations A-F with robust output & labeling.
Saves plots to plots/traffic/
"""

import os
import glob
import traceback
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
from statsmodels.tsa.seasonal import seasonal_decompose

plt.ioff()
sns.set(style="whitegrid")

# -----------------------
# Helpers
# -----------------------
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_fig(path):
    try:
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"   ✔ Saved: {path}")
    except Exception as e:
        print(f"   ❌ Failed saving {path}: {e}")
        plt.close()

def safe_plot(title, path, plot_fn):
    """Run plot_fn() and save figure; robust to exceptions."""
    print(f"\n--> {title}")
    try:
        plot_fn()
        save_fig(path)
    except Exception as e:
        print(f"   ❌ Error creating {title}: {e}")
        traceback.print_exc()
        plt.close()

# -----------------------
# Core pipeline
# -----------------------
def generate_traffic_visualisations(out_root="plots/traffic", rolling_window=24, clip_pct=0.90):
    ensure_dir(out_root)

    # ---------- Load data ----------
    files = glob.glob("data/traffic/*.csv")
    if not files:
        raise FileNotFoundError("No traffic CSVs found in data/traffic/")
    print("Loading files:", files)

    parts = []
    for f in files:
        try:
            parts.append(pd.read_csv(f))
            print("  loaded:", f)
        except Exception as e:
            print("  failed load:", f, e)
    df = pd.concat(parts, ignore_index=True)
    print("Rows loaded:", len(df))

    # ---------- Parse timestamps (dayfirst) ----------
    df["Report Date"] = pd.to_datetime(df.get("Report Date"), dayfirst=True, errors="coerce")
    df["Time Period Ending Parsed"] = pd.to_datetime(df.get("Time Period Ending"), dayfirst=True, errors="coerce")
    # Hour of day for intraday analysis
    df["Hour"] = df["Time Period Ending Parsed"].dt.hour
    df["Weekday"] = df["Report Date"].dt.day_name()
    df["Date"] = df["Report Date"].dt.date

    # ---------- Columns of interest ----------
    size_cols = ["0 - 520 cm", "521  - 660 cm", "661 - 1160 cm", "1160+ cm"]
    # normalize column names (strip)
    df.columns = [c.strip() for c in df.columns]

    # convert numeric columns
    for c in size_cols + ["Avg mph", "Total Volume"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # drop rows without total volume
    df = df[df["Total Volume"].notna()].copy()
    if df.empty:
        raise ValueError("No rows with Total Volume found after parsing.")

    # ---------- Clip top X% of Total Volume ----------
    vol_cut = df["Total Volume"].quantile(clip_pct)
    df["Total Volume Clipped"] = df["Total Volume"].clip(upper=vol_cut)

    # ---------- Output folder ----------
    ensure_dir(out_root)

    # ---------- Produce a small summary CSV ----------
    summary = {
        "rows": len(df),
        "start_date": df["Report Date"].min(),
        "end_date": df["Report Date"].max(),
        "sites": df["Site Name"].nunique() if "Site Name" in df.columns else None,
        "total_volume_sum": df["Total Volume"].sum(),
        "total_volume_clipped_sum": df["Total Volume Clipped"].sum(),
    }
    pd.DataFrame([summary]).to_csv(os.path.join(out_root, "summary_stats.csv"), index=False)
    print("Wrote summary_stats.csv")

    # -----------------------
    # A: Traffic flow over time
    # -----------------------
    def plot_total_timeseries():
        plt.figure(figsize=(14,5))
        sns.lineplot(x="Report Date", y="Total Volume Clipped", data=df, linewidth=1)
        plt.title("Total Traffic Volume Over Time (top 10% clipped)")
        plt.xlabel("Date")
        plt.ylabel("Total Volume (vehicles)")
    safe_plot("A - Total Volume Time Series", os.path.join(out_root, "A_total_volume_timeseries.png"), plot_total_timeseries)

    def plot_rolling_mean_median():
        ser = df.sort_values("Report Date").set_index("Report Date")["Total Volume Clipped"]
        rolling_mean = ser.rolling(window=rolling_window, min_periods=1).mean()
        rolling_median = ser.rolling(window=rolling_window, min_periods=1).median()
        plt.figure(figsize=(14,5))
        plt.plot(rolling_mean.index, rolling_mean.values, label=f"{rolling_window}-period mean", linewidth=1.5)
        plt.plot(rolling_median.index, rolling_median.values, label=f"{rolling_window}-period median", linewidth=1.2, linestyle="--")
        plt.legend()
        plt.title(f"Rolling mean & median (window={rolling_window})")
        plt.xlabel("Date")
        plt.ylabel("Total Volume (vehicles)")
    safe_plot("A - Rolling mean & median", os.path.join(out_root, "A_rolling_mean_median.png"), plot_rolling_mean_median)

    # -----------------------
    # B: Volume by vehicle size class
    # -----------------------
    def plot_size_stacked_area():
        grouped = df.groupby("Report Date")[size_cols].sum().fillna(0)
        if grouped.sum().sum() == 0:
            raise ValueError("No vehicle-size data available for stacked area.")
        plt.figure(figsize=(14,6))
        grouped.plot.area(ax=plt.gca(), cmap="tab20")
        plt.title("Vehicle Size Class Volume Over Time (stacked)")
        plt.xlabel("Date")
        plt.ylabel("Number of vehicles")
    safe_plot("B - Stacked area: size classes", os.path.join(out_root, "B_size_stacked_area.png"), plot_size_stacked_area)

    def plot_size_distribution_hist():
        melted = df.melt(id_vars=["Report Date"], value_vars=size_cols, var_name="Size", value_name="Count")
        plt.figure(figsize=(12,6))
        sns.histplot(data=melted, x="Count", hue="Size", element="step", stat="count", bins=40)
        plt.title("Distribution of Counts per Vehicle Size Class")
        plt.xlabel("Count (vehicles per period)")
        plt.ylabel("Frequency")
    safe_plot("B - Size distribution histogram", os.path.join(out_root, "B_size_distribution_hist.png"), plot_size_distribution_hist)

    def plot_size_boxplots_faceted():
        # create 2x2 grid where each subplot shows boxplot for one size class with its own y-scale
        fig, axes = plt.subplots(2,2, figsize=(14,10))
        axes = axes.flatten()
        for ax, col in zip(axes, size_cols):
            if col not in df.columns:
                ax.text(0.5,0.5,f"{col} missing", ha='center')
                continue
            sns.boxplot(x=df[col], ax=ax)
            ax.set_title(f"Boxplot: {col}")
            ax.set_xlabel(f"Count (vehicles) for {col}")
        plt.suptitle("Vehicle Size Class Boxplots (each subplot own y-scale)")
        plt.tight_layout(rect=[0,0,1,0.95])
    safe_plot("B - Size-class boxplots (faceted)", os.path.join(out_root, "B_size_boxplots_faceted.png"), plot_size_boxplots_faceted)

    # -----------------------
    # C: Hourly & daily traffic patterns
    # -----------------------
    def plot_hour_weekday_heatmap():
        pivot = df.pivot_table(index="Hour", columns="Weekday", values="Total Volume Clipped", aggfunc="mean")
        # Reorder weekdays Monday..Sunday
        weekdays = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        cols_present = [d for d in weekdays if d in pivot.columns]
        plt.figure(figsize=(12,6))
        sns.heatmap(pivot[cols_present], annot=False, fmt=".0f", cmap="YlGnBu")
        plt.title("Average Traffic Volume: Hour (rows) × Weekday (cols)")
        plt.xlabel("Weekday")
        plt.ylabel("Hour of day")
    safe_plot("C - Hour × Weekday heatmap", os.path.join(out_root, "C_hour_weekday_heatmap.png"), plot_hour_weekday_heatmap)

    def plot_peak_hour_analysis():
        hourly_sum = df.groupby("Hour")["Total Volume Clipped"].sum()
        top_hours = hourly_sum.sort_values(ascending=False).head(10)
        plt.figure(figsize=(10,5))
        sns.barplot(x=top_hours.index.astype(str), y=top_hours.values)
        plt.title("Peak Hours (top 10 by summed volume)")
        plt.xlabel("Hour of day")
        plt.ylabel("Total Volume (vehicles)")
    safe_plot("C - Peak hour analysis", os.path.join(out_root, "C_peak_hours_top10.png"), plot_peak_hour_analysis)

    def plot_average_daily_profile():
        avg_profile = df.groupby("Hour")["Total Volume Clipped"].mean()
        plt.figure(figsize=(12,5))
        sns.lineplot(x=avg_profile.index, y=avg_profile.values)
        plt.title("Average Daily Profile (mean volume by hour)")
        plt.xlabel("Hour of day")
        plt.ylabel("Mean Total Volume (vehicles)")
        plt.xticks(range(0,24))
    safe_plot("C - Average daily profile", os.path.join(out_root, "C_average_daily_profile.png"), plot_average_daily_profile)

    # -----------------------
    # D: Traffic flow stability (CV, anomalies)
    # -----------------------
    def plot_cv_by_hour():
        byh = df.groupby("Hour")["Total Volume Clipped"].agg(["mean","std"])
        byh["cv"] = byh["std"] / byh["mean"].replace(0, np.nan)
        plt.figure(figsize=(12,5))
        sns.lineplot(x=byh.index, y="cv", data=byh)
        plt.title("Coefficient of Variation (CV) by Hour")
        plt.xlabel("Hour")
        plt.ylabel("CV (std / mean)")
    safe_plot("D - CV by hour", os.path.join(out_root, "D_cv_by_hour.png"), plot_cv_by_hour)

    def plot_anomalies():
        s = df.sort_values("Report Date")["Total Volume Clipped"].reset_index(drop=True)
        # z-score anomalies
        z = (s - s.mean()) / s.std()
        anomalies_z = z[np.abs(z) > 3].index
        # IQR anomalies
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        anomalies_iqr = s[(s < (q1 - 1.5*iqr)) | (s > (q3 + 1.5*iqr))].index

        dates = df.sort_values("Report Date")["Report Date"].reset_index(drop=True)
        plt.figure(figsize=(14,5))
        sns.lineplot(x=dates, y=s, label="Volume (clipped)")
        if len(anomalies_z):
            plt.scatter(dates.iloc[anomalies_z], s.iloc[anomalies_z], color="red", label="z-score anomalies")
        if len(anomalies_iqr):
            plt.scatter(dates.iloc[anomalies_iqr], s.iloc[anomalies_iqr], color="orange", label="IQR anomalies", marker="x")
        plt.title("Anomalies in Total Volume (z-score and IQR)")
        plt.xlabel("Date")
        plt.ylabel("Total Volume (vehicles)")
        plt.legend()
    safe_plot("D - Anomaly detection", os.path.join(out_root, "D_anomalies.png"), plot_anomalies)

    # -----------------------
    # E: Trends across months / seasons
    # -----------------------
    def plot_monthly_trends():
        monthly = df.set_index("Report Date").resample("M")["Total Volume Clipped"].sum().reset_index()
        monthly["Month"] = monthly["Report Date"].dt.to_period("M").astype(str)
        plt.figure(figsize=(12,5))
        sns.lineplot(x="Report Date", y="Total Volume Clipped", data=monthly, marker="o")
        plt.title("Monthly Total Volume")
        plt.xlabel("Month")
        plt.ylabel("Total Volume (vehicles)")
    safe_plot("E - Monthly totals", os.path.join(out_root, "E_monthly_totals.png"), plot_monthly_trends)

    def plot_month_on_month_change():
        monthly = df.set_index("Report Date").resample("M")["Total Volume Clipped"].sum()
        mom = monthly.pct_change().dropna() * 100.0
        plt.figure(figsize=(12,5))
        sns.barplot(x=mom.index.astype(str), y=mom.values)
        plt.xticks(rotation=45)
        plt.title("Month-on-Month % Change in Total Volume")
        plt.xlabel("Month")
        plt.ylabel("% change")
    safe_plot("E - Month-on-Month change", os.path.join(out_root, "E_mom_change.png"), plot_month_on_month_change)

    def plot_seasonal_decompose():
        try:
            
            ser = df.set_index("Report Date")["Total Volume Clipped"].resample("D").sum().fillna(method="ffill")
            res = seasonal_decompose(ser, model="additive", period=7, two_sided=False)
            plt.figure(figsize=(14,9))
            ax1 = plt.subplot(4,1,1); res.observed.plot(ax=ax1); ax1.set_title("Observed")
            ax2 = plt.subplot(4,1,2); res.trend.plot(ax=ax2); ax2.set_title("Trend")
            ax3 = plt.subplot(4,1,3); res.seasonal.plot(ax=ax3); ax3.set_title("Seasonal")
            ax4 = plt.subplot(4,1,4); res.resid.plot(ax=ax4); ax4.set_title("Residual")
            plt.tight_layout()
        except Exception as e:
            raise RuntimeError("seasonal_decompose failed (install statsmodels?)") from e
    safe_plot("E - Seasonal decomposition (opt)", os.path.join(out_root, "E_seasonal_decompose.png"), plot_seasonal_decompose)

    # -----------------------
    # F: Correlations (Avg mph, volume, size-classes)
    # -----------------------
    def plot_weather_traffic_corr_like():
        cols = ["Total Volume Clipped"]
        if "Avg mph" in df.columns:
            cols.append("Avg mph")
        for c in size_cols:
            if c in df.columns:
                cols.append(c)
        available = [c for c in cols if c in df.columns]
        if len(available) < 2:
            raise ValueError("Not enough columns for correlation heatmap")
        corr = df[available].corr()
        plt.figure(figsize=(8,6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation: volume, avg mph, size-classes")
    safe_plot("F - Correlation heatmap", os.path.join(out_root, "F_corr_heatmap.png"), plot_weather_traffic_corr_like)

    def plot_avgmph_vs_volume():
        if "Avg mph" not in df.columns:
            raise ValueError("Avg mph not present")
        plt.figure(figsize=(10,6))
        sns.scatterplot(x="Avg mph", y="Total Volume Clipped", data=df, alpha=0.5)
        sns.regplot(x="Avg mph", y="Total Volume Clipped", data=df, scatter=False, color="red")
        plt.title("Average Speed vs Volume (clipped)")
        plt.xlabel("Avg speed (mph)")
        plt.ylabel("Total Volume (vehicles)")
    safe_plot("F - Avg mph vs Volume", os.path.join(out_root, "F_avgmph_vs_volume.png"), plot_avgmph_vs_volume)

    print("\n✅ All A-F plots generated. Check folder:", out_root)

if __name__ == "__main__":
    generate_traffic_visualisations()
