# traffic_weather_relations.py
"""
Merges traffic data with weather and creates relationship visualisations.
Saves to plots/traffic_weather/
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import numpy as np
import traceback

plt.ioff()
sns.set(style="whitegrid")

# -----------------------
# Helpers
# -----------------------
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_plot(path):
    try:
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"   ✔ Saved: {path}")
    except Exception as e:
        print(f"   ❌ Failed saving {path}: {e}")

def safe_plot(title, path, plot_fn):
    print(f"\n--> {title}")
    try:
        plot_fn()
        save_plot(path)
    except Exception as e:
        print(f"   ❌ Error creating {title}: {e}")
        traceback.print_exc()
        plt.close()

# -----------------------
# Helper to clean NOAA fields like "+0075,1"
# -----------------------
def clean_noaa_value(col):
    return (
        col.astype(str)
           .str.split(",", expand=True)[0]
           .str.replace("+", "", regex=False)
           .str.replace("M", "", regex=False)
           .str.strip()
    )

# -----------------------
# Main
# -----------------------
def generate_traffic_weather(out_root="plots/traffic_weather"):
    ensure_dir(out_root)

    # Load traffic
    t_files = glob.glob("data/traffic/*.csv")
    if not t_files:
        raise FileNotFoundError("No traffic CSVs found in data/traffic/")
    tdfs = [pd.read_csv(f) for f in t_files]
    tdf = pd.concat(tdfs, ignore_index=True)
    tdf["Report Date"] = pd.to_datetime(tdf["Report Date"], dayfirst=True, errors="coerce")
    tdf["Report Date Hour"] = tdf["Report Date"].dt.floor("H")
    tdf["Time Period Ending Parsed"] = pd.to_datetime(tdf["Time Period Ending"], dayfirst=True, errors="coerce")
    for c in ["Total Volume","Avg mph"]:
        if c in tdf.columns:
            tdf[c] = pd.to_numeric(tdf[c], errors="coerce")
    # clip top 10%
    if "Total Volume" in tdf.columns:
        tdf["Total Volume Clipped"] = tdf["Total Volume"].clip(upper=tdf["Total Volume"].quantile(0.90))

    # Load weather (use same cleaning as weather script)
    w_files = glob.glob("data/weather/*.csv")
    if not w_files:
        raise FileNotFoundError("No weather CSVs found in data/weather/")
    wdfs = []
    for f in w_files:
        w = pd.read_csv(f, low_memory=False)
        # clean TMP, DEW, SLP, VIS if present
        for col in ["TMP","DEW","SLP","VIS","CIG"]:
            if col in w.columns:
                w[col] = clean_noaa_value(w[col])
                w[col] = pd.to_numeric(w[col], errors="coerce")
        w["DATE"] = pd.to_datetime(w["DATE"], errors="coerce")
        w["DATE_HOUR"] = w["DATE"].dt.floor("H")
        wdfs.append(w)
    wdf = pd.concat(wdfs, ignore_index=True)

    # Merge traffic and weather by nearest hour (asof)
    tdf_sorted = tdf.sort_values("Report Date Hour")
    wdf_sorted = wdf.sort_values("DATE_HOUR")

    merged = pd.merge_asof(
        tdf_sorted,
        wdf_sorted,
        left_on="Report Date Hour",
        right_on="DATE_HOUR",
        direction="nearest",
        tolerance=pd.Timedelta("1H")
    )

    # drop rows where no weather matched (if many)
    merged = merged[merged["TMP"].notna() | merged["DEW"].notna() | merged["SLP"].notna()]

    out = out_root
    ensure_dir(out)

    # ---- 1. Volume vs Temperature (scatter + regression) ----
    def plot_vol_vs_temp():
        if "TMP" not in merged.columns:
            raise ValueError("TMP not present in merged data")
        plt.figure(figsize=(10,6))
        sns.scatterplot(x="TMP", y="Total Volume Clipped", data=merged, alpha=0.5)
        sns.regplot(x="TMP", y="Total Volume Clipped", data=merged, scatter=False, color="red")
        plt.title("Traffic Volume vs Temperature (°C)")
    safe_plot("Volume vs Temperature", os.path.join(out,"volume_vs_temp.png"), plot_vol_vs_temp)

    # ---- 2. Volume vs Dew Point ----
    def plot_vol_vs_dew():
        if "DEW" not in merged.columns:
            raise ValueError("DEW not present")
        plt.figure(figsize=(10,6))
        sns.scatterplot(x="DEW", y="Total Volume Clipped", data=merged, alpha=0.5)
        sns.regplot(x="DEW", y="Total Volume Clipped", data=merged, scatter=False, color="red")
        plt.title("Traffic Volume vs Dew Point (°C)")
    safe_plot("Volume vs Dew Point", os.path.join(out,"volume_vs_dew.png"), plot_vol_vs_dew)

    # ---- 3. Hourly heatmap of mean volume by temp bin ----
    def plot_hour_temp_heatmap():
        if "TMP" not in merged.columns:
            raise ValueError("TMP not present")
        merged["TMP_bin"] = pd.qcut(merged["TMP"].rank(method="first"), 8, labels=False)
        pivot = merged.pivot_table(index=merged["Report Date"].dt.hour, columns="TMP_bin", values="Total Volume Clipped", aggfunc="mean")
        plt.figure(figsize=(12,6))
        sns.heatmap(pivot, cmap="YlOrRd")
        plt.title("Mean Hourly Volume by Temperature bin")
        plt.xlabel("Temp bin")
        plt.ylabel("Hour of day")
    safe_plot("Hourly heatmap by temp bin", os.path.join(out,"hour_temp_heatmap.png"), plot_hour_temp_heatmap)

    # ---- 4. Rolling comparison volume vs temp ----
    def plot_rolling_comp():
        tmp = merged.sort_values("Report Date").set_index("Report Date")
        tmp["vol_roll_24"] = tmp["Total Volume Clipped"].rolling(24, min_periods=1).mean()
        tmp["tmp_roll_24"] = tmp["TMP"].rolling(24, min_periods=1).mean()
        plt.figure(figsize=(14,6))
        ax = plt.gca()
        ax2 = ax.twinx()
        sns.lineplot(x=tmp.index, y="vol_roll_24", data=tmp, ax=ax, label="Volume (24h rolling)", color="tab:blue")
        sns.lineplot(x=tmp.index, y="tmp_roll_24", data=tmp, ax=ax2, label="Temp (24h rolling)", color="tab:red")
        ax.set_ylabel("Volume")
        ax2.set_ylabel("Temperature (°C)")
        plt.title("Rolling Volume vs Temperature (24h)")
    safe_plot("Rolling Volume vs Temp", os.path.join(out,"rolling_vol_vs_temp.png"), plot_rolling_comp)

    # ---- 5. Boxplots: volume by weather condition (visibility bins) ----
    def plot_volume_by_vis():
        if "VIS" not in merged.columns:
            raise ValueError("VIS not present")
        merged["VIS_bin"] = pd.qcut(merged["VIS"].rank(method="first"), 4, labels=["low","med","high","very_high"])
        plt.figure(figsize=(10,6))
        sns.boxplot(x="VIS_bin", y="Total Volume Clipped", data=merged, order=["low","med","high","very_high"])
        plt.title("Volume by Visibility bins")
    try:
        safe_plot("Volume by Visibility bins", os.path.join(out,"volume_by_visibility.png"), plot_volume_by_vis)
    except Exception:
        pass

    # ---- 6. Correlations (weather vs traffic) ----
    def plot_weather_traffic_corr():
        cols = []
        if "Total Volume Clipped" in merged.columns:
            cols.append("Total Volume Clipped")
        for c in ["TMP","DEW","SLP","VIS","Avg mph"]:
            if c in merged.columns:
                cols.append(c)
        if len(cols) < 2:
            raise ValueError("Not enough columns to compute correlation")
        corr = merged[cols].corr()
        plt.figure(figsize=(8,6))
        sns.heatmap(corr, annot=True, cmap="coolwarm")
        plt.title("Correlation: traffic vs weather")
    safe_plot("Correlation heatmap (traffic/weather)", os.path.join(out,"weather_traffic_corr.png"), plot_weather_traffic_corr)

    # ---- 7. Scatter: Avg mph vs Volume colored by TMP quartile ----
    def plot_speed_volume_temp():
        if "Avg mph" not in merged.columns:
            raise ValueError("Avg mph not present")
        merged_local = merged.dropna(subset=["Avg mph","Total Volume Clipped","TMP"])
        merged_local["tmp_q"] = pd.qcut(merged_local["TMP"], 4, labels=False)
        plt.figure(figsize=(12,6))
        sns.scatterplot(x="Avg mph", y="Total Volume Clipped", hue="tmp_q", palette="viridis", data=merged_local, alpha=0.6)
        plt.title("Avg mph vs Volume colored by Temp quartile")
    try:
        safe_plot("Avg mph vs Volume by Temp quartile", os.path.join(out,"avgmph_vs_volume_tempq.png"), plot_speed_volume_temp)
    except Exception:
        pass

    print("\nDone. All traffic+weather visuals saved to:", out)

if __name__ == "__main__":
    generate_traffic_weather()
