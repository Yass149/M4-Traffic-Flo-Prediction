# traffic_visualizations.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import numpy as np
import traceback

plt.ioff()  # Disable interactive plotting

# ============================
# Helpers
# ============================
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def save_plot(fig, path):
    try:
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"   ✔ Saved:", path)
    except Exception as e:
        print(f"   ❌ Failed to save {path}: {e}")


def safe_plot(title, out_path, plot_func):
    """
    Wraps each plot in a try/catch so one failure does not stop the pipeline.
    Same pattern used in Weather Visualisations.
    """
    print(f"\n➡ Generating plot: {title}")
    try:
        fig = plt.figure(figsize=(14, 6))
        plot_func(fig)
        save_plot(fig, out_path)
    except Exception as e:
        print(f"   ❌ Error generating {title}: {e}")
        traceback.print_exc()


# ============================
# Main Function
# ============================
def generate_traffic_visualisations(out="plots/traffic"):
    print("\n====================================")
    print("🔵 GENERATING TRAFFIC VISUALISATIONS")
    print("====================================\n")

    ensure_dir(out)

    # ----------- Load CSVs -----------
    print("📥 Loading CSV files from data/traffic ...")
    data_files = glob.glob("data/traffic/*.csv")

    if not data_files:
        raise FileNotFoundError("No CSV files found in data/traffic/")

    print("   ✔ Files found:", len(data_files))

    df_list = []
    for f in data_files:
        try:
            part = pd.read_csv(f)
            df_list.append(part)
            print("   ✔ Loaded:", f)
        except Exception as e:
            print("   ❌ Failed to load:", f, e)

    df = pd.concat(df_list, ignore_index=True)
    print("\n📊 Total rows loaded:", len(df))

    # ----------- Fix timestamps -----------
    print("\n🕒 Fixing timestamp formats ...")
    try:
        df["Report Date"] = pd.to_datetime(df["Report Date"], dayfirst=True, errors="coerce")
        df["Hour"] = pd.to_datetime(df["Time Period Ending"], dayfirst=True, errors="coerce").dt.hour
        print("   ✔ Timestamp parsing successful")
    except Exception as e:
        print("   ❌ Timestamp parse error:", e)

    # ----------- Speed columns -----------
    speed_cols = [
        '0 - 10 mph','11 - 15 mph','16 - 20 mph','21 - 25 mph','26 - 30 mph',
        '31 - 35 mph','36 - 40 mph','41 - 45 mph','46 - 50 mph','51 - 55 mph',
        '56 - 60 mph','61 - 70 mph','71 - 80 mph','80+ mph'
    ]

    # ----------- Clip extreme values -----------
    print("\n📉 Clipping extreme volumes (top 10%) ...")
    try:
        vol_upper = df['Total Volume'].quantile(0.90)
        df['Total Volume Clipped'] = df['Total Volume'].clip(upper=vol_upper)
        print("   ✔ Clipping done")
    except Exception as e:
        print("   ❌ Error clipping:", e)

    # ============================
    # PLOTS
    # ============================

    # ---- 1. Timeseries ----
    safe_plot(
        "Time Series: Total Volume (Clipped)",
        f"{out}/total_volume_timeseries.png",
        lambda fig: sns.lineplot(x='Report Date', y='Total Volume Clipped', data=df)
    )

    # ---- 2. Rolling average ----
    safe_plot(
        "Rolling Average Volume (24h)",
        f"{out}/total_volume_rolling.png",
        lambda fig: sns.lineplot(
            x='Report Date',
            y=df.sort_values('Report Date')['Total Volume Clipped']
            .rolling(24).mean(),
            color='orange'
        )
    )

    # ---- 3. Speed distribution ----
    def plot_speed_dist(fig):
        df_speed = df.melt(id_vars=['Report Date'], value_vars=speed_cols,
                           var_name='Speed Range', value_name='Count')
        sns.barplot(x='Speed Range', y='Count', data=df_speed, ci=None)
        plt.xticks(rotation=90)

    safe_plot(
        "Speed Distribution",
        f"{out}/speed_distribution.png",
        plot_speed_dist
    )

    # ---- 4. Hour × Day Type ----
    if "Day Type" in df.columns:
        def plot_hour_daytype(fig):
            pivot = df.pivot_table(
                index='Hour',
                columns='Day Type',
                values='Total Volume Clipped',
                aggfunc='sum'
            )
            sns.heatmap(pivot, annot=True, fmt=".0f", cmap='YlGnBu')

        safe_plot(
            "Heatmap: Hour × Day Type",
            f"{out}/volume_hour_daytype_heatmap.png",
            plot_hour_daytype
        )

    # ---- 5. Cumulative site volume ----
    safe_plot(
        "Volume by Site",
        f"{out}/volume_by_site.png",
        lambda fig: sns.barplot(
            x=df.groupby('Site Name')['Total Volume Clipped']
            .sum()
            .sort_values(ascending=False).index,
            y=df.groupby('Site Name')['Total Volume Clipped']
            .sum()
            .sort_values(ascending=False).values
        ) or plt.xticks(rotation=90)
    )

    # ---- 6. Stacked area speeds ----
    def plot_area(fig):
        df_speed_time = df.groupby('Report Date')[speed_cols].sum()
        df_speed_time.plot.area(ax=plt.gca(), cmap='tab20')
        plt.ylabel("Volume")

    safe_plot(
        "Stacked Area Speed Time Series",
        f"{out}/speed_area_timeseries.png",
        plot_area
    )

    # ---- 7. Correlation heatmap ----
    safe_plot(
        "Correlation Heatmap",
        f"{out}/speed_corr_heatmap.png",
        lambda fig: sns.heatmap(
            df[speed_cols + ['Total Volume Clipped']].corr(),
            annot=True, cmap='coolwarm'
        )
    )

    print("\n✅ ALL PLOTS SAVED in:", out)


# ============================
# Script Entry
# ============================
if __name__ == "__main__":
    try:
        generate_traffic_visualisations()
    except Exception as e:
        print("\n🔥 Fatal Error:", e)
        traceback.print_exc()
