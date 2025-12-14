# weather_visualisations.py  (Notebook-friendly, inline plots)

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from windrose import WindroseAxes
from sklearn.preprocessing import StandardScaler
import matplotlib.dates as mdates
import glob

plt.ioff()  # interactive mode off for script usage

# =============================
# Helper — Clean NOAA value formats
# =============================
def clean_noaa_value(col):
    return (
        col.astype(str)
        .str.split(",", expand=True)[0]
        .str.replace("+", "", regex=False)
        .str.replace("M", "", regex=False)
        .str.replace(" ", "", regex=False)
        .replace("", None)
        .astype(float)
    )

# =============================
# Main function
# =============================
def plot_weather_from_csv(show_plots=True, clip_top=0.90):
    """
    Reads all CSVs from data/weather/, processes them, and plots inline.

    Args:
        show_plots (bool): Whether to display plots inline (use True in notebooks)
        clip_top (float): Top percentile to clip TMP, DEW, SLP
    """
    data_files = glob.glob("data/weather/*.csv")
    if not data_files:
        raise FileNotFoundError("No CSV files found in data/weather/")

    numeric_noaa = ["TMP", "DEW", "SLP", "VIS", "CIG"]

    for file in data_files:
        print(f"\nProcessing: {file}\n{'='*60}")
        df = pd.read_csv(file, low_memory=False)

        # -----------------------------
        # Clean NOAA numeric columns
        # -----------------------------
        for col in numeric_noaa:
            if col in df.columns:
                df[col] = clean_noaa_value(df[col])
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # -----------------------------
        # Parse DATE column
        # -----------------------------
        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
        df = df.sort_values("DATE")

        # -----------------------------
        # Clip TMP, DEW, SLP
        # -----------------------------
        for col in ["TMP", "DEW", "SLP"]:
            if col in df.columns:
                upper = df[col].quantile(clip_top)
                df[f"{col}_clipped"] = df[col].clip(upper=upper)

        # -----------------------------
        # 1A — Subplots TMP, DEW, SLP
        # -----------------------------
        try:
            fig, axes = plt.subplots(3, 1, figsize=(15, 12), sharex=True)
            axes_info = [
                ("TMP_clipped", "Temperature (°C)", "tab:red"),
                ("DEW_clipped", "Dew Point (°C)", "tab:blue"),
                ("SLP_clipped", "Sea-Level Pressure", "tab:green"),
            ]

            for ax, (col, title, color) in zip(axes, axes_info):
                sns.lineplot(ax=ax, x="DATE", y=col, data=df, color=color)
                ax.set_title(title)

            axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
            axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
            plt.setp(axes[-1].get_xticklabels(), rotation=45)
            plt.tight_layout()
            if show_plots:
                plt.show()
            plt.close()
        except Exception as e:
            print("⚠ Subplot time series skipped:", e)

        # -----------------------------
        # 1B — Normalized plots
        # -----------------------------
        try:
            scaler = StandardScaler()
            df_norm = df.copy()
            for col in ["TMP_clipped", "DEW_clipped", "SLP_clipped"]:
                df_norm[f"{col.replace('_clipped','')}_norm"] = scaler.fit_transform(df_norm[[col]])

            for col, color in zip(["TMP_norm", "DEW_norm", "SLP_norm"], ["tab:red","tab:blue","tab:green"]):
                plt.figure(figsize=(15,6))
                sns.lineplot(x="DATE", y=col, data=df_norm, color=color)
                plt.title(f"Normalized {col.replace('_norm','').upper()} Over Time")
                plt.xlabel("Date")
                plt.ylabel("Z-score")
                plt.tight_layout()
                if show_plots:
                    plt.show()
                plt.close()
        except Exception as e:
            print("⚠ Normalised time series skipped:", e)

        # -----------------------------
        # 1C — Monthly averages
        # -----------------------------
        try:
            monthly = df.set_index("DATE").resample("M")[[f"{c}_clipped" for c in ["TMP","DEW","SLP"]]].mean().reset_index()
            for col in ["TMP_clipped", "DEW_clipped", "SLP_clipped"]:
                plt.figure(figsize=(15,6))
                sns.lineplot(x="DATE", y=col, data=monthly)
                plt.title(f"Monthly Average {col.replace('_clipped','')}")
                plt.xlabel("Date")
                plt.ylabel(col.replace('_clipped',''))
                plt.tight_layout()
                if show_plots:
                    plt.show()
                plt.close()
        except Exception as e:
            print("⚠ Monthly avg time series skipped:", e)

        # -----------------------------
        # 2 — Wind Rose
        # -----------------------------
        try:
            df[["wind_dir","wind_speed"]] = df["WND"].astype(str).str.split(",", n=2, expand=True)[[0,1]].astype(float)
            ax = WindroseAxes.from_ax()
            ax.bar(df["wind_dir"], df["wind_speed"], normed=True, opening=0.8, edgecolor="white")
            ax.set_title("Wind Rose")
            if show_plots:
                plt.show()
            plt.close()
        except Exception as e:
            print("⚠ Wind rose skipped:", e)

        # -----------------------------
        # 3 — Correlation heatmap
        # -----------------------------
        try:
            plt.figure(figsize=(12,10))
            sns.heatmap(df[[f"{c}_clipped" for c in ["TMP","DEW","SLP"]]].corr(), annot=True, cmap="coolwarm")
            plt.title("Correlation Heatmap")
            plt.tight_layout()
            if show_plots:
                plt.show()
            plt.close()
        except Exception as e:
            print("⚠ Heatmap skipped:", e)

        # -----------------------------
        # 4 — Quarterly boxplot TMP
        # -----------------------------
        try:
            df["QUARTER"] = df["DATE"].dt.quarter.replace({1:"Q1",2:"Q2",3:"Q3",4:"Q4"})
            plt.figure(figsize=(12,6))
            sns.boxplot(x="QUARTER", y="TMP_clipped", data=df)
            plt.title("Temperature Distribution by Quarter (Top 10% Clipped)")
            plt.tight_layout()
            if show_plots:
                plt.show()
            plt.close()
        except Exception as e:
            print("⚠ Quarterly boxplot skipped:", e)
         
        # -----------------------------
        # 5 — Optional pairplot
        # -----------------------------   
            
        try:
            sns.pairplot(df[[f"{c}_clipped" for c in ["TMP","DEW","SLP"]]])
            if show_plots:
                plt.show()
            plt.close()
        except Exception as e:
            print("⚠ Pairplot skipped:", e)

    print("\n✅ Finished processing all weather files.\n")
