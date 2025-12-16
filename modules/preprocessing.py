"""
Data Preprocessing Module for Traffic and Weather Analysis.

This module serves as the ETL (Extract, Transform, Load) pipeline for the project.
It contains two specialized processors and a merging logic:
1.  `WeatherPreprocessor`: Handles the complex decoding of NOAA ISD FM-12 weather data,
    including physical consistency checks (e.g., T >= Td) and unit conversions.
2.  `TrafficPreprocessor`: Standardizes traffic sensor data, aggregates multiple
    sensors, and resamples them to a regular grid.
3.  `process_and_merge`: Aligns the two distinct time series onto a single
    15-minute timeline using a left-join strategy to preserve ground-truth targets.

Dependencies:
    - pandas
    - numpy
    - logging
    - os, glob
"""

import os
import glob
import pandas as pd
import numpy as np
import logging

class WeatherPreprocessor:
    """
    End-to-end NOAA ISD FM-12 Weather Preprocessor for traffic modeling.

    This class transforms raw NOAA ISD CSV files into clean, 15-minute interval datasets.
    It handles the specific "FM-12" string encoding used by meteorological stations,
    applies physical constraints (e.g., clipping humidity to 0-100%), and prepares
    the data for time-series modeling.

    Attributes:
        weather_dir (str): Path to the directory containing raw CSV files.
        log (logging.Logger): Logger instance for tracking processing errors.
        FM12_FIELDS (list): List of specific NOAA columns required for decoding.
    """

    FM12_FIELDS = ["WND", "VIS", "TMP", "DEW", "SLP", "AA1", "CIG", "MA1", "OD1", "MD1"]

    def __init__(self, weather_dir: str = "data/weather", verbose: bool = False):
        """
        Initializes the weather preprocessor.

        Args:
            weather_dir (str, optional): Directory containing NOAA ISD CSV files. 
                                         Defaults to "data/weather".
            verbose (bool, optional): If True, sets logging level to INFO. 
                                      Defaults to False (WARNING).
        """
        self.weather_dir = weather_dir
        logging.basicConfig(
            level=logging.INFO if verbose else logging.WARNING,
            format="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%H:%M:%S"
        )
        self.log = logging.getLogger("WeatherPreprocessor")

    def load_raw(self) -> pd.DataFrame:
        """
        Loads and merges all NOAA weather CSV files from the configured directory.

        It parses the 'DATE' column into a DatetimeIndex, removes duplicate timestamps,
        and sorts the data chronologically.

        Returns:
            pd.DataFrame: A raw dataframe containing all columns from the CSVs, 
                          indexed by timestamp.

        Raises:
            FileNotFoundError: If the weather directory does not exist or is empty.
            KeyError: If the required 'DATE' column is missing from the data.
            RuntimeError: If parsing results in an empty dataframe.
        """
        if not os.path.exists(self.weather_dir):
            raise FileNotFoundError(f"Weather directory not found: {self.weather_dir}")

        csv_files = sorted(glob.glob(os.path.join(self.weather_dir, "*.csv")))
        if not csv_files:
            raise FileNotFoundError("No CSV files found in weather directory.")

        frames = []
        for f in csv_files:
            try:
                df = pd.read_csv(f, dtype=str, low_memory=False)
                frames.append(df)
            except Exception as e:
                self.log.error(f"Failed to load {f}: {e}")

        if not frames:
            raise RuntimeError("No CSV files loaded successfully.")

        df = pd.concat(frames, ignore_index=True)
        if "DATE" not in df.columns:
            raise KeyError("DATE column missing from NOAA dataset.")

        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
        df = df.dropna(subset=["DATE"]).sort_values("DATE").drop_duplicates("DATE")
        df = df.set_index("DATE")
        df.index.name = "timestamp"

        if df.empty:
            raise RuntimeError("No valid data after DATE parsing.")

        return df

    def select_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filters the dataframe to retain only relevant FM-12 weather fields.

        Args:
            df (pd.DataFrame): The raw dataframe with all NOAA columns.

        Returns:
            pd.DataFrame: A subset of the dataframe containing only `FM12_FIELDS`.

        Raises:
            ValueError: If none of the required FM-12 columns are present.
        """
        existing = [c for c in self.FM12_FIELDS if c in df.columns]
        if not existing:
            raise ValueError("No FM-12 weather columns found.")

        return df[existing].copy()

    def _split(self, series: pd.Series, idx: int, missing) -> pd.Series:
        """
        Helper method to parse comma-separated FM-12 strings.

        NOAA data often packs multiple values into one string (e.g., "330,1,N,0021,1").
        This method extracts a specific component by index.

        Args:
            series (pd.Series): The column of FM-12 strings.
            idx (int): The zero-based index of the component to extract.
            missing (str | list): Value(s) representing missing data to be replaced with NaN.

        Returns:
            pd.Series: A series of extracted values (as strings) with NaNs handled.
        """
        try:
            parts = series.str.split(",", expand=True)
            col = parts[idx]
            if isinstance(missing, list):
                col = col.replace(missing, np.nan)
            else:
                col = col.replace(missing, np.nan)
            return col
        except Exception as e:
            self.log.warning(f"FM-12 decode error: {e}")
            return pd.Series([np.nan] * len(series), index=series.index)

    def decode(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Decodes FM-12 encoded columns into human-readable meteorological features.

        Performs extraction, scaling (e.g., dividing by 10 for temperature), and 
        calculation of derived features (e.g., Relative Humidity via Magnus formula).

        Features generated:
        - Temperature/Dewpoint (C), Relative Humidity (%)
        - Wind Direction (deg), Speed (m/s)
        - Visibility (m), Pressure (hPa), Precipitation (mm)
        - Cloud Cover (oktas & %), Ceiling Height (m)

        Args:
            df (pd.DataFrame): The dataframe with raw FM-12 string columns.

        Returns:
            pd.DataFrame: A copy of the input dataframe with new decoded feature columns.
        """
        out = df.copy()

        # Temperature (°C, /10 from encoded)
        if "TMP" in df:
            out["temperature_C"] = self._split(df["TMP"], 0, "99999").astype(float) / 10.0

        # Dewpoint (°C, /10 from encoded)
        if "DEW" in df:
            out["dewpoint_C"] = self._split(df["DEW"], 0, "99999").astype(float) / 10.0

        # Relative humidity (Magnus formula)
        T, Td = out.get("temperature_C"), out.get("dewpoint_C")
        if T is not None and Td is not None:
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                rh = 100.0 * np.exp((17.625 * Td) / (243.04 + Td)) / np.exp((17.625 * T) / (243.04 + T))
                out["rel_humidity"] = rh.clip(0, 100)

        # Wind (dir index 0, speed index 3 /10)
        if "WND" in df:
            try:
                parts = df["WND"].str.split(",", expand=True, n=4)
                out["wind_dir_deg"] = pd.to_numeric(parts[0].replace("999", np.nan), errors='coerce')
                out["wind_speed_ms"] = pd.to_numeric(parts[3].replace("9999", np.nan), errors='coerce') / 10.0
            except Exception:
                out["wind_dir_deg"] = np.nan
                out["wind_speed_ms"] = np.nan

        # Visibility (meters)
        if "VIS" in df:
            out["visibility_m"] = self._split(df["VIS"], 0, ["999999", "99999"]).astype(float)

        # Pressure (hPa, /10 from encoded)
        if "SLP" in df:
            out["pressure_hPa"] = self._split(df["SLP"], 0, "99999").astype(float) / 10.0

        # Precipitation (mm, AA1 index 1)
        if "AA1" in df:
            out["precip_mm"] = self._split(df["AA1"], 1, "9999").astype(float)

        # Ceiling (meters * 10)
        if "CIG" in df:
            cig = self._split(df["CIG"], 0, "99999").astype(float)
            out["ceiling_m"] = cig * 10.0
            out["low_ceiling_flag"] = (out["ceiling_m"] < 200).astype(float)

        # Cloud cover (oktas → %)
        if "MA1" in df:
            okta = self._split(df["MA1"], 0, "99").astype(float)
            out["cloud_oktas"] = okta
            out["cloud_pct"] = (okta / 8.0) * 100.0

        # Obscuration (fog/haze codes)
        if "OD1" in df:
            code = self._split(df["OD1"], 0, "99").astype(float)
            out["obscuration_code"] = code
            mapping = {1: "fog", 2: "mist", 3: "smoke", 4: "haze", 5: "dust", 6: "sand", 7: "spray", 9: "other"}
            out["obscuration_type"] = out["obscuration_code"].map(mapping)
            out["fog_flag"] = (code == 1).astype(float)

        # Snow depth (mm)
        if "MD1" in df:
            snow = self._split(df["MD1"], 0, ["999", "99"]).astype(float)
            out["snow_depth_mm"] = snow
            out["snow_flag"] = (snow > 0).astype(float)

        return out

    def clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies meteorological consistency checks and handles outliers.

        Enforces physical laws and cleans data anomalies:
        - **Precipitation:** Clipped to [0, 50mm].
        - **Ceiling:** Infers values based on Fog (50m) or Clear Sky (20km).
        - **Thermodynamics:** Ensures Dewpoint <= Temperature.
        - **Outliers:** Applies IQR clipping for extreme UK temperatures.
        - **Filling:** Uses forward/backward fill to handle missing steps.

        Args:
            df (pd.DataFrame): The dataframe containing decoded features.

        Returns:
            pd.DataFrame: A cleaned dataframe with no NaNs and physically consistent values.
        """
        df = df.copy()

        # Precipitation (non-negative, realistic max)
        df["precip_mm"] = df.get("precip_mm", 0).fillna(0).clip(0, 50)
        df["rain_flag"] = (df["precip_mm"] > 0).astype(float)

        # Ceiling height (meteorological logic)
        df["ceiling_m"] = df.get("ceiling_m", np.nan).astype(float)
        df.loc[df["ceiling_m"] > 20000, "ceiling_m"] = 20000  # Unlimited → 20km
        
        # Fog implies low ceiling, clear sky implies high ceiling
        fog_low = df["ceiling_m"].isna() & (df.get("fog_flag", 0) == 1)
        df.loc[fog_low, "ceiling_m"] = 50.0
        clear_sky = df["ceiling_m"].isna() & (df.get("cloud_oktas", np.nan) == 0)
        df.loc[clear_sky, "ceiling_m"] = 20000.0
        
        df["ceiling_m"] = df["ceiling_m"].ffill().bfill()
        df["ceiling_category"] = pd.cut(df["ceiling_m"], bins=[-1, 200, 500, 1000, 25000],
                                       labels=["LIFR", "IFR", "MVFR", "VFR"]).fillna("VFR")
        df["low_ceiling_flag"] = (df["ceiling_m"] < 200).astype(float)

        # Temperature/Dewpoint physics (Td ≤ T)
        df["temperature_C"] = df.get("temperature_C", np.nan).astype(float)
        df["dewpoint_C"] = df.get("dewpoint_C", np.nan).astype(float)
        missing_T = df["temperature_C"].isna() & df["dewpoint_C"].notna()
        df.loc[missing_T, "temperature_C"] = df.loc[missing_T, "dewpoint_C"] + 2.0
        missing_Td = df["dewpoint_C"].isna() & df["temperature_C"].notna()
        df.loc[missing_Td, "dewpoint_C"] = df.loc[missing_Td, "temperature_C"] - 2.0

        # Physical limits
        df["wind_speed_ms"] = df.get("wind_speed_ms", np.nan).astype(float).clip(0, 100)
        df["wind_dir_deg"] = df.get("wind_dir_deg", np.nan).astype(float).clip(0, 360)
        df["visibility_m"] = df.get("visibility_m", np.nan).astype(float).clip(0, 100000)
        df["pressure_hPa"] = df.get("pressure_hPa", np.nan).astype(float).clip(800, 1100)
        
        # UK-specific temperature limits + outlier clipping
        def iqr_clip(series, k=3.0):
            if series.dropna().empty:
                return series
            q1, q3 = series.quantile([0.05, 0.95])
            iqr = q3 - q1
            return series.clip(q1 - k * iqr, q3 + k * iqr)

        df["temperature_C"] = iqr_clip(df["temperature_C"]).clip(-30, 40)
        df["dewpoint_C"] = iqr_clip(df["dewpoint_C"]).clip(-30, 40)
        df.loc[df["dewpoint_C"] > df["temperature_C"], "dewpoint_C"] = df["temperature_C"]

        # Cloud cover consistency
        df["cloud_oktas"] = df.get("cloud_oktas", np.nan).clip(0, 8)
        df["cloud_pct"] = (df["cloud_oktas"] / 8.0) * 100.0

        # Recalculate RH and finalize flags
        with np.errstate(all='ignore'):
            T, Td = df["temperature_C"], df["dewpoint_C"]
            rh = 100.0 * np.exp((17.625 * Td) / (243.04 + Td)) / np.exp((17.625 * T) / (243.04 + T))
            df["rel_humidity"] = rh.clip(0, 100)

        for col in ["fog_flag", "snow_flag", "low_ceiling_flag", "rain_flag"]:
            df[col] = df.get(col, 0).fillna(0).astype(float)

        df["obscuration_type"] = df.get("obscuration_type", "none").fillna("none")
        df["snow_depth_mm"] = df.get("snow_depth_mm", 0).clip(lower=0)

        return df.ffill().bfill()

    def resample(self, df: pd.DataFrame, freq: str = "15min") -> pd.DataFrame:
        """
        Resamples the weather dataset to a regular time grid (default 15 mins).

        Strategy:
        - **Continuous (Temp, Wind):** Interpolated over time.
        - **Accumulation (Precipitation):** Forward-filled, then gaps assumed 0.
        - **Categorical (Flags):** Forward-filled (persistence model).
        - **Derived (Humidity):** Re-calculated after interpolation to maintain physics.

        Args:
            df (pd.DataFrame): Cleaned weather data with a DatetimeIndex.
            freq (str, optional): Target frequency string. Defaults to "15min".

        Returns:
            pd.DataFrame: The resampled dataframe aligned to the target grid.
        """
        if df.empty:
            raise ValueError("Cannot resample empty dataframe.")

        df15 = df.resample(freq).asfreq()

        continuous = [
            "temperature_C", "dewpoint_C", "rel_humidity", "wind_dir_deg",
            "wind_speed_ms", "visibility_m", "pressure_hPa", "ceiling_m",
            "cloud_oktas", "cloud_pct", "snow_depth_mm"
        ]

        for col in continuous:
            if col in df15:
                df15[col] = pd.to_numeric(df15[col], errors="coerce").interpolate(
                    method="time", limit_direction="both"
                )

        if "precip_mm" in df15:
            df15["precip_mm"] = pd.to_numeric(df15["precip_mm"], errors="coerce").ffill().fillna(0)

        for col in ["low_ceiling_flag", "fog_flag", "snow_flag", "rain_flag",
                    "obscuration_type", "ceiling_category"]:
            if col in df15:
                df15[col] = df15[col].ffill()

        if "cloud_oktas" in df15:
            df15["cloud_oktas"] = df15["cloud_oktas"].round().clip(0, 8)
            df15["cloud_pct"] = (df15["cloud_oktas"] / 8.0) * 100.0

        if all(col in df15 for col in ["temperature_C", "dewpoint_C"]):
            with np.errstate(all="ignore"):
                T, Td = df15["temperature_C"], df15["dewpoint_C"]
                rh = 100.0 * np.exp((17.625 * Td) / (243.04 + Td)) / np.exp((17.625 * T) / (243.04 + T))
                df15["rel_humidity"] = rh.clip(0, 100)

        return df15

    def drop_raw_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Removes the original FM-12 string columns from the dataframe.

        Args:
            df (pd.DataFrame): The dataframe containing both raw and processed columns.

        Returns:
            pd.DataFrame: Dataframe with only the cleaned, numeric feature columns.
        """
        raw_cols = [c for c in self.FM12_FIELDS if c in df.columns]
        return df.drop(columns=raw_cols)

    def run(self, freq: str = '15min', save_path: str | None = None) -> pd.DataFrame:
        """
        Executes the full weather preprocessing pipeline.

        Steps: Load -> Select FM-12 -> Decode -> Clean -> Drop Raw -> Resample.

        Args:
            freq (str, optional): Target frequency for resampling. Defaults to '15min'.
            save_path (str | None, optional): If provided, saves the result to this CSV path.

        Returns:
            pd.DataFrame: The final processed weather dataset.
        """
        df = self.load_raw()
        df = self.select_columns(df)
        df = self.decode(df)
        df = self.clean_features(df)
        df = self.drop_raw_columns(df)
        df = self.resample(df, freq)

        if save_path:
            df.to_csv(save_path)
            self.log.info(f"Weather data saved: {save_path}")
        return df


class TrafficPreprocessor:
    """
    Handles loading, standardizing, and aggregating traffic sensor data.

    This class is responsible for:
    1.  Loading raw traffic CSVs.
    2.  Standardizing column names (e.g., 'Total Volume' -> 'total_volume').
    3.  Creating a unified DatetimeIndex.
    4.  Aggregating data from multiple sensors (handling outages via mean aggregation).

    Attributes:
        data_dir (str): Directory containing traffic data CSVs.
    """

    def __init__(self, data_dir: str = "data/traffic"):
        """
        Args:
            data_dir (str, optional): Path to traffic data. Defaults to "data/traffic".
        """
        self.data_dir = data_dir

    # 1) LOADING ONLY
    def load_raw(self) -> pd.DataFrame | None:
        """
        Loads and concatenates raw traffic CSV files.

        Returns:
            pd.DataFrame | None: Concatenated raw data, or None if no files found.
        """
        data_dir = self.data_dir

        if not os.path.exists(data_dir):
            print(f"❌ Error: Directory '{data_dir}' not found.")
            return None

        csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
        if not csv_files:
            print("❌ Error: No CSV files found.")
            return None

        print(f"📂 Found {len(csv_files)} traffic files. Loading...")

        df_list = []
        for f in csv_files:
            try:
                temp_df = pd.read_csv(f)
                cols_to_drop = [
                    c for c in temp_df.columns
                    if "timestamp" in c.lower() or "unnamed" in c.lower()
                ]
                if cols_to_drop:
                    temp_df.drop(columns=cols_to_drop, inplace=True)
                df_list.append(temp_df)
            except Exception as e:
                print(f"⚠️ Warning: Could not read {f}. Reason: {e}")

        if not df_list:
            return None

        return pd.concat(df_list, ignore_index=True)

    # 2) STANDARDISATION
    def standardise(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardises column names and constructs a clean DatetimeIndex.

        Parses 'Report Date' and 'Time Period Ending' into a single 'timestamp'.

        Args:
            raw_df (pd.DataFrame): The raw loaded dataframe.

        Returns:
            pd.DataFrame: Dataframe with 'timestamp' index and standard column names.
        """
        df = raw_df.copy()

        # Parse timestamps
        df["timestamp"] = pd.to_datetime(
            df["Report Date"] + " " + df["Time Period Ending"],
            dayfirst=True,
            errors="coerce",
        )
        df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()

        # Standardise column names
        if "Total Volume" in df.columns:
            df = df.rename(columns={"Total Volume": "total_volume"})
        if "Avg mph" in df.columns:
            df = df.rename(columns={"Avg mph": "avg_mph"})

        return df

    # 3) TREATMENT / CLEANING
    def process(self, std_df: pd.DataFrame, freq: str = "15min") -> pd.DataFrame | None:
        """
        Aggregates sensor data and resamples to a regular grid.

        - **Aggregation:** Uses MEAN to combine multiple sensors (robust to single sensor outages).
        - **Resampling:** Defaults to MEAN for 15-minute intervals.

        Args:
            std_df (pd.DataFrame): Standardized dataframe.
            freq (str, optional): Target frequency. Defaults to "15min".

        Returns:
            pd.DataFrame | None: Final processed traffic data with index reset.
        """
        try:
            df = std_df.copy()

            print("   > Aggregating multiple sensors (Using Mean to handle outages)...")
            grouped_df = df.groupby(df.index).agg({
                "total_volume": "mean" if freq == "15min" else "sum",
                "avg_mph": "mean",
            })

            final_df = grouped_df.resample(freq).mean()

            print(f"✅ Traffic Data Processed. Rows: {len(final_df)}")
            return final_df.reset_index()

        except Exception as e:
            print(f"❌ Critical Error during processing: {e}")
            return None

    # 4) CONVENIENCE WRAPPER
    def load_traffic_data(self, freq: str = "15min") -> pd.DataFrame | None:
        """
        Executes the full traffic pipeline: Load -> Standardise -> Process.

        Args:
            freq (str, optional): Target frequency. Defaults to "15min".

        Returns:
            pd.DataFrame | None: The final processed traffic dataset.
        """
        raw_df = self.load_raw()
        if raw_df is None:
            return None
        std_df = self.standardise(raw_df)
        return self.process(std_df)


def process_and_merge(traffic_df: pd.DataFrame, weather_df: pd.DataFrame, freq: str = "15min") -> pd.DataFrame:
    """
    Merges traffic and weather data onto a unified time grid.

    Strategy:
    1.  **Traffic (Target):** Resampled via `asfreq()`. Gaps remain NaN (no artificial targets).
    2.  **Weather (Features):** Resampled and Forward-Filled (atmosphere evolves continuously).
    3.  **Merge:** Left Join onto Traffic timestamps.
    4.  **Leakage Prevention:** Weather is forward-filled *after* join with a limit of 2 steps.

    Args:
        traffic_df (pd.DataFrame): Processed traffic data.
        weather_df (pd.DataFrame): Processed weather data.
        freq (str, optional): Target frequency. Defaults to "15min".

    Returns:
        pd.DataFrame: The merged dataset ready for model training.

    Raises:
        ValueError: If inputs are empty.
        RuntimeError: If merging fails.
    """
    if traffic_df is None or traffic_df.empty:
        raise ValueError("traffic_df is empty.")
    if weather_df is None or weather_df.empty:
        raise ValueError("weather_df is empty.")

    # 1. Prepare Traffic (Target)
    try:
        traffic_df = traffic_df.copy()
        if "timestamp" in traffic_df.columns:
            traffic_df["timestamp"] = pd.to_datetime(
                traffic_df["timestamp"], errors="coerce"
            )
            traffic_df = traffic_df.dropna(subset=["timestamp"]).set_index("timestamp")

        traffic_df = traffic_df.sort_index()
        traffic_df = traffic_df.resample(freq).asfreq()
    except Exception as e:
        raise RuntimeError(f"Traffic timestamp processing failed: {e}")

    # 2. Prepare Weather (Features)
    try:
        weather_df = weather_df.copy()
        weather_df.index = pd.to_datetime(weather_df.index, errors="coerce")
        weather_df = weather_df.sort_index()
        weather_df = weather_df.resample(freq).ffill()
    except Exception as e:
        raise RuntimeError(f"Weather timestamp processing failed: {e}")

    # 3. Merge
    try:
        merged = traffic_df.join(weather_df, how="left")

        weather_cols = weather_df.columns
        merged[weather_cols] = merged[weather_cols].ffill(limit=2)

        if "total_volume" in merged.columns:
            merged["total_volume"] = merged["total_volume"].clip(lower=0)
    except Exception as e:
        raise RuntimeError(f"Failed to merge weather + traffic: {e}")

    return merged