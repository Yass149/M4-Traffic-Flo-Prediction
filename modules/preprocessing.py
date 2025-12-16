import os
import glob
import pandas as pd
import numpy as np
import logging

class WeatherPreprocessor:
    """
    End-to-end NOAA ISD FM-12 Weather Preprocessor for traffic modeling.

    Transforms raw NOAA ISD CSV files into clean, 15-minute interval datasets
    with decoded meteorological features, physical consistency checks,
    outlier handling, and time-series ready resampling.
    """

    FM12_FIELDS = ["WND", "VIS", "TMP", "DEW", "SLP", "AA1", "CIG", "MA1", "OD1", "MD1"]

    def __init__(self, weather_dir: str = "data/weather", verbose: bool = False):
        """
        Initialize the weather preprocessor.

        Parameters
        ----------
        weather_dir : str
            Directory containing NOAA ISD CSV files (default: "data/weather").
        verbose : bool
            Enable detailed logging if True (default: False).
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
        Load and merge all NOAA weather CSV files from the weather directory.

        Combines multiple CSV files, parses DATE column to datetime index,
        removes duplicates, and sorts chronologically.

        Returns
        -------
        pd.DataFrame
            Raw weather data indexed by timestamp.

        Raises
        ------
        FileNotFoundError
            If weather directory or CSV files are missing.
        KeyError
            If DATE column is absent.
        RuntimeError
            If no valid data after parsing.
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
        Filter dataframe to FM-12 weather fields only.

        Parameters
        ----------
        df : pd.DataFrame
            Raw NOAA dataframe with all columns.

        Returns
        -------
        pd.DataFrame
            Dataframe containing only available FM-12 fields.

        Raises
        ------
        ValueError
            If no FM-12 columns are present.
        """
        existing = [c for c in self.FM12_FIELDS if c in df.columns]
        if not existing:
            raise ValueError("No FM-12 weather columns found.")

        return df[existing].copy()

    def _split(self, series: pd.Series, idx: int, missing) -> pd.Series:
        """
        Safely extract component from comma-separated FM-12 strings.

        Parameters
        ----------
        series : pd.Series
            FM-12 encoded string column (e.g., "330,1,N,0021,1").
        idx : int
            Zero-based index of component to extract.
        missing : str or list of str
            Value(s) to replace with NaN.

        Returns
        -------
        pd.Series
            Extracted values with missing data handled.
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
        Decode FM-12 encoded columns into meteorological features.

        Extracts and converts:
        - temperature_C, dewpoint_C, rel_humidity (%)
        - wind_dir_deg, wind_speed_ms
        - visibility_m, pressure_hPa, precip_mm
        - ceiling_m, low_ceiling_flag
        - cloud_oktas, cloud_pct (%)
        - obscuration_code, obscuration_type, fog_flag
        - snow_depth_mm, snow_flag

        Parameters
        ----------
        df : pd.DataFrame
            Raw FM-12 encoded dataframe.

        Returns
        -------
        pd.DataFrame
            Decoded features added to copy of input.
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
        Apply meteorological cleaning, consistency checks, and outlier clipping.

        Key operations:
        - Precipitation: clip [0,50mm], add rain_flag
        - Ceiling: fog→50m, clear→20km, ffill/bfill, aviation categories
        - Temperature/Dewpoint: T≥Td physics, UK clip [-30,40°C]
        - Outliers: IQR clipping + hard physical limits
        - Final forward/backward fill for completeness

        Parameters
        ----------
        df : pd.DataFrame
            Decoded weather features.

        Returns
        -------
        pd.DataFrame
            Cleaned features with zero NaNs and physical consistency.
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
        Resample the weather dataset to a regular time grid.

        Behaviour
        ---------
        - Continuous variables:
            Time-based interpolation on the new grid.
        - Precipitation:
            Forward-fill then fill remaining gaps with 0 (step-wise).
        - Flags & categoricals:
            Forward-fill state (no interpolation).
        - Relative humidity:
            Recomputed from temperature and dewpoint after interpolation.

        Parameters
        ----------
        df : pd.DataFrame
            Cleaned weather data with a DatetimeIndex.
        freq : str, optional
            Target resampling frequency (e.g. "15min", "1H").

        Returns
        -------
        pd.DataFrame
            Weather data resampled to the specified frequency.
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
        Remove original FM-12 string columns after successful decoding.

        Parameters
        ----------
        df : pd.DataFrame
            Decoded dataframe with raw + processed columns.

        Returns
        -------
        pd.DataFrame
            Cleaned dataframe without raw FM-12 strings.
        """
        raw_cols = [c for c in self.FM12_FIELDS if c in df.columns]
        return df.drop(columns=raw_cols)

    def run(self,freq:str='15min',save_path: str | None = None) -> pd.DataFrame:
        """
        Execute complete preprocessing pipeline.

        Pipeline: load → select FM-12 → decode → clean → drop raw → resample 15min.

        Parameters
        ----------
        save_path : str, optional
            Path to save cleaned CSV (default: None).

        Returns
        -------
        pd.DataFrame
            Final cleaned, resampled 15-minute weather dataset with 19 features.
        """
        df = self.load_raw()
        df = self.select_columns(df)
        df = self.decode(df)
        df = self.clean_features(df)
        df = self.drop_raw_columns(df)
        df = self.resample(df,freq)

        if save_path:
            df.to_csv(save_path)
            self.log.info(f"Weather data saved: {save_path}")
        return df
class TrafficPreprocessor:
    """
    Load and preprocess traffic data.

    1) Raw loading/concatenation
    2) Standardisation (names + timestamp index)
    3) Cleaning, aggregation, and resampling
    """

    def __init__(self, data_dir: str = "data/traffic"):
        self.data_dir = data_dir

    # 1) LOADING ONLY
    def load_raw(self) -> pd.DataFrame | None:
        """Load and concatenate raw traffic CSV files without aggregation."""
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
        Standardise column names and create a datetime index.

        - Build `timestamp` from `Report Date` + `Time Period Ending`
        - Set sorted DatetimeIndex
        - Rename `Total Volume` → `total_volume`, `Avg mph` → `avg_mph`
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
    def process(self, std_df: pd.DataFrame, freq: str="15min") -> pd.DataFrame | None:
        """
        Aggregate sensors and resample to a 15‑minute grid.

        - Aggregate duplicate timestamps across sensors using MEAN
        - Resample to regular 15‑minute intervals with MEAN
        - Reset index before returning
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
    def load_traffic_data(self,freq: str="15min") -> pd.DataFrame | None:
        """
        Full pipeline: load → standardise → process.
        """
        raw_df = self.load_raw()
        if raw_df is None:
            return None
        std_df = self.standardise(raw_df)
        return self.process(std_df)

def process_and_merge(traffic_df: pd.DataFrame, weather_df: pd.DataFrame, freq: str = "15min") -> pd.DataFrame:
    """
    Merge regularly sampled traffic data with weather features on a common time grid.

    Behaviour
    ---------
    - Traffic (target):
        Parsed to a DatetimeIndex, sorted, and resampled to the requested
        frequency using `asfreq()` so gaps remain as NaN (no forward-fill).
    - Weather (features):
        Parsed to a DatetimeIndex, sorted, resampled to the same frequency,
        and forward-filled (atmosphere evolves smoothly).
    - Merge:
        Left join weather onto traffic timestamps only.
        Forward-fill *weather columns only* (limit=2) to avoid target leakage.
        Clip any negative `total_volume` values to zero.

    Parameters
    ----------
    traffic_df : pd.DataFrame
        Cleaned traffic data with `timestamp` and `total_volume` (and optionally `avg_mph`).
    weather_df : pd.DataFrame
        Processed weather data indexed by datetime.
    freq : str, optional
        Target resampling frequency (e.g. "15min", "1H").

    Returns
    -------
    pd.DataFrame
        Merged traffic–weather dataframe on the specified grid.
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
