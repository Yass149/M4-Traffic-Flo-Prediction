import os
import glob
import pandas as pd
import numpy as np
import logging

class WeatherPreprocessor:
    """
    End-to-end NOAA ISD FM-12 Weather Preprocessor.

    This class provides a pipeline for transforming raw NOAA ISD weather
    CSV files into a clean, analysis-ready dataset suitable for regression
    and time-series forecasting in traffic modelling contexts.

    Pipeline steps
    --------------
    1) File discovery and safe CSV loading with error handling
    2) Optional station filter (by STATION column)
    3) Selection of relevant FM-12 encoded weather fields
    4) Decoding of FM-12 groups into numerical values:
         - Temperature, dewpoint, relative humidity
         - Wind speed & direction
         - Visibility
         - Sea-level pressure
         - Rainfall (AA1)
         - Ceiling height
         - Cloud cover (oktas + %)
         - Fog / haze / obscuration
         - Snow depth
    5) Meteorologically-informed cleaning and outlier handling:
         - Physical consistency checks
         - T / Td relationships
         - Fog & ceiling interactions
         - Snow & precipitation logic
    6) Resampling to 15-minute intervals with:
         - Time-based interpolation for continuous variables
         - Forward-fill for flags & categorical states
         - No interpolation of precipitation
    7) Optional saving of the cleaned dataset to CSV.

    Original raw FM-12 string columns are preserved until explicitly
    removed via `drop_raw_columns()`; decoded features never overwrite
    raw data.
    """

    FM12_FIELDS = ["WND", "VIS", "TMP", "DEW", "SLP", "AA1",
                   "CIG", "MA1", "OD1", "MD1"]

    def __init__(self,
                 weather_dir: str = "data/weather",
                 verbose: bool = False):
        """
        Parameters
        ----------
        weather_dir : str
            Directory containing NOAA ISD CSV files.
        verbose : bool
            If True, enables detailed logging for debugging.
        """

        self.weather_dir = weather_dir

        # Logging setup
        logging.basicConfig(
            level=logging.INFO if verbose else logging.WARNING,
            format="%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%H:%M:%S"
        )
        self.log = logging.getLogger("WeatherPreprocessor")

    # ------------------------------------------------------------------
    # STEP 1 — LOAD RAW NOAA CSV FILES
    # ------------------------------------------------------------------

    def load_raw(self) -> pd.DataFrame:
        """
        Load and merge all NOAA weather CSV files into a single dataframe.

        Returns
        -------
        pandas.DataFrame
            Raw NOAA weather data indexed by timestamp.

        Raises
        ------
        FileNotFoundError:
            If the directory or CSV files are missing.
        KeyError:
            If DATE column does not exist.
        RuntimeError:
            If resulting dataframe is empty after DATE parsing.
        """
        if not os.path.exists(self.weather_dir):
            raise FileNotFoundError(f"Weather folder does not exist: {self.weather_dir}")

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
            raise RuntimeError("No CSV files could be loaded successfully.")

        df = pd.concat(frames, ignore_index=True)

        if "DATE" not in df.columns:
            raise KeyError("DATE column missing in NOAA dataset.")


        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
        df = df.dropna(subset=["DATE"]).sort_values("DATE")
        df = df.drop_duplicates("DATE")

        df = df.set_index("DATE")
        df.index.name = "timestamp"

        if df.empty:
            raise RuntimeError("Loaded dataframe is empty after DATE parsing.")

        return df

    # ------------------------------------------------------------------
    # STEP 2 — SELECT RELEVANT FM-12 COLUMNS ONLY
    # ------------------------------------------------------------------

    def select_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Keep only relevant FM-12 encoded weather fields plus STATION if present.

        Parameters
        ----------
        df : pandas.DataFrame
            Raw NOAA dataframe containing many unused metadata fields.

        Returns
        -------
        pandas.DataFrame
            Dataframe filtered to FM-12 weather columns (and STATION if present).

        Raises
        ------
        ValueError:
            If none of the FM-12 fields are present.
        """
        existing = [c for c in self.FM12_FIELDS if c in df.columns]
        if not existing:
            raise ValueError("No FM-12 weather columns found in dataset.")

        keep_cols = existing.copy()

        return df[keep_cols].copy()

    # ------------------------------------------------------------------
    # Utility: safe splitting of encoded fields
    # ------------------------------------------------------------------

    def _split(self, series: pd.Series, idx: int, missing) -> pd.Series:
        """
        Safely splits FM-12 encoded strings like '330,1,N,0021,1'.

        Parameters
        ----------
        series : pandas.Series
            Encoded FM-12 string column.
        idx : int
            Index of the component to extract.
        missing : str or list
            Value(s) representing missing data.

        Returns
        -------
        pandas.Series
            Extracted raw values with missing values handled safely.
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

    # ------------------------------------------------------------------
    # STEP 3 — DECODE ALL FM-12 FIELDS
    # ------------------------------------------------------------------

    def decode(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Decode all raw FM-12 encoded weather columns.

        Adds numerical features including:
        - temperature_C, dewpoint_C, rel_humidity
        - wind_speed_ms, wind_dir_deg
        - visibility_m, pressure_hPa, precip_mm
        - ceiling_m, low_ceiling_flag
        - cloud_oktas, cloud_pct
        - obscuration_code, obscuration_type, fog_flag
        - snow_depth_mm, snow_flag

        Returns
        -------
        pandas.DataFrame
        """
        out = df.copy()

        # ---------------- Temperature ----------------
        if "TMP" in df.columns:
            try:
                out["temperature_C"] = (
                    self._split(df["TMP"], 0, "99999").astype(float) / 10.0
                )
            except Exception:
                out["temperature_C"] = np.nan

        # ---------------- Dewpoint ----------------
        if "DEW" in df.columns:
            try:
                out["dewpoint_C"] = (
                    self._split(df["DEW"], 0, "99999").astype(float) / 10.0
                )
            except Exception:
                out["dewpoint_C"] = np.nan

        # ---------------- Relative humidity (initial) ----------------
        try:
            T = out.get("temperature_C")
            Td = out.get("dewpoint_C")
            if T is not None and Td is not None:
                with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                    rh = (100.0 *
                          np.exp((17.625 * Td) / (243.04 + Td)) /
                          np.exp((17.625 * T) / (243.04 + T)))
                out["rel_humidity"] = rh.clip(0, 100)
        except Exception as e:
            self.log.warning(f"Initial RH calculation failed: {e}")
            out["rel_humidity"] = np.nan

        # ---------------- Wind ----------------
        if "WND" in df.columns:
            try:
                parts = df["WND"].str.split(",", expand=True)
                out["wind_dir_deg"] = parts[0].replace("999", np.nan).astype(float)
                out["wind_speed_ms"] = parts[3].replace("9999", np.nan).astype(float) / 10.0
            except Exception:
                out["wind_dir_deg"] = np.nan
                out["wind_speed_ms"] = np.nan

        # ---------------- Visibility ----------------
        if "VIS" in df.columns:
            try:
                out["visibility_m"] = (
                    self._split(df["VIS"], 0, ["999999", "99999"]).astype(float)
                )
            except Exception:
                out["visibility_m"] = np.nan

        # ---------------- Sea-level Pressure ----------------
        if "SLP" in df.columns:
            try:
                out["pressure_hPa"] = (
                    self._split(df["SLP"], 0, "99999").astype(float) / 10.0
                )
            except Exception:
                out["pressure_hPa"] = np.nan

        # ---------------- Precipitation (AA1) ----------------
        if "AA1" in df.columns:
            try:
                # AA1,1,mm,period,quality   → we usually want the amount (index 1)
                out["precip_mm"] = (
                    self._split(df["AA1"], 1, "9999").astype(float)
                )
            except Exception:
                out["precip_mm"] = np.nan

        # ---------------- Ceiling height ----------------
        if "CIG" in df.columns:
            try:
                cig = self._split(df["CIG"], 0, "99999").astype(float)
                out["ceiling_m"] = cig * 10.0
                out["low_ceiling_flag"] = (out["ceiling_m"] < 200).astype(float)
            except Exception:
                out["ceiling_m"] = np.nan
                out["low_ceiling_flag"] = np.nan

        # ---------------- Cloud amount (MA1) ----------------
        if "MA1" in df.columns:
            try:
                okta = self._split(df["MA1"], 0, "99").astype(float)
                out["cloud_oktas"] = okta
                out["cloud_pct"] = (okta / 8.0) * 100.0
            except Exception:
                out["cloud_oktas"] = np.nan
                out["cloud_pct"] = np.nan

        # ---------------- Obscuration (fog / haze) (OD1) ----------------
        if "OD1" in df.columns:
            try:
                code = self._split(df["OD1"], 0, "99").astype(float)
                out["obscuration_code"] = code
                mapping = {
                    1: "fog",
                    2: "mist",
                    3: "smoke",
                    4: "haze",
                    5: "dust",
                    6: "sand",
                    7: "spray",
                    9: "other",
                }
                out["obscuration_type"] = out["obscuration_code"].map(mapping)
                out["fog_flag"] = (out["obscuration_code"] == 1).astype(float)
            except Exception:
                out["obscuration_code"] = np.nan
                out["obscuration_type"] = np.nan
                out["fog_flag"] = np.nan

        # ---------------- Snow depth (MD1) ----------------
        if "MD1" in df.columns:
            try:
                snow = self._split(df["MD1"], 0, ["999", "99"]).astype(float)
                out["snow_depth_mm"] = snow
                out["snow_flag"] = (snow > 0).astype(float)
            except Exception:
                out["snow_depth_mm"] = np.nan
                out["snow_flag"] = np.nan

        return out

    # METEOROLOGICAL CLEANING & OUTLIER HANDLING

    def clean_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply meteorological cleaning and outlier handling
        to decoded FM-12 weather features.

        This method implements best-practice rules for handling:
        - Missing values
        - Physically impossible values
        - Precipitation interpretation
        - Fog and ceiling interactions
        - Snow depth
        - Cloudiness
        - Visibility and pressure limits

        Returns
        -------
        pandas.DataFrame
            Cleaned, physically-consistent weather dataset.
        """
        df = df.copy()

        # ---------------- 1. Obscuration handling ----------------
        df["obscuration_type"] = df.get("obscuration_type", np.nan).fillna("none")
        df["obscuration_code"] = df.get("obscuration_code", 0).fillna(0).astype(float)
        df["fog_flag"] = df.get("fog_flag", 0).fillna(0).astype(float)

        # ---------------- 2. Precipitation handling ----------------
        # Missing AA1 is usually interpreted as "no recent precipitation"
        df["precip_mm"] = df.get("precip_mm", 0).fillna(0).astype(float)
        df["precip_mm"] = df["precip_mm"].clip(lower=0)
        df["rain_flag"] = (df["precip_mm"] > 0).astype(float)

        # ---------------- 3. Ceiling height logic ----------------
        df["ceiling_m"] = df.get("ceiling_m", np.nan).astype(float)

        # If ceiling missing but fog present → very low ceiling (e.g., dense fog)
        fog_low = df["ceiling_m"].isna() & (df["fog_flag"] == 1)
        df.loc[fog_low, "ceiling_m"] = 50.0  # metres

        # If ceiling missing and sky clear → high ceiling (essentially unlimited)
        df["cloud_oktas"] = df.get("cloud_oktas", np.nan).astype(float)
        clear_sky = df["ceiling_m"].isna() & (df["cloud_oktas"] == 0)
        df.loc[clear_sky, "ceiling_m"] = 5000.0

        # Remaining missing ceilings → forward-fill (slow atmospheric drift)
        df["ceiling_m"] = df["ceiling_m"].ffill()

        # Ceiling category (aviation standard)
        df["ceiling_category"] = pd.cut(
            df["ceiling_m"],
            bins=[-1, 200, 500, 1000, 20000],
            labels=["LIFR", "IFR", "MVFR", "VFR"]
        )

        # Low ceiling flag consistency
        df["low_ceiling_flag"] = (df["ceiling_m"] < 200).astype(float)

        # ---------------- 4. Snow handling ----------------
        df["snow_depth_mm"] = df.get("snow_depth_mm", np.nan).astype(float)
        df["snow_depth_mm"] = df["snow_depth_mm"].clip(lower=0)
        df["snow_flag"] = df.get("snow_flag", 0).fillna(0).astype(float)

        # ---------------- 5. T / Td physical consistency ----------------
        df["temperature_C"] = df.get("temperature_C", np.nan).astype(float)
        df["dewpoint_C"] = df.get("dewpoint_C", np.nan).astype(float)

        # If dewpoint exists but temperature missing → infer minimal spread (2°C)
        missing_T = df["temperature_C"].isna() & df["dewpoint_C"].notna()
        df.loc[missing_T, "temperature_C"] = df.loc[missing_T, "dewpoint_C"] + 2.0

        # If temperature exists but dewpoint missing → assume Td = T - 2°C
        missing_Td = df["dewpoint_C"].isna() & df["temperature_C"].notna()
        df.loc[missing_Td, "dewpoint_C"] = df.loc[missing_Td, "temperature_C"] - 2.0

        # Dewpoint cannot exceed temperature
        dewpoint_too_high = df["dewpoint_C"] > df["temperature_C"]
        df.loc[dewpoint_too_high, "dewpoint_C"] = df.loc[dewpoint_too_high, "temperature_C"]

        # ---------------- 6. Physical anomaly → NaN (not clip yet) ----------------
        # Temperature plausible range for UK climate
        df.loc[df["temperature_C"] < -40, "temperature_C"] = np.nan
        df.loc[df["temperature_C"] > 50, "temperature_C"] = np.nan

        df.loc[df["dewpoint_C"] < -50, "dewpoint_C"] = np.nan
        df.loc[df["dewpoint_C"] > 50, "dewpoint_C"] = np.nan

        df["wind_speed_ms"] = df.get("wind_speed_ms", np.nan).astype(float)
        df.loc[df["wind_speed_ms"] < 0, "wind_speed_ms"] = np.nan
        df.loc[df["wind_speed_ms"] > 80, "wind_speed_ms"] = np.nan  # extreme

        df["wind_dir_deg"] = df.get("wind_dir_deg", np.nan).astype(float)
        df.loc[df["wind_dir_deg"] < 0, "wind_dir_deg"] = np.nan
        df.loc[df["wind_dir_deg"] > 360, "wind_dir_deg"] = np.nan

        df["visibility_m"] = df.get("visibility_m", np.nan).astype(float)
        df.loc[df["visibility_m"] < 0, "visibility_m"] = np.nan
        df.loc[df["visibility_m"] > 100000, "visibility_m"] = np.nan

        df["pressure_hPa"] = df.get("pressure_hPa", np.nan).astype(float)
        df.loc[df["pressure_hPa"] < 870, "pressure_hPa"] = np.nan
        df.loc[df["pressure_hPa"] > 1080, "pressure_hPa"] = np.nan

        # ---------------- 7. Statistical outlier handling (IQR) ----------------
        def iqr_clip(series: pd.Series, k: float = 3.0) -> pd.Series:
            if series.dropna().empty:
                return series
            q1, q3 = series.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower = q1 - k * iqr
            upper = q3 + k * iqr
            return series.clip(lower, upper)

        df["temperature_C"] = iqr_clip(df["temperature_C"])
        df["dewpoint_C"] = iqr_clip(df["dewpoint_C"])
        df["pressure_hPa"] = iqr_clip(df["pressure_hPa"])
        df["visibility_m"] = iqr_clip(df["visibility_m"])
        df["wind_speed_ms"] = iqr_clip(df["wind_speed_ms"])

        # ---------------- 8. Cloudiness constraints ----------------
        df["cloud_oktas"] = df["cloud_oktas"].clip(lower=0, upper=8)
        df["cloud_pct"] = (df["cloud_oktas"] / 8.0) * 100.0

        # ---------------- 9. Recompute relative humidity ----------------
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            T = df["temperature_C"]
            Td = df["dewpoint_C"]
            rh = (
                100.0 *
                np.exp((17.625 * Td) / (243.04 + Td)) /
                np.exp((17.625 * T) / (243.04 + T))
            )
        df["rel_humidity"] = rh.clip(0, 100)

        # ---------------- 10. Final categorical/flag cleanup ----------------
        for col in ["fog_flag", "snow_flag", "low_ceiling_flag", "rain_flag"]:
            df[col] = df.get(col, 0).fillna(0).astype(float)

        return df

    #  RESAMPLE TO 15-MINUTE INTERVALS

    def resample(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Resample the weather dataset to 15-minute intervals.

        Continuous features use time-based interpolation.
        Categorical / binary features are forward-filled.
        Precipitation is NOT interpolated (step-wise behaviour).

        Returns
        -------
        pandas.DataFrame
        """
        if df.empty:
            raise ValueError("Cannot resample an empty dataframe.")

        # Create a regular 15-min grid
        df15 = df.resample("15min").asfreq()

        continuous = [
            "temperature_C", "dewpoint_C", "rel_humidity",
            "wind_dir_deg", "wind_speed_ms",
            "visibility_m", "pressure_hPa",
            "ceiling_m", "cloud_oktas", "cloud_pct",
            "snow_depth_mm"
        ]

        # Precip handled separately (no interpolation)
        precip_col = "precip_mm"

        flags = ["low_ceiling_flag", "fog_flag", "snow_flag", "rain_flag"]
        categorical = ["obscuration_type", "ceiling_category", "obscuration_code"]

        # ---- Continuous: true temporal interpolation ----
        for c in continuous:
            if c in df15.columns:
                df15[c] = pd.to_numeric(df15[c], errors="coerce")
                df15[c] = df15[c].interpolate(
                    method="time",
                    limit_direction="both"
                )

        # ---- Precipitation: step-wise / forward-fill, then fill NaN with 0 ----
        if precip_col in df15.columns:
            df15[precip_col] = pd.to_numeric(df15[precip_col], errors="coerce")
            df15[precip_col] = df15[precip_col].ffill().fillna(0.0)

        # ---- Flags: forward-fill ----
        for f in flags:
            if f in df15.columns:
                df15[f] = df15[f].ffill().fillna(0.0).astype(float)

        # ---- Categorical: forward-fill ----
        for cat in categorical:
            if cat in df15.columns:
                df15[cat] = df15[cat].ffill()

        # ---- Cloud okta integer & cloud percentage consistency ----
        if "cloud_oktas" in df15.columns:
            df15["cloud_oktas"] = df15["cloud_oktas"].round().clip(0, 8)
            df15["cloud_pct"] = (df15["cloud_oktas"] / 8.0) * 100.0

        # ---- Recompute RH after interpolation (final consistency) ----
        if "temperature_C" in df15.columns and "dewpoint_C" in df15.columns:
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                T = df15["temperature_C"]
                Td = df15["dewpoint_C"]
                rh = (
                    100.0 *
                    np.exp((17.625 * Td) / (243.04 + Td)) /
                    np.exp((17.625 * T) / (243.04 + T))
                )
            df15["rel_humidity"] = rh.clip(0, 100)

        return df15

    # ------------------------------------------------------------------
    # STEP 6 — DROP RAW FM-12 COLUMNS
    # ------------------------------------------------------------------

    def drop_raw_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove original FM-12 string fields after decoding.

        These columns are not usable for modelling and should be
        removed once the numerical decoded features are generated.
        """
        raw_cols = self.FM12_FIELDS
        df_clean = df.drop(columns=[c for c in raw_cols if c in df.columns])
        return df_clean

    # ------------------------------------------------------------------
    # STEP 7 — FINAL PIPELINE
    # ------------------------------------------------------------------

    def run(self, save_path: str | None = None) -> pd.DataFrame:
        """
        Run the full weather preprocessing pipeline.

        Parameters
        ----------
        save_path : str, optional
            If provided, the clean dataset will be exported to CSV.
Ò
        Returns
        -------
        pandas.DataFrame
            Fully decoded, cleaned, resampled weather dataset.
        """
        df = self.load_raw()
        df = self.select_columns(df)
        df = self.decode(df)
        df = self.clean_features(df)
        df = self.drop_raw_columns(df)
        df = self.resample(df)

        if save_path:
            try:
                df.to_csv(save_path)
                self.log.info(f"Saved cleaned weather file → {save_path}")
            except Exception as e:
                self.log.error(f"Failed to save CSV: {e}")

        return df
    

def process_and_merge(traffic_df: pd.DataFrame, weather_df: pd.DataFrame):
    """
    Merge 15-minute traffic data with processed 15-minute weather.
    
    FIX: Prevents 'Target Leakage' by ensuring we do not forward-fill 
    missing traffic data, only weather data.
    """
    if traffic_df is None or traffic_df.empty:
        raise ValueError("traffic_df is empty.")
    if weather_df is None or weather_df.empty:
        raise ValueError("weather_df is empty.")

    # 1. Prepare Traffic (Target)
    try:
        traffic_df = traffic_df.copy()
        if "timestamp" in traffic_df.columns:
            traffic_df["timestamp"] = pd.to_datetime(traffic_df["timestamp"], errors="coerce")
            traffic_df = traffic_df.dropna(subset=["timestamp"]).set_index("timestamp")
        
        traffic_df = traffic_df.sort_index()
        # Ensure distinct 15min grid
        traffic_df = traffic_df.resample("15min").asfreq()
    except Exception as e:
        raise RuntimeError(f"Traffic timestamp processing failed: {e}")

    # 2. Prepare Weather (Features)
    try:
        weather_df = weather_df.copy()
        weather_df.index = pd.to_datetime(weather_df.index, errors="coerce")
        weather_df = weather_df.sort_index()
        # Weather is allowed to be forward filled (atmosphere changes slowly)
        weather_df = weather_df.resample("15min").ffill()
    except Exception as e:
        raise RuntimeError(f"Weather timestamp processing failed: {e}")

    # 3. Merge
    try:
        # Left join: We only care about times where we have TRAFFIC data
        merged = traffic_df.join(weather_df, how="left")
        
        # FIX: Fill only weather columns, NOT traffic columns
        weather_cols = weather_df.columns
        merged[weather_cols] = merged[weather_cols].ffill(limit=2)

        # Sanity Check: Clip negatives (just in case)
        if "total_volume" in merged.columns:
            merged["total_volume"] = merged["total_volume"].clip(lower=0)

    except Exception as e:
        raise RuntimeError(f"Failed to merge weather + traffic: {e}")

    return merged