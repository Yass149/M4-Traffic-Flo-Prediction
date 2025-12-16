"""
Data Audit and Quality Assurance Module.

This module provides tools to audit the raw datasets before they enter the preprocessing
pipelines. It includes specialized logic for:
1.  **Weather Data (FM-12):** Parsing and validating comma-separated NOAA string codes
    (checking for '9999' sentinels, missing components, and physical range sanity).
2.  **Traffic Data:** Checking for time-series continuity, zero-volume anomalies,
    and physical realism (e.g., negative speeds).

Dependencies:
    - numpy
    - pandas
"""

import numpy as np
import pandas as pd


def sentinel_share(series: pd.Series, bad_values: list | str | int) -> float:
    """
    Calculates the percentage of non-NA entries that match specific 'bad' sentinel values.

    This is useful for NOAA data where missing values are often encoded as specific
    integers (e.g., 9999) rather than standard NaNs.

    Args:
        series (pd.Series): The data series to check.
        bad_values (list | str | int): A single value or list of values to treat as 'bad'.

    Returns:
        float: The percentage (0.0 to 100.0) of present data that matches the bad values.
               Returns 0.0 if the series is entirely NA.
    """
    if not isinstance(bad_values, (list, tuple, set)):
        bad_values = [bad_values]
    s = series.astype("string")
    n = s.notna().sum()
    if n == 0:
        return 0.0
    bad = s.isin(bad_values).sum()
    return 100.0 * bad / n


def split_fm12(series: pd.Series, idx: int, missing: list | str) -> pd.Series:
    """
    Extracts a specific component from a comma-separated FM-12 weather string.

    NOAA Integrated Surface Database (ISD) data packs multiple observations into single
    strings. For example, a wind column might look like: `"330,1,N,0021,1"`.
    This function splits the string and returns the component at `idx`, replacing
    specific sentinel codes with NaN.

    

    Args:
        series (pd.Series): Series containing FM-12 formatted strings.
        idx (int): Zero-based index of the component to extract (e.g., 0 for wind dir, 3 for speed).
        missing (list | str): Value(s) within that specific component that represent
                              missing data (e.g., "9999").

    Returns:
        pd.Series: The extracted component series with missing values replaced by NaN.
    """
    try:
        parts = series.astype("string").str.split(",", expand=True)
        col = parts[idx]
        if isinstance(missing, (list, tuple, set)):
            col = col.replace(list(missing), np.nan)
        else:
            col = col.replace(missing, np.nan)
        return col
    except Exception as e:
        print(f"FM‑12 decode error for {series.name} (idx={idx}): {e}")
        return pd.Series([np.nan] * len(series), index=series.index)


def weather_data_audit(df: pd.DataFrame) -> dict:
    """
    Performs a comprehensive QA/QC audit on FM-12 hourly weather data.

    This function analyzes the dataframe for structural integrity, missingness, and
    physical plausibility. It specifically handles the complexity of FM-12 sentinel
    codes (e.g., '99999' for missing pressure).

    **Side Effect:**
        This function **modifies the input DataFrame** by adding new columns representing
        the split components (e.g., `TMP_val`, `TMP_q`) to facilitate inspection.

    Args:
        df (pd.DataFrame): Raw weather dataframe containing FM-12 columns (TMP, DEW, SLP, AA1).

    Returns:
        dict: A nested dictionary containing the audit report:
            - **basic:** Rows, cols, duplicate index %, time coverage.
            - **missing:** Raw NA % for every column.
            - **sentinels:** Detailed breakdown of sentinel code frequency for specific fields.
            - **ranges:** Min/Max values for physical variables (Temp, Dew, Pressure) post-cleaning.
    """
    report = {
        "basic": {},
        "missing": {},
        "sentinels": {},
        "ranges": {},
    }

    # ---------- BASIC STRUCTURE ----------

    n_rows, n_cols = df.shape
    dup_pct = 100.0 * df.index.duplicated().sum() / n_rows if n_rows else 0.0

    if isinstance(df.index, pd.DatetimeIndex):
        t_start = df.index.min()
        t_end = df.index.max()
        freq = pd.infer_freq(df.index)
    else:
        t_start = t_end = freq = None

    report["basic"] = {
        "rows": int(n_rows),
        "cols": int(n_cols),
        "index_is_datetime": isinstance(df.index, pd.DatetimeIndex),
        "time_start": t_start,
        "time_end": t_end,
        "time_freq_guess": freq,
        "duplicate_index_pct": dup_pct,
    }

    # ---------- RAW MISSING VALUES ----------

    for col in df.columns:
        s = df[col]
        na_pct = 100.0 * s.isna().sum() / len(s) if len(s) else 0.0
        report["missing"][col] = {"na_pct_raw": na_pct}

    # ---------- SENTINEL CODES (FM‑12) ----------

    sent = {}

    fm12_fields = [
        ("TMP", "9999,9",  ["9999"]),    # air temperature
        ("DEW", "9999,9",  ["9999"]),    # dew point
        ("SLP", "99999,9", ["99999"]),   # sea‑level pressure
    ]

    for col, raw_sentinel, value_sentinels in fm12_fields:
        if col not in df.columns:
            continue

        s = df[col].astype("string")

        # raw FM‑12 tuple sentinel %
        raw_pct = sentinel_share(s, raw_sentinel)

        # split into value / quality; treat sentinel codes as missing in value
        val = split_fm12(s, 0, missing=value_sentinels)
        q   = split_fm12(s, 1, missing=["9"])  # '9' ~ missing/unknown flag

        val_col = f"{col}_val"
        q_col   = f"{col}_q"
        df[val_col] = val
        df[q_col]   = q

        # sentinel shares in the unsanitized value component
        val_raw = split_fm12(s, 0, missing=[])  # no replacement
        val_sentinel_pct = sentinel_share(val_raw, value_sentinels)
        q_raw = split_fm12(s, 1, missing=[])
        q9_pct = sentinel_share(q_raw, "9")

        # NA% after replacement for the split components
        val_na_pct = 100.0 * val.isna().sum() / len(val)
        q_na_pct   = 100.0 * q.isna().sum() / len(q)

        report["missing"][val_col] = {"na_pct_raw": val_na_pct}
        report["missing"][q_col]   = {"na_pct_raw": q_na_pct}

        sent[col] = {
            "raw_sentinel": raw_sentinel,
            "raw_sentinel_pct": raw_pct,
            "value_sentinels": value_sentinels,
            "value_sentinels_pct": val_sentinel_pct,
            "quality_9_pct": q9_pct,
        }

    # AA1 (precip): code, amount, period, quality
    if "AA1" in df.columns:
        aa = df["AA1"].astype("string")

        # split, treating sentinels as missing where appropriate
        aa_code = split_fm12(aa, 0, missing=["99"])
        aa_val  = split_fm12(aa, 1, missing=["9999"])
        aa_per  = split_fm12(aa, 2, missing=["9"])
        aa_q    = split_fm12(aa, 3, missing=["9"])

        df["AA1_code"] = aa_code
        df["AA1_val"]  = aa_val
        df["AA1_per"]  = aa_per
        df["AA1_q"]    = aa_q

        # sentinel shares in raw components (no replacement)
        aa_code_raw = split_fm12(aa, 0, missing=[])
        aa_val_raw  = split_fm12(aa, 1, missing=[])
        aa_q_raw    = split_fm12(aa, 3, missing=[])

        code06_pct   = sentinel_share(aa_code_raw, "06")
        code99_pct   = sentinel_share(aa_code_raw, "99")
        val9999_pct  = sentinel_share(aa_val_raw, "9999")
        q9_pct       = sentinel_share(aa_q_raw, "9")

        # NA% after replacement
        for comp_name, comp_series in [
            ("AA1_code", aa_code),
            ("AA1_val",  aa_val),
            ("AA1_per",  aa_per),
            ("AA1_q",    aa_q),
        ]:
            report["missing"][comp_name] = {
                "na_pct_raw": 100.0 * comp_series.isna().sum() / len(comp_series)
            }

        sent["AA1"] = {
            "code_06_pct": code06_pct,
            "code_99_pct": code99_pct,
            "val_9999_pct": val9999_pct,
            "quality_9_pct": q9_pct,
        }

    report["sentinels"] = sent

    # ---------- SIMPLE RANGE CHECKS ----------

    ranges = {}

    def robust_min_max(series, low_clip=None, high_clip=None):
        s = pd.to_numeric(series, errors="coerce")
        if low_clip is not None:
            s = s[s > low_clip]
        if high_clip is not None:
            s = s[s < high_clip]
        if s.dropna().empty:
            return None, None
        return float(s.min()), float(s.max())

    # TMP_val / DEW_val roughly [-50, 50] °C
    for col in ["TMP_val", "DEW_val"]:
        if col in df.columns:
            mn, mx = robust_min_max(df[col], low_clip=-100, high_clip=100)
            ranges[col] = {
                "min": mn,
                "max": mx,
                "expected_min": -50,
                "expected_max": 50,
            }

    # SLP_val roughly [85000, 108000] Pa
    if "SLP_val" in df.columns:
        mn, mx = robust_min_max(df["SLP_val"], low_clip=8000, high_clip=110000)
        ranges["SLP_val"] = {
            "min": mn,
            "max": mx,
            "expected_min": 85000,
            "expected_max": 108000,
        }

    report["ranges"] = ranges

    # ---------- PRINT SUMMARY ----------

    print("=== BASIC STRUCTURE ===")
    print(f"Rows: {n_rows}, Cols: {n_cols}")
    if isinstance(df.index, pd.DatetimeIndex):
        print(f"Time span: {t_start}  →  {t_end}")
        print(f"Inferred freq: {freq}")
    print(f"Duplicate index: {dup_pct:5.2f}%\n")

    print("=== MISSING VALUES (raw & split) ===")
    miss_sorted = sorted(report["missing"].items(),
                         key=lambda kv: kv[1]["na_pct_raw"],
                         reverse=True)
    for col, info in miss_sorted:
        print(f"  {col:12s}: {info['na_pct_raw']:5.2f}% NA")
    print()

    print("=== FM‑12 SENTINELS ===")
    for col, info in sent.items():
        print(f"[{col}]")
        for k, v in info.items():
            if isinstance(v, (int, float)):
                print(f"  {k:22s}: {v:6.2f}")
            else:
                print(f"  {k:22s}: {v}")
        print()

    print("=== VALUE RANGES (post‑split) ===")
    for col, info in ranges.items():
        print(f"{col:10s}: min={info['min']}, max={info['max']} "
              f"vs expected [{info['expected_min']}, {info['expected_max']}]")
    print()

    return report


def traffic_data_audit(df: pd.DataFrame) -> dict:
    """
    QA/QC audit for a cleaned traffic panel.

    This function expects a traffic dataframe that has typically been standardized
    (e.g., via `TrafficPreprocessor`). It checks for common sensor errors like
    negative speeds, zero-volume runs, and data gaps.

    **Note:** This function operates on a copy of the dataframe and does not modify
    the original input.

    Args:
        df (pd.DataFrame): Traffic data with at least `total_volume` and `avg_mph` columns.
                           Ideally indexed by Datetime, but can handle a 'timestamp' column.

    Returns:
        dict: A nested dictionary containing the audit report:
            - **basic:** Rows, cols, freq, index duplicates.
            - **missing:** NA % per column.
            - **volumes:** Stats on `total_volume` (min, max, % zero, time-of-day means).
            - **speeds:** Stats on `avg_mph` (% >90mph, % negative).
            - **consistency:** Completeness of daily records (e.g., % days with full 96 intervals).
    """

    report = {
        "basic": {},
        "missing": {},
        "volumes": {},
        "speeds": {},
        "consistency": {},
    }

    # ---------- ENSURE TIME INDEX ----------

    df = df.copy()

    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        df["timestamp"] = ts
        df = df.set_index("timestamp").sort_index()
    elif isinstance(df.index, pd.DatetimeIndex):
        ts = df.index
    else:
        ts = None

    # ---------- BASIC STRUCTURE ----------

    n_rows, n_cols = df.shape
    if isinstance(df.index, pd.DatetimeIndex):
        t_start = df.index.min()
        t_end = df.index.max()
        freq = pd.infer_freq(df.index)
        dup_pct = 100.0 * df.index.duplicated().sum() / len(df)
    else:
        t_start = t_end = freq = dup_pct = None

    report["basic"] = {
        "rows": int(n_rows),
        "cols": int(n_cols),
        "has_datetime_index": isinstance(df.index, pd.DatetimeIndex),
        "time_start": t_start,
        "time_end": t_end,
        "time_freq_guess": freq,
        "duplicate_index_pct": dup_pct,
    }

    # ---------- MISSINGNESS ----------

    miss = {}
    for col in df.columns:
        na_pct = 100.0 * df[col].isna().sum() / len(df) if len(df) else 0.0
        miss[col] = {"na_pct": na_pct}
    report["missing"] = miss

    # ---------- VOLUME STATS ----------

    vol_stats = {}
    if "total_volume" in df.columns:
        v = pd.to_numeric(df["total_volume"], errors="coerce")

        vol_stats["na_pct"] = 100.0 * v.isna().sum() / len(v)
        vol_stats["min"] = float(v.min()) if v.dropna().size else None
        vol_stats["max"] = float(v.max()) if v.dropna().size else None
        vol_stats["mean"] = float(v.mean()) if v.dropna().size else None
        vol_stats["pct_zero"] = 100.0 * (v == 0).sum() / len(v)

        # non‑integer check
        non_int_mask = v.notna() & (v % 1 != 0)
        vol_stats["non_integer_pct"] = 100.0 * non_int_mask.sum() / len(v)

        # time‑of‑day pattern (if we have a datetime index)
        if isinstance(df.index, pd.DatetimeIndex):
            vol_by_tod = v.groupby(df.index.time).mean()
            vol_stats["by_time_of_day_mean"] = vol_by_tod.to_dict()

    report["volumes"] = vol_stats

    # ---------- SPEED STATS ----------

    speed_stats = {}
    if "avg_mph" in df.columns:
        s = pd.to_numeric(df["avg_mph"], errors="coerce")

        speed_stats["na_pct"] = 100.0 * s.isna().sum() / len(s)
        speed_stats["min"] = float(s.min()) if s.dropna().size else None
        speed_stats["max"] = float(s.max()) if s.dropna().size else None
        speed_stats["mean"] = float(s.mean()) if s.dropna().size else None

        # simple realism checks
        speed_stats["pct_over_90mph"] = 100.0 * (s > 90).sum() / len(s)
        speed_stats["pct_negative"] = 100.0 * (s < 0).sum() / len(s)

    report["speeds"] = speed_stats

    # ---------- CONSISTENCY / COVERAGE ----------

    cons = {}

    # daily completeness: share of days with 96×15min bins (if regular 15‑min data)
    if isinstance(df.index, pd.DatetimeIndex):
        per_day = df.groupby(df.index.normalize()).size()
        cons["days_with_96_intervals_pct"] = \
            100.0 * (per_day == 96).sum() / len(per_day)
        cons["days_with_any_data"] = int((per_day > 0).sum())

    report["consistency"] = cons

    # ---------- PRINT SUMMARY ----------

    print("=== TRAFFIC BASIC STRUCTURE ===")
    print(f"Rows: {n_rows}, Cols: {n_cols}")
    if isinstance(df.index, pd.DatetimeIndex):
        print(f"Time span: {t_start}  →  {t_end}")
        print(f"Inferred freq: {freq}")
        print(f"Duplicate index: {dup_pct:5.2f}%")
    print()

    print("=== MISSING COLUMNS ===")
    miss_sorted = sorted(miss.items(), key=lambda kv: kv[1]["na_pct"], reverse=True)
    for col, info in miss_sorted:
        print(f"  {col:12s}: {info['na_pct']:5.2f}% NA")
    print()

    print("=== VOLUME STATS ===")
    for k, v in vol_stats.items():
        if k != "by_time_of_day_mean":
            print(f"  {k:20s}: {v}")
    print()

    print("=== SPEED STATS ===")
    for k, v in speed_stats.items():
        print(f"  {k:20s}: {v}")
    print()

    print("=== CONSISTENCY ===")
    for k, v in cons.items():
        print(f"  {k:20s}: {v}")
    print()

    return report