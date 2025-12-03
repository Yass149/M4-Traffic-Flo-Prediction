import pandas as pd
import glob
import os

def load_traffic_data(data_dir='data/traffic'):
    """
    Loads, cleans, and aggregates traffic data from multiple CSV files.

    Optimized logic:
    1. Loads all CSVs.
    2. Proactively removes conflicting columns (timestamp/unnamed) before merging.
    3. Merges data and removes duplicate columns.
    4. Parses dates using UK format (dayfirst=True).
    5. Aggregates data: Sums volume across sites, averages speed.
    6. Resamples to a strict 15-minute interval grid.

    Args:
        data_dir (str): Path to traffic data directory.

    Returns:
        pd.DataFrame: Cleaned DataFrame with 'timestamp', 'total_volume', 'avg_mph'.
                      Returns None if critical errors occur.
    """
    # 1. Validate Directory
    if not os.path.exists(data_dir):
        print(f"❌ Error: Directory '{data_dir}' not found.")
        return None

    # 2. Find Files
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not csv_files:
        print(f"❌ Error: No CSV files found in '{data_dir}'.")
        return None

    print(f"📂 Found {len(csv_files)} traffic files. Loading...")

    df_list = []
    
    # 3. Load Files Loop
    for f in csv_files:
        try:
            # Read CSV
            temp_df = pd.read_csv(f)
            
            # OPTIMIZATION: Drop 'timestamp' or 'Unnamed' columns immediately to prevent merge conflicts
            cols_to_drop = [c for c in temp_df.columns if 'timestamp' in c.lower() or 'unnamed' in c.lower()]
            if cols_to_drop:
                temp_df.drop(columns=cols_to_drop, inplace=True)
            
            df_list.append(temp_df)
        except Exception as e:
            print(f"⚠️ Warning: Could not read {f}. Reason: {e}")

    if not df_list:
        return None

    # 4. Merge Data
    raw_df = pd.concat(df_list, ignore_index=True)
    
    # OPTIMIZATION: Drop duplicate columns (if any exist after merge)
    raw_df = raw_df.loc[:, ~raw_df.columns.duplicated()]

    try:
        # 5. Parse Dates (Robust Method)
        # Combine Date + Time. Use dayfirst=True for UK formats (DD/MM/YYYY)
        raw_df['timestamp'] = pd.to_datetime(
            raw_df['Report Date'] + ' ' + raw_df['Time Period Ending'], 
            dayfirst=True, 
            errors='coerce'
        )
        
        # Validation: Check if parsing failed
        if raw_df['timestamp'].isnull().all():
            print("❌ Error: Date parsing failed. Check CSV date format.")
            return None
            
        raw_df.dropna(subset=['timestamp'], inplace=True)

        # 6. Standardize Column Names
        if 'Total Volume' in raw_df.columns:
            raw_df.rename(columns={'Total Volume': 'total_volume'}, inplace=True)
        if 'Avg mph' in raw_df.columns:
            raw_df.rename(columns={'Avg mph': 'avg_mph'}, inplace=True)

        # 7. Aggregate (Summing separate camera sites)
        # We group by time and SUM the volume, but AVERAGE the speed
        aggregated_df = raw_df.groupby('timestamp').agg({
            'total_volume': 'sum',
            'avg_mph': 'mean'
        }).sort_index()

        # 8. Resample (Ensure strict 15-min grid)
        # This creates NaNs for missing periods (important for Time Series models)
        final_df = aggregated_df.resample('15min').asfreq()

        # Final Output Stats
        print(f"✅ Traffic Data Loaded. Rows: {len(final_df)}")
        print(f"   Timeframe: {final_df.index.min()} to {final_df.index.max()}")
        
        return final_df.reset_index()

    except Exception as e:
        print(f"❌ Critical Error during processing: {e}")
        return None