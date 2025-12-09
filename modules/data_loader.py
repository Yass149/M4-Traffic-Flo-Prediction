import pandas as pd
import glob
import os

#-- TRAFFIC DATA LOADER FUNCTION --#
def load_traffic_data(data_dir='data/traffic'):
    """
    Loads, cleans, and aggregates traffic data from multiple CSV files.

    Pipeline Logic:
    1. Loads all CSVs.
    2. Parses dates.
    3. Resamples directly to a 15-minute grid using Aggregation (Sum Volume, Mean Speed).
       * This fixes the bug where slightly misaligned timestamps caused data loss.

    Args:
        data_dir (str): Path to traffic data directory.

    Returns:
        pd.DataFrame: Cleaned DataFrame with 'timestamp', 'total_volume', 'avg_mph'.
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
            temp_df = pd.read_csv(f)
            
            # Clean columns immediately
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
    raw_df = raw_df.loc[:, ~raw_df.columns.duplicated()]

    try:
        # 5. Parse Dates
        raw_df['timestamp'] = pd.to_datetime(
            raw_df['Report Date'] + ' ' + raw_df['Time Period Ending'], 
            dayfirst=True, 
            errors='coerce'
        )
        raw_df.dropna(subset=['timestamp'], inplace=True)

        # 6. Standardize Column Names
        if 'Total Volume' in raw_df.columns:
            raw_df.rename(columns={'Total Volume': 'total_volume'}, inplace=True)
        if 'Avg mph' in raw_df.columns:
            raw_df.rename(columns={'Avg mph': 'avg_mph'}, inplace=True)

        # 7. Resample & Aggregate (THE FIX)
        # Instead of groupby() then resample(), we resample the raw data directly.
        # This captures all records within the 15min window, even if seconds are off.
        
        raw_df.set_index('timestamp', inplace=True)
        
        final_df = raw_df.resample('15min').agg({
            'total_volume': 'sum',
            'avg_mph': 'mean'
        })

        # Fill missing speeds with 0 or forward fill if needed, but for now keep NaNs 
        # so the Imputer in Regression handles them.
        
        print(f"✅ Traffic Data Loaded. Rows: {len(final_df)}")
        print(f"   Timeframe: {final_df.index.min()} to {final_df.index.max()}")
        
        return final_df.reset_index()

    except Exception as e:
        print(f"❌ Critical Error during processing: {e}")
        return None