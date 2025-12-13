import pandas as pd
import glob
import os

def load_traffic_data(data_dir='data/traffic'):
    """
    Loads, cleans, and aggregates traffic data.
    
    CRITICAL UPGRADE:
    - Handles duplicates (Multiple Cameras) by taking the MEAN instead of SUM.
    - Prevents data drops when one camera goes offline.
    """
    # 1. Validate Directory
    if not os.path.exists(data_dir):
        print(f"❌ Error: Directory '{data_dir}' not found.")
        return None

    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not csv_files:
        print(f"❌ Error: No CSV files found.")
        return None

    print(f"📂 Found {len(csv_files)} traffic files. Loading...")

    df_list = []
    
    # 2. Load Files
    for f in csv_files:
        try:
            temp_df = pd.read_csv(f)
            # Basic cleaning
            cols_to_drop = [c for c in temp_df.columns if 'timestamp' in c.lower() or 'unnamed' in c.lower()]
            if cols_to_drop:
                temp_df.drop(columns=cols_to_drop, inplace=True)
            df_list.append(temp_df)
        except Exception as e:
            print(f"⚠️ Warning: Could not read {f}. Reason: {e}")

    if not df_list:
        return None

    # 3. Merge
    raw_df = pd.concat(df_list, ignore_index=True)

    # 4. Parse Dates
    try:
        raw_df['timestamp'] = pd.to_datetime(
            raw_df['Report Date'] + ' ' + raw_df['Time Period Ending'], 
            dayfirst=True, 
            errors='coerce'
        )
        raw_df.dropna(subset=['timestamp'], inplace=True)
        raw_df.set_index('timestamp', inplace=True)

        # 5. Standardize Columns
        if 'Total Volume' in raw_df.columns:
            raw_df.rename(columns={'Total Volume': 'total_volume'}, inplace=True)
        if 'Avg mph' in raw_df.columns:
            raw_df.rename(columns={'Avg mph': 'avg_mph'}, inplace=True)

        # --- THE FIX IS HERE ---
        # Step 6a: Handle Duplicates/Multiple Cameras FIRST
        # If we have multiple rows for "08:00", we average them.
        # This handles the case where one camera is missing.
        print("   > Aggregating multiple sensors (Using Mean to handle outages)...")
        
        # We group by the exact timestamp first to merge cameras
        grouped_df = raw_df.groupby(raw_df.index).agg({
            'total_volume': 'mean',  # CHANGED FROM SUM TO MEAN
            'avg_mph': 'mean'
        })
        
        # Step 6b: Resample to 15min grid
        # Now we fit it to the perfect 15-minute intervals
        final_df = grouped_df.resample('15min').mean() # Mean is safer for resampling too

        print(f"✅ Traffic Data Loaded. Rows: {len(final_df)}")
        return final_df.reset_index()

    except Exception as e:
        print(f"❌ Critical Error during processing: {e}")
        return None