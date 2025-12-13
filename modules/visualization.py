import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

class TrafficVisualizer:
    """
    Handles all Exploratory Data Analysis (EDA) plotting.
    Consolidates Ahmed's three scripts into one pipeline-ready class.
    
    Assigned to: Ahmed
    Refactored by: Integration Lead
    """
    
    def __init__(self, df):
        """
        Args:
            df (pd.DataFrame): The fully merged 'full_df' (Traffic + Weather + Features).
        """
        self.df = df.copy()
        # Set professional style
        sns.set_theme(style="whitegrid")
        plt.rcParams['figure.figsize'] = (14, 7)
        
    def plot_traffic_overview(self):
        """
        Plots the full time series of traffic volume.
        (From Ahmed's traffic_visualisations.py)
        """
        plt.figure()
        plt.plot(self.df.index, self.df['total_volume'], label='Traffic Volume', color='#1f77b4', alpha=0.6, linewidth=0.5)
        
        # Add a rolling average for clarity
        rolling_mean = self.df['total_volume'].rolling(window=96*7).mean() # 7 Day rolling
        plt.plot(self.df.index, rolling_mean, label='7-Day Moving Avg', color='red', linewidth=1.5)
        
        plt.title("Traffic Volume History (M4 Motorway: 2021-2024)", fontsize=16, weight='bold')
        plt.ylabel("Volume (Count/15min)")
        plt.xlabel("Date")
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()

    def plot_hourly_profile(self):
        """
        Plots the average 24-hour traffic profile.
        Essential for showing 'Rush Hour' patterns.
        """
        plt.figure()
        # Extract time if not present
        if 'time' not in self.df.columns:
            self.df['time'] = self.df.index.time
            
        daily_avg = self.df.groupby('time')['total_volume'].mean()
        
        # Create plot
        time_labels = [t.strftime('%H:%M') for t in daily_avg.index]
        plt.plot(time_labels, daily_avg.values, color='#ff7f0e', linewidth=3)
        plt.fill_between(time_labels, daily_avg.values, color='#ff7f0e', alpha=0.2)
        
        # Format X-axis to show every 2 hours
        plt.xticks(ticks=range(0, len(time_labels), 8), labels=time_labels[::8], rotation=45)
        plt.title("Average Daily Traffic Profile (Seasonality)", fontsize=16, weight='bold')
        plt.ylabel("Average Volume")
        plt.xlabel("Time of Day")
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

    def plot_weather_impact(self):
        """
        Scatter plot of Speed vs Volume, colored by Rain.
        (From Ahmed's traffic_weather_visualisations.py)
        """
        plt.figure(figsize=(12, 8))
        
        # Filter out static to make plot clearer
        subset = self.df.sample(5000) if len(self.df) > 5000 else self.df
        
        sns.scatterplot(
            data=subset, 
            x='avg_mph', 
            y='total_volume', 
            hue='rain_flag', 
            palette={0: 'blue', 1: 'red'},
            alpha=0.5
        )
        
        plt.title("Impact of Speed and Rain on Traffic Volume", fontsize=16, weight='bold')
        plt.xlabel("Average Speed (mph)")
        plt.ylabel("Traffic Volume")
        plt.legend(title="Rain (1=Yes)", loc='upper left')
        plt.show()

    def plot_feature_correlation(self):
        """
        Plots heatmap to justify Feature Selection (5% marks).
        (Combines logic from all files)
        """
        plt.figure(figsize=(10, 8))
        
        # Select numeric columns only
        cols = ['total_volume', 'avg_mph', 'temperature_C', 'rain_mm', 'wind_speed_ms', 'visibility_m', 'hour']
        # Only use columns that exist
        valid_cols = [c for c in cols if c in self.df.columns]
        
        corr_df = self.df[valid_cols].corr()
        
        sns.heatmap(corr_df, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
        plt.title("Feature Correlation Matrix", fontsize=16, weight='bold')
        plt.tight_layout()
        plt.show()

    def plot_day_type_comparison(self):
        """
        Boxplot: School Term (0) vs Holiday/Weekend (1).
        Required for Phase 3 'Twist' Visual Proof.
        """
        if 'day_type' not in self.df.columns:
            return

        plt.figure(figsize=(10, 6))
        sns.boxplot(x='day_type', y='total_volume', data=self.df, palette="Set2")
        plt.xticks([0, 1], ['School Term (0)', 'Holiday/Weekend (1)'])
        plt.title("Distribution: School Term vs. Holiday Traffic", fontsize=16, weight='bold')
        plt.ylabel("Volume")
        plt.show()

    def plot_regression_performance(self, y_true, y_pred, model_name='Model', r2_score=None):
        """
        Plots a professional dashboard for Regression Evaluation.
        1. Reality Check: Zooms into the last 7 days of VALID data (skips trailing zeros).
        2. Scatter Plot: Actual vs Predicted with a perfect fit line.
        """
        # Ensure inputs are Series with matching indices
        if not isinstance(y_true, pd.Series):
            y_true = pd.Series(y_true)
        if not isinstance(y_pred, pd.Series):
            y_pred = pd.Series(y_pred, index=y_true.index)

        # --- SMART SLICE LOGIC ---
        # Find the last timestamp where Actual Volume > 10 (Avoids the flat line at the end)
        valid_indices = y_true[y_true > 10].index
        if len(valid_indices) > 0:
            last_valid_idx = valid_indices.max()
            loc_idx = y_true.index.get_loc(last_valid_idx)
            # Slice last 7 days from that point
            intervals = 96 * 7
            start_loc = max(0, loc_idx - intervals)
            zoom_slice = slice(start_loc, loc_idx)
        else:
            # Fallback if data is weird
            zoom_slice = slice(-96*7, None)

        # Create Plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # 1. Time Series Zoom
        axes[0].plot(y_true.index[zoom_slice], y_true.iloc[zoom_slice], label='Actual', color='black', alpha=0.5, linewidth=2)
        axes[0].plot(y_pred.index[zoom_slice], y_pred.iloc[zoom_slice], label='Predicted', color='#2ca02c', linewidth=1.5, linestyle='--')
        axes[0].set_title(f"{model_name} Reality Check: Last 7 Valid Days", fontsize=14, weight='bold')
        axes[0].set_ylabel("Volume")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 2. Scatter Plot (Sampled for speed/clarity)
        # Filter out zeros for the scatter to show true correlation
        mask = y_true > 10
        y_t_clean = y_true[mask]
        y_p_clean = y_pred[mask]
        
        # Downsample if too large
        if len(y_t_clean) > 2000:
            indices = np.random.choice(len(y_t_clean), 2000, replace=False)
            y_t_samp = y_t_clean.iloc[indices]
            y_p_samp = y_p_clean.iloc[indices]
        else:
            y_t_samp, y_p_samp = y_t_clean, y_p_clean

        axes[1].scatter(y_t_samp, y_p_samp, alpha=0.15, color='#1f77b4')
        
        # Perfect fit line
        limit = y_t_clean.max()
        axes[1].plot([0, limit], [0, limit], 'r--', linewidth=2, label='Perfect Fit')
        
        title = "Prediction Accuracy"
        if r2_score is not None:
            title += f" (R2 = {r2_score:.2f})"
            
        axes[1].set_title(title, fontsize=14, weight='bold')
        axes[1].set_xlabel("Actual Volume")
        axes[1].set_ylabel("Predicted Volume")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()