import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np

class TrafficVisualizer:
    """
    Handles all plotting for Traffic Analysis (EDA, Regression, Time Series).
    """

    def __init__(self, df):
        self.df = df.copy()
        # Ensure date components exist for plotting
        if 'hour' not in self.df.columns:
            self.df['hour'] = self.df.index.hour
        if 'day_of_week' not in self.df.columns:
            self.df['day_name'] = self.df.index.day_name()
            self.df['day_of_week'] = self.df.index.dayofweek

    # ===============================================================
    # 1. EXPLORATORY DATA ANALYSIS (EDA)
    # ===============================================================

    def plot_traffic_overview(self):
        plt.figure(figsize=(15, 5))
        plt.plot(self.df.index, self.df['total_volume'], color='#333333', linewidth=0.5, alpha=0.8)
        plt.title("Traffic Volume Overview (Full Timeline)", fontsize=14, weight='bold')
        plt.ylabel("Volume (Vehicles / 15min)")
        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_weekly_patterns(self):
        plt.figure(figsize=(12, 6))
        order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        if 'day_name' not in self.df.columns: self.df['day_name'] = self.df.index.day_name()
        sns.boxplot(x='day_name', y='total_volume', data=self.df, order=order, palette="viridis")
        plt.title("Traffic Volume Distribution by Day of Week", fontsize=14, weight='bold')
        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_hourly_profile(self):
        self.df['is_weekend'] = self.df.index.dayofweek >= 5
        weekday = self.df[~self.df['is_weekend']].groupby(self.df[~self.df['is_weekend']].index.hour)['total_volume'].mean()
        weekend = self.df[self.df['is_weekend']].groupby(self.df[self.df['is_weekend']].index.hour)['total_volume'].mean()
        
        plt.figure(figsize=(12, 6))
        plt.plot(weekday.index, weekday.values, label='Weekdays (Mon-Fri)', color='#1f77b4', linewidth=3)
        plt.plot(weekend.index, weekend.values, label='Weekends (Sat-Sun)', color='#ff7f0e', linewidth=3)
        plt.title("Average Hourly Traffic Profile", fontsize=14, weight='bold')
        plt.xlabel("Hour of Day")
        plt.xticks(range(0, 24))
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_correlation_heatmap(self):
        cols = ['total_volume', 'avg_mph', 'temperature_C', 'precip_mm', 'wind_speed_ms', 'visibility_m']
        valid_cols = [c for c in cols if c in self.df.columns]
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.df[valid_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
        plt.title("Correlation Matrix", fontsize=14, weight='bold')
        plt.show()

    def plot_speed_flow_relationship(self):
        if 'avg_mph' not in self.df.columns: return
        plt.figure(figsize=(10, 6))
        plt.scatter(self.df['total_volume'], self.df['avg_mph'], alpha=0.05, color='purple', s=2)
        plt.title("Fundamental Diagram (Speed vs Flow)", fontsize=14, weight='bold')
        plt.xlabel("Flow (Volume)")
        plt.ylabel("Speed (mph)")
        plt.grid(True, alpha=0.3)
        plt.show()

    # ===============================================================
    # 2. REGRESSION PLOTS
    # ===============================================================

    def plot_regression_performance(self, y_true, y_pred, model_name="Model", r2_score=None):
        plt.figure(figsize=(14, 6))
        
        # Subplot 1: Time Series (Last 7 Days)
        plt.subplot(1, 2, 1)
        subset_n = 96 * 7
        y_true_sub = y_true.tail(subset_n) if hasattr(y_true, 'tail') else y_true[-subset_n:]
        y_pred_sub = y_pred[-subset_n:]
        
        plt.plot(y_true_sub.values, label='Actual', color='grey', alpha=0.8, linewidth=2)
        plt.plot(y_pred_sub, label='Predicted', color='green', linestyle='--', linewidth=2)
        plt.title(f"{model_name}: Last 7 Days", fontsize=12, weight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Subplot 2: Scatter
        plt.subplot(1, 2, 2)
        plt.scatter(y_true, y_pred, alpha=0.05, color='#1f77b4')
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Fit')
        plt.title(f"R2 = {r2_score:.2f}" if r2_score else "Prediction Accuracy", fontsize=12, weight='bold')
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    # ===============================================================
    # 3. TIME SERIES PLOTS (PUBLICATION QUALITY)
    # ===============================================================

    def plot_forecast_publication_quality(self, train, test, forecast, rmse_val=None):
        """
        Plots a high-quality forecast zoomed in on the transition period.
        """
        plt.figure(figsize=(16, 7))
        
        # 1. Zoom Logic: Only show last 7 days of training
        # This fixes the "squashed to the right" issue
        context_days = 7
        train_zoom = train.tail(24 * context_days)

        # 2. Plot Training (History)
        plt.plot(train_zoom.index, train_zoom, label='Historical (Last 7 Days)', 
                 color='gray', alpha=0.6, linewidth=1.5)

        # 3. Plot Actual Test Data
        plt.plot(test.index, test, label='Actual Ground Truth', 
                 color='#1f77b4', linewidth=2.5, alpha=0.8)

        # 4. Plot Forecast
        plt.plot(test.index, forecast, label='SARIMAX Forecast', 
                 color='#2ca02c', linewidth=3, linestyle='--')

        # 5. Confidence Band
        if rmse_val is not None:
            plt.fill_between(test.index, 
                             forecast - rmse_val, 
                             forecast + rmse_val, 
                             color='#2ca02c', alpha=0.15, label='Uncertainty (±1 RMSE)')

        # 6. Split Line
        split_time = train_zoom.index[-1]
        plt.axvline(x=split_time, color='red', linestyle=':', linewidth=2, label='Forecast Start')

        # 7. Formatting
        plt.title("M4 Traffic Forecast: SARIMAX with Fourier Seasonality", fontsize=16, weight='bold')
        plt.xlabel("Date & Time", fontsize=12)
        plt.ylabel("Traffic Volume (Vehicles/Hour)", fontsize=12)
        plt.legend(loc='upper left', fontsize=11, frameon=True)
        
        # Date Formatting
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%a %H:%M'))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_acf_pacf(self, series, lags=48):
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
        plot_acf(series, lags=lags, ax=ax1)
        plot_pacf(series, lags=lags, ax=ax2)
        plt.tight_layout()
        plt.show()

    def plot_decomposition(self, decomposition):
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 12), sharex=True)
        decomposition.observed.plot(ax=ax1, color='black')
        ax1.set_title('STL Decomposition')
        decomposition.trend.plot(ax=ax2, color='blue')
        ax2.set_ylabel('Trend')
        decomposition.seasonal.plot(ax=ax3, color='green')
        ax3.set_ylabel('Seasonal')
        decomposition.resid.plot(ax=ax4, color='red', marker='.', linestyle='None')
        ax4.set_ylabel('Residual')
        plt.tight_layout()
        plt.show()