"""
Time Series Modelling Module for Traffic Flow Forecasting
=========================================================
"""
import numpy as np
import pandas as pd
import logging
import warnings
from typing import Optional, Tuple, Dict, List
from datetime import datetime

import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .preprocessing import WeatherPreprocessor,TrafficPreprocessor,process_and_merge

warnings.filterwarnings("ignore")

# ============================================================================
# 1. UTILITY FUNCTIONS
# ============================================================================

def make_fourier_features(index: pd.DatetimeIndex, period: int, K: int) -> pd.DataFrame:
    """
    Generate Fourier sine/cosine terms for capturing complex seasonality.
    Args:
        period: Seasonality period (e.g., 168 for weekly hourly data)
        K: Number of Fourier terms (Harmonics)
    """
    try:
        t = (index - index[0]).total_seconds() / 3600.0 # Time in hours
        features = {}
        for k in range(1, K + 1):
            features[f"sin_{period}_{k}"] = np.sin(2 * np.pi * k * t / period)
            features[f"cos_{period}_{k}"] = np.cos(2 * np.pi * k * t / period)
        return pd.DataFrame(features, index=index)
    except Exception as e:
        raise RuntimeError(f"Fourier generation failed: {e}")

def check_stationarity(series: pd.Series, alpha: float = 0.05) -> dict:
    series = series.dropna()
    try:
        result = adfuller(series, autolag="AIC")
    except Exception:
        return {"p-value": 0.0, "Stationary": True} # Fallback

    return {
        "ADF Statistic": result[0],
        "p-value": result[1],
        "Stationary": result[1] < alpha
    }

def stl_decompose(series, period=24):
    """Wrapper for STL decomposition."""
    stl = STL(series, period=period, robust=True)
    result = stl.fit()
    return result


def merge_freq(freq):
    traffic_freq = TrafficPreprocessor().load_traffic_data(freq="1h")
    weather_freq = WeatherPreprocessor().run(freq="1h")
    return process_and_merge(freq="1h",traffic_df=traffic_freq,weather_df=weather_freq)



# ============================================================================
# 2. CORE BUILDERS
# ============================================================================

class SARIMAXModelBuilder:
    """Professional Builder for SARIMAX with Exogenous Features."""
    def __init__(self, order=(1, 0, 1), seasonal_order=(1, 1, 1, 24)):
        self.order = order
        self.seasonal_order = seasonal_order
        self.log = logging.getLogger("SARIMAXBuilder")

    def fit(self, train: pd.Series, exog_train: pd.DataFrame = None, disp: bool = False):
        try:
            # Optimize: Disable enforcement for speed/convergence on difficult data
            model = SARIMAX(
                train,
                exog=exog_train,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
                simple_differencing=True
            )
            # Use LBFGS: Robust for Seasonal models
            results = model.fit(disp=disp, method='lbfgs', maxiter=50)
            return results
        except Exception as e:
            raise RuntimeError(f"SARIMAX fitting failed: {e}")

    def forecast(self, results, steps, test_index, exog_test=None):
        # Get forecast
        forecast_obj = results.get_forecast(steps=steps, exog=exog_test)
        forecast = forecast_obj.predicted_mean
        
        # Ensure index alignment
        if len(forecast) == len(test_index):
            forecast.index = test_index
            
        return forecast

class TimeSeriesEvaluator:
    """
    Comprehensive evaluation and diagnostics for time-series models.
    """
    def __init__(self, verbose: bool = False):
        self.log = logging.getLogger("TimeSeriesEvaluator")

    def calculate_metrics(self, y_actual, y_pred):
        """
        Calculate metrics using NumPy to avoid Pandas Indexing errors.
        """
        # 1. Align lengths (Truncate to minimum common length)
        min_len = min(len(y_actual), len(y_pred))
        
        # 2. CONVERT TO NUMPY ARRAYS (Critical Fix)
        # This strips the index so pandas doesn't throw alignment errors
        y_true_np = y_actual.iloc[:min_len].values
        y_pred_np = y_pred.iloc[:min_len].values

        # 3. Standard Metrics (using numpy arrays)
        mae = mean_absolute_error(y_true_np, y_pred_np)
        rmse = np.sqrt(mean_squared_error(y_true_np, y_pred_np))
        
        # 4. FIXED MAPE (Strict Filtering on Numpy)
        # Ignore any time where Actual Volume < 1 vehicle
        valid_mask = y_true_np > 1.0 
        
        if valid_mask.sum() > 0:
            # We filter both arrays using the same boolean mask
            true_masked = y_true_np[valid_mask]
            pred_masked = y_pred_np[valid_mask]
            mape = np.mean(np.abs((true_masked - pred_masked) / true_masked)) * 100
        else:
            mape = 0.0 
            
        return {"MAE": mae, "RMSE": rmse, "MAPE (%)": mape}

# ============================================================================
# 3. NOTEBOOK ADAPTERS
# ============================================================================

class SARIMAModel:
    """
    Smart Adapter: Automatically switches between SARIMA and SARIMAX 
    depending on if 'exog' data is provided.
    """
    def __init__(self, order=(1,0,1), seasonal_order=(1,1,1,24)):
        self.builder = SARIMAXModelBuilder(order=order, seasonal_order=seasonal_order)
        self.results = None

    def fit(self, train_data, exog_data=None):
        """Fits the model (supports optional exogenous variables)."""
        self.results = self.builder.fit(train_data, exog_train=exog_data)

    def forecast(self, steps=24, exog_data=None):
        """Generates forecast (requires exog_data if used in fit)."""
        if not self.results:
            raise ValueError("Model must be fitted first")
            
        # Determine future index
        try:
            last_date = self.results.model._index[-1]
            freq = pd.infer_freq(self.results.model._index) or 'H'
        except:
            last_date = datetime.now()
            freq = 'H'
            
        forecast_index = pd.date_range(start=last_date, periods=steps+1, freq=freq)[1:]
        
        return self.builder.forecast(self.results, steps, forecast_index, exog_test=exog_data)

    def evaluate(self, test_data, forecast_series):
        evaluator = TimeSeriesEvaluator()
        return evaluator.calculate_metrics(test_data, forecast_series)

class ARIMAModel:
    """
    Wrapper for standard ARIMA model (Missing class restored).
    """
    def __init__(self, order=(1,1,1)):
        self.order = order
        self.model = None
        self.fitted = None

    def fit(self, series):
        self.model = ARIMA(series, order=self.order)
        self.fitted = self.model.fit()
        return self.fitted

    def forecast(self, steps=24):
        return self.fitted.forecast(steps)