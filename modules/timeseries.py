"""
Time Series Models Module
-------------------------

Provides:
- Stationarity checking (ADF test)
- ARIMA model wrapper
- SARIMA model wrapper

Designed for forecasting traffic volume or similar univariate time-series.
"""

import pandas as pd
import numpy as np

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


# ===============================================================
# 1. Stationarity Check (ADF Test)
# ===============================================================

def check_stationarity(series: pd.Series, alpha: float = 0.05) -> dict:
    """
    Perform the Augmented Dickey-Fuller (ADF) test to determine
    whether a time series is stationary.

    Parameters
    ----------
    series : pd.Series
        The time-series data to test.
    alpha : float, optional
        Significance threshold (default = 0.05).

    Returns
    -------
    dict
        Dictionary with ADF statistic, p-value, critical values,
        and stationarity boolean.
    """
    series = series.dropna()

    try:
        result = adfuller(series, autolag="AIC")
    except Exception as e:
        raise RuntimeError(f"ADF test failed: {e}")

    test_stat, p_value, lags, nobs, critical_values, ic_best = result

    return {
        "ADF Statistic": test_stat,
        "p-value": p_value,
        "Critical Values": critical_values,
        "Stationary": p_value < alpha
    }


# ===============================================================
# 2. ARIMA Model Class
# ===============================================================

class ARIMAModel:
    """
    Wrapper for ARIMA(p,d,q) modeling.

    Provides clean, simple fit() and forecast() methods.
    """

    def __init__(self, order=(1, 1, 1)):
        self.order = order
        self.model = None
        self.fitted = None

    def fit(self, series: pd.Series):
        """
        Fit the ARIMA model.

        Parameters
        ----------
        series : pd.Series
            Prepared univariate time-series.

        Returns
        -------
        ARIMAResults
        """
        try:
            self.model = ARIMA(series, order=self.order)
            self.fitted = self.model.fit()
        except Exception as e:
            raise RuntimeError(f"ARIMA fitting failed: {e}")

        return self.fitted

    def forecast(self, steps: int = 24) -> pd.Series:
        """
        Forecast future points using the fitted ARIMA model.

        Parameters
        ----------
        steps : int
            Number of future time periods to predict.

        Returns
        -------
        pd.Series
        """
        if self.fitted is None:
            raise RuntimeError("Model must be fitted before forecasting.")

        return self.fitted.forecast(steps)


# ===============================================================
# 3. SARIMA Model Class
# ===============================================================

class SARIMAModel:
    """
    Wrapper for SARIMA/SARIMAX modeling.

    Optimized for traffic/weather datasets and long 15-minute series.

    Parameters
    ----------
    order : tuple
        ARIMA(p,d,q) specification.
    seasonal_order : tuple
        Seasonal (P,D,Q,m) specification.
    exog : pd.DataFrame or None
        Optional exogenous variables (e.g., weather features).
    """

    def __init__(
        self,
        order=(1, 1, 1),
        seasonal_order=(0, 1, 1, 96),  # 96 = daily seasonality for 15-min data
        exog=None
    ):
        self.order = order
        self.seasonal_order = seasonal_order
        self.exog = exog
        self.model = None
        self.fitted = None

    def fit(self, series: pd.Series):
        """
        Fit the SARIMA model.

        Parameters
        ----------
        series : pd.Series
            Time series to model.

        Returns
        -------
        SARIMAXResults
        """

        try:
            self.model = SARIMAX(
                series,
                order=self.order,
                seasonal_order=self.seasonal_order,
                exog=self.exog,
                enforce_stationarity=False,
                enforce_invertibility=False,
                simple_differencing=True,
                concentrate_scale=True
            )
            self.fitted = self.model.fit(disp=False)

        except Exception as e:
            raise RuntimeError(f"SARIMA fitting failed: {e}")

        return self.fitted

    def forecast(self, steps: int = 96, exog_future=None) -> pd.Series:
        """
        Forecast future values from the fitted SARIMA model.

        Parameters
        ----------
        steps : int
            Number of future steps to predict.
        exog_future : pd.DataFrame, optional
            Future exogenous variables matching model spec.

        Returns
        -------
        pd.Series
        """
        if self.fitted is None:
            raise RuntimeError("Model must be fitted before forecasting.")

        forecast_obj = self.fitted.get_forecast(steps=steps, exog=exog_future)
        return forecast_obj.predicted_mean
