import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class TrafficRegressor:
    """
    Handles training, tuning, and evaluation of Regression models for traffic prediction.
    
    This class manages the full regression pipeline, including:
    - Data cleaning (dropping missing targets).
    - Feature imputation (handling missing values).
    - Pipeline construction (Scaling + Modelling).
    - Hyperparameter tuning via GridSearch.
    """

    def __init__(self, df, target_col='total_volume'):
        """
        Initializes the Regressor, separates features (X) and target (y), and handles missing targets.

        Args:
            df (pd.DataFrame): The merged dataframe containing traffic, weather, and time features.
            target_col (str): The name of the target column to predict. Defaults to 'total_volume'.

        Returns:
            None
        """
        self.df = df.copy()
        self.target_col = target_col
        self.models = {}
        self.best_params = {}
        
        # 1. CLEANING: Drop rows where the TARGET is missing
        initial_count = len(self.df)
        self.df.dropna(subset=[target_col], inplace=True)
        dropped = initial_count - len(self.df)
        if dropped > 0:
            print(f"[Regression] Dropped {dropped} rows with missing Target ('{target_col}').")

        # 2. SEPARATION: Define features (X) and target (y)
        self.X = self.df.drop(columns=[target_col]).select_dtypes(include=[np.number])
        self.y = self.df[target_col]
        
        # 3. VALIDATION: Use TimeSeriesSplit
        # Strict requirement for Time Series data to avoid leakage
        self.cv_split = TimeSeriesSplit(n_splits=5)

    def run_linear_baseline(self):
        """
        Trains a standard Linear Regression model (OLS) using a pipeline.
        
        The pipeline includes:
        1. SimpleImputer (Mean): Handles missing feature values.
        2. StandardScaler: Normalizes features for optimal performance.
        3. LinearRegression: The estimator.

        Args:
            None

        Returns:
            None: The trained model is stored in self.models['Linear'].
        """
        print(f"Regression: Training Baseline Linear Model on {len(self.X)} rows...")
        
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')), 
            ('scaler', StandardScaler()),
            ('model', LinearRegression())
        ])
        
        pipeline.fit(self.X, self.y)
        self.models['Linear'] = pipeline
        print("Regression: Baseline Complete.")

    def run_ridge_tuning(self):
        """
        Trains a Ridge Regression model (L2 Regularization) using GridSearchCV.
        
        Optimizes the 'alpha' hyperparameter using TimeSeriesSplit cross-validation.

        Args:
            None

        Returns:
            None: The best estimator is stored in self.models['Ridge'].
                  Best parameters are stored in self.best_params['Ridge'].
        """
        print("Regression: Tuning Ridge Model...")
        
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('model', Ridge())
        ])
        
        param_grid = {
            'model__alpha': [0.1, 1.0, 10.0, 50.0, 100.0]
        }
        
        grid = GridSearchCV(
            pipeline, 
            param_grid, 
            cv=self.cv_split, 
            scoring='neg_root_mean_squared_error',
            n_jobs=-1
        )
        
        grid.fit(self.X, self.y)
        
        self.models['Ridge'] = grid.best_estimator_
        self.best_params['Ridge'] = grid.best_params_
        print(f"Regression: Ridge Tuned. Best Alpha: {grid.best_params_['model__alpha']}")

    def run_lasso_tuning(self):
        """
        Trains a Lasso Regression model (L1 Regularization) using GridSearchCV.
        
        Useful for feature selection as it forces irrelevant feature coefficients to zero.
        Optimizes 'alpha' using TimeSeriesSplit cross-validation.

        Args:
            None

        Returns:
            None: The best estimator is stored in self.models['Lasso'].
                  Best parameters are stored in self.best_params['Lasso'].
        """
        print("Regression: Tuning Lasso Model...")
        
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('model', Lasso(max_iter=3000, tol=1e-3))
        ])
        
        param_grid = {
            'model__alpha': [0.01, 0.1, 1.0, 5.0, 10.0]
        }
        
        grid = GridSearchCV(
            pipeline, 
            param_grid, 
            cv=self.cv_split, 
            scoring='neg_root_mean_squared_error',
            n_jobs=-1
        )
        
        grid.fit(self.X, self.y)
        
        self.models['Lasso'] = grid.best_estimator_
        self.best_params['Lasso'] = grid.best_params_
        print(f"Regression: Lasso Tuned. Best Alpha: {grid.best_params_['model__alpha']}")

    def evaluate(self):
        """
        Evaluates all trained models against the training set using RMSE, MAE, and R2.

        Args:
            None

        Returns:
            pd.DataFrame: A summary table containing the metrics for every model in self.models.
        """
        results = []
        
        for name, model in self.models.items():
            y_pred = model.predict(self.X)
            
            rmse = np.sqrt(mean_squared_error(self.y, y_pred))
            mae = mean_absolute_error(self.y, y_pred)
            r2 = r2_score(self.y, y_pred)
            
            results.append({
                'Model': name,
                'RMSE': round(rmse, 2),
                'MAE': round(mae, 2),
                'R2': round(r2, 4)
            })
            
        return pd.DataFrame(results)

    def get_feature_importance(self):
        """
        Extracts feature coefficients from the Lasso model to identify key drivers of traffic.

        Args:
            None

        Returns:
            pd.Series: A sorted series of non-zero coefficients (feature importance).
            str: Error message if the Lasso model has not been trained yet.
        """
        if 'Lasso' not in self.models:
            return "Lasso model not trained."
            
        lasso_model = self.models['Lasso'].named_steps['model']
        
        coefs = pd.Series(lasso_model.coef_, index=self.X.columns)
        return coefs[coefs != 0].sort_values(key=abs, ascending=False)