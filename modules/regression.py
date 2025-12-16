"""
Traffic Regression Module.

This module encapsulates the machine learning logic for traffic volume prediction
using linear models. It handles the end-to-end regression pipeline, including:
1.  Data Preprocessing: Handling missing values, scaling, and one-hot encoding.
2.  Feature Engineering: Creating interaction terms (e.g., Weekly Profiles).
3.  Model Training: Implementing Linear, Ridge, and Lasso regression.
4.  Hyperparameter Tuning: Using Grid Search with Time Series Cross-Validation.
5.  Evaluation: Calculating RMSE, MAE, and R2 scores with post-processing (negative clipping).

Dependencies:
    - pandas
    - numpy
    - sklearn (model_selection, linear_model, preprocessing, compose, impute, pipeline, metrics)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class TrafficRegressor:
    """
    A unified interface for training, tuning, and evaluating traffic regression models.

    This class standardizes the regression workflow. It manages dataset preparation,
    constructs scikit-learn pipelines for various algorithms, and executes rigorous
    evaluation strategies suitable for time-series data.

    Attributes:
        df (pd.DataFrame): The preprocessed training dataset.
        target_col (str): The name of the target variable column (default: 'total_volume').
        models (dict): A dictionary storing trained model objects (keys: model names).
        X (pd.DataFrame): The feature matrix.
        y (pd.Series): The target vector.
        cv_split (TimeSeriesSplit): The cross-validation splitting strategy.
    """

    def __init__(self, df: pd.DataFrame, target_col: str = 'total_volume'):
        """
        Initializes the TrafficRegressor and performs initial feature engineering.

        The initialization process includes:
        1. Dropping rows where the target variable is missing.
        2. Creating a 'weekly_profile' interaction feature (Day + Hour + School Status).
        3. Separating features (X) and target (y).
        4. Setting up a TimeSeriesSplit for validation.

        Args:
            df (pd.DataFrame): The input dataframe containing traffic data and features.
            target_col (str, optional): The name of the target column to predict. 
                                        Defaults to 'total_volume'.
        """
        self.df = df.copy()
        self.target_col = target_col
        self.models = {}
        
        # 1. CLEANING
        self.df.dropna(subset=[target_col], inplace=True)

        # 2. FEATURE ENGINEERING: SUPER-INTERACTION
        # This combines Day + Hour + School Status into one powerful feature.
        if 'day_of_week' in self.df.columns and 'hour' in self.df.columns:
            if 'day_type' in self.df.columns:
                # "Monday_08_0" (Term) vs "Monday_08_1" (Holiday)
                self.df['weekly_profile'] = (
                    self.df['day_of_week'].astype(str) + "_" + 
                    self.df['hour'].astype(str) + "_" + 
                    self.df['day_type'].astype(str)
                )
            else:
                # Fallback for Blind Test (No day_type)
                self.df['weekly_profile'] = (
                    self.df['day_of_week'].astype(str) + "_" + 
                    self.df['hour'].astype(str)
                )

        # 3. SEPARATION
        self.X = self.df.drop(columns=[target_col])
        self.y = self.df[target_col]
        
        # 4. VALIDATION
        self.cv_split = TimeSeriesSplit(n_splits=5)

    def _get_preprocessor(self, aggressive: bool = False) -> ColumnTransformer:
        """
        Constructs a scikit-learn ColumnTransformer for data preprocessing.

        This method dynamically selects features based on the 'aggressive' flag.
        - Standard Mode: Uses separate 'hour', 'day_of_week', and 'day_type'.
        - Aggressive Mode: Uses the combined 'weekly_profile' interaction feature 
          for higher granularity (often yields better R2 scores).

        Args:
            aggressive (bool, optional): If True, uses the 'weekly_profile' interaction 
                                         feature instead of individual time components. 
                                         Defaults to False.

        Returns:
            ColumnTransformer: A configured preprocessor object ready for a Pipeline.
        """
        # 1. Determine Categorical Strategy
        if aggressive and 'weekly_profile' in self.X.columns:
            target_cats = ['weekly_profile'] # The "God Mode" Feature
        else:
            target_cats = ['hour', 'day_of_week', 'day_type']

        # 2. Filter existing columns
        categorical_cols = [c for c in target_cats if c in self.X.columns]
        numerical_cols = [c for c in self.X.select_dtypes(include=np.number).columns if c not in categorical_cols]
        
        # Transformers
        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numerical_cols),
                ('cat', categorical_transformer, categorical_cols)
            ]
        )
        return preprocessor

    def run_linear_baseline(self):
        """
        Trains a baseline Linear Regression model.

        Uses the standard feature set (non-aggressive preprocessing). The result
        is stored in the `self.models` dictionary under the key 'Linear'.
        """
        print(f"Regression: Training Baseline Linear Model...")
        pipeline = Pipeline([
            ('preprocessor', self._get_preprocessor(aggressive=False)),
            ('model', LinearRegression())
        ])
        pipeline.fit(self.X, self.y)
        self.models['Linear'] = pipeline
        print("Regression: Baseline Complete.")

    def run_ridge_tuning(self):
        """
        Trains and tunes a Ridge Regression model using GridSearchCV.

        Uses the 'Aggressive' preprocessing strategy (Weekly Interaction Profile)
        to capture complex time-based patterns. It optimizes the 'alpha' regularization
        parameter to prevent overfitting.
        """
        print("Regression: Tuning Ridge Model (Weekly Interaction Strategy)...")
        
        pipeline = Pipeline([
            ('preprocessor', self._get_preprocessor(aggressive=True)),
            ('model', Ridge())
        ])
        
        # Expanded Grid to find the absolute perfect Alpha
        param_grid = {
            'model__alpha': [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
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
        print(f"Regression: Ridge Tuned. Best Alpha: {grid.best_params_['model__alpha']}")

    def run_lasso_tuning(self):
        """
        Trains and tunes a Lasso Regression model for feature selection.

        Lasso (L1 regularization) drives coefficients of less important features to zero.
        This method operates only on numerical features to provide interpretability regarding
        physical variables (e.g., weather, speed).
        """
        print("Regression: Tuning Lasso Model...")
        pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('model', Lasso(max_iter=3000, tol=1e-3))
        ])
        
        X_num = self.X.select_dtypes(include=np.number)
        param_grid = {'model__alpha': [0.1, 1.0, 5.0, 10.0]}
        
        grid = GridSearchCV(
            pipeline, param_grid, cv=self.cv_split, 
            scoring='neg_root_mean_squared_error', n_jobs=-1
        )
        
        grid.fit(X_num, self.y)
        self.models['Lasso'] = grid.best_estimator_
        print(f"Regression: Lasso Tuned. Best Alpha: {grid.best_params_['model__alpha']}")

    def evaluate(self) -> pd.DataFrame:
        """
        Evaluates all trained models and returns a performance summary.

        Applies 'Negative Clipping' to predictions (forcing values >= 0) to ensure
        physical realism before calculating metrics.

        Returns:
            pd.DataFrame: A table containing 'RMSE', 'MAE', and 'R2' for each model.
        """
        results = []
        for name, model in self.models.items():
            if name == 'Lasso':
                X_eval = self.X.select_dtypes(include=np.number)
            else:
                X_eval = self.X
            
            y_pred = model.predict(X_eval)
            
            # CLIP NEGATIVES: Forces prediction >= 0
            y_pred = np.maximum(y_pred, 0)
            
            rmse = np.sqrt(mean_squared_error(self.y, y_pred))
            mae = mean_absolute_error(self.y, y_pred)
            r2 = r2_score(self.y, y_pred)
            results.append({'Model': name, 'RMSE': round(rmse, 2), 'MAE': round(mae, 2), 'R2': round(r2, 4)})
        return pd.DataFrame(results)

    def get_feature_importance(self) -> pd.Series:
        """
        Extracts non-zero coefficients from the trained Lasso model.

        Useful for identifying which numerical features (e.g., temperature, wind)
        have the strongest impact on traffic volume.

        Returns:
            pd.Series: Sorted series of feature coefficients. Returns a string message
                       if the Lasso model has not been trained.
        """
        if 'Lasso' not in self.models: return "Lasso model not trained."
        model = self.models['Lasso'].named_steps['model']
        X_num = self.X.select_dtypes(include=np.number)
        coefs = pd.Series(model.coef_, index=X_num.columns)
        return coefs[coefs != 0].sort_values(key=abs, ascending=False)