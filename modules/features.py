import pandas as pd
from datetime import datetime, timedelta

class FeatureEngineer:
    """
    Applies feature engineering to the traffic DataFrame.
    Incorporates Ahmed's holiday logic but adapted for in-memory DataFrames.
    """

    def __init__(self):
        # raw holiday strings
        self.HOLIDAYS_RAW = [
            "2021-02-15 till 2021-02-19", "2021-04-01 till 2021-04-16", "2021-05-03",
            "2021-05-31 till 2021-06-04", "2021-07-22 till 2021-08-31", "2021-08-02 till 2021-08-31",
            "2021-10-25 till 2021-10-29", "2021-12-20 till 2022-01-03", "2022-02-21 till 2022-02-25",
            "2022-04-11 till 2022-04-22", "2022-05-02", "2022-05-30 till 2022-06-03",
            "2022-07-21 till 2022-07-29", "2022-08-01 till 2022-09-01", "2022-10-24 till 2022-10-28",
            "2022-12-21 till 2023-01-03", "2023-02-13 till 2023-02-17", "2023-04-03 till 2023-04-14",
            "2023-05-01", "2023-05-29 till 2023-06-02", "2023-07-24 till 2023-07-31",
            "2023-08-01 till 2023-08-31", "2023-10-23 till 2023-10-27", "2023-12-21 till 2024-01-05",
            "2024-02-12 till 2024-02-16", "2024-03-29 till 2024-04-12", "2024-05-06",
            "2024-05-27 till 2024-05-31", "2024-07-25 till 2024-09-02", "2024-10-28 till 2024-11-1",
            "2024-12-23 till 2025-01-3", "2025-02-12 till 2025-02-21", "2025-04-07 till 2025-04-21",
            "2025-05-05", "2025-05-26 till 2025-12-30", "2025-07-24 till 2025-09-02"
        ]
        self.holiday_set = self._generate_holiday_set()

    def _generate_holiday_set(self):
        """Parses Ahmed's string format into a set of date objects."""
        holiday_set = set()
        for h in self.HOLIDAYS_RAW:
            if "till" in h:
                start_str, end_str = h.split(" till ")
                start = datetime.strptime(start_str, "%Y-%m-%d").date()
                end = datetime.strptime(end_str, "%Y-%m-%d").date()
                current = start
                while current <= end:
                    holiday_set.add(current)
                    current += timedelta(days=1)
            else:
                holiday_set.add(datetime.strptime(h, "%Y-%m-%d").date())
        return holiday_set

    def _get_day_type(self, timestamp):
        """
        Returns 1 if Holiday/Weekend, 0 if School Term.
        """
        date_obj = timestamp.date()
        
        # Check Weekend (Sat=5, Sun=6)
        if date_obj.weekday() >= 5:
            return 1 # Off Day
        
        # Check Holiday List
        if date_obj in self.holiday_set:
            return 1 # Off Day
            
        return 0 # School Term

    def add_features(self, df):
        """
        Takes the merged dataframe and adds time features + day_type.
        """
        df = df.copy()
        
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # 1. Standard Time Features
        df['hour'] = df.index.hour
        df['month'] = df.index.month
        df['day_of_week'] = df.index.dayofweek
        
        # 2. Apply Ahmed's Logic (The Twist)
        # We create 'day_type' (0 or 1)
        df['day_type'] = df.index.to_series().apply(self._get_day_type)
        
        return df