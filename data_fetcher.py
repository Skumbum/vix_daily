import pandas as pd
import yfinance as yf
import os
from datetime import datetime, timedelta

class DataFetcher:
    def __init__(self, data_dir: str = "D:\\PyCharmProjects\\vix_daily", min_rows: int = 100):
        self.data_dir = data_dir
        self.data_file = os.path.join(data_dir, "vix_data.csv")
        self.ticker = "^VIX"
        self.start_date = "1985-01-01"
        self.end_date = "2025-04-18"  # Fetch up to 2025-04-17 + 1 day to ensure inclusion
        self.min_rows = min_rows
        self.max_retries = 3  # Limit for retrying fetch due to duplicates

    def fetch_vix_data(self, retry_count=0):
        """
        Fetch VIX data from Yahoo Finance or load from local file if it exists and has enough rows.
        Returns a pandas DataFrame with 'Date' and 'Close' columns.
        """
        # Check if local file exists and is usable
        if os.path.exists(self.data_file):
            print(f"Loading VIX data from {self.data_file}")
            data = pd.read_csv(self.data_file, parse_dates=['Date'])
            if len(data) >= self.min_rows and data['Date'].max() >= pd.to_datetime('2025-04-17'):
                print(f"Found {len(data)} rows in local file, which meets the minimum requirement of {self.min_rows}")
                return data
            else:
                print(f"Local file has only {len(data)} rows (minimum required: {self.min_rows}) or doesn't include 2025-04-17. Refetching...")

        # Fetch from Yahoo Finance
        print(f"Fetching VIX data from Yahoo Finance ({self.start_date} to {self.end_date})...")
        try:
            vix = yf.download(self.ticker, start=self.start_date, end=self.end_date, auto_adjust=True)
            if vix.empty:
                raise ValueError("No data retrieved from Yahoo Finance")
            print(f"Retrieved {len(vix)} rows from Yahoo Finance")

            # Reset index to get 'Date' as a column
            vix = vix[['Close']].reset_index()

            # Ensure 'Date' is datetime
            vix['Date'] = pd.to_datetime(vix['Date'])

            # Check for duplicates in the raw fetched data
            if vix['Date'].duplicated().any():
                duplicate_count = vix['Date'].duplicated().sum()
                print(f"Warning: Found {duplicate_count} duplicate dates in raw fetched data.")
                duplicated_dates = vix[vix['Date'].duplicated(keep=False)]['Date'].value_counts()
                print(f"Most common duplicated dates:\n{duplicated_dates.head()}")

            # Append the hardcoded row for 2025-04-17 if not already present
            if vix['Date'].max() < pd.to_datetime('2025-04-17'):
                new_row = pd.DataFrame({'Date': [pd.to_datetime('2025-04-17')], 'Close': [29.65]})
                vix = pd.concat([vix, new_row], ignore_index=True)
                print("Appended 2025-04-17 data point")

            # Save to file
            vix.to_csv(self.data_file, index=False)
            print(f"VIX data saved to {self.data_file} with {len(vix)} rows")
            return vix

        except Exception as e:
            print(f"Error fetching VIX data: {e}")
            if retry_count >= self.max_retries:
                print("Max retries reached. Using placeholder dataset as fallback...")
                vix = pd.DataFrame({
                    'Date': [
                        pd.to_datetime('2025-04-15'),
                        pd.to_datetime('2025-04-16'),
                        pd.to_datetime('2025-04-17')
                    ],
                    'Close': [33.60, 31.20, 29.65]
                })
                vix.to_csv(self.data_file, index=False)
                print(f"Placeholder data saved to {self.data_file}")
                return vix
            else:
                print(f"Retrying fetch (attempt {retry_count + 1}/{self.max_retries})...")
                return self.fetch_vix_data(retry_count + 1)

    def get_data(self):
        """Return the VIX data as a pandas DataFrame with 'Date' as index."""
        data = self.fetch_vix_data()
        print(f"After fetch_vix_data: {len(data)} rows")

        # Check for duplicates in 'Date'
        if data['Date'].duplicated().any():
            duplicate_count = data['Date'].duplicated().sum()
            print(f"Warning: Found {duplicate_count} duplicate dates.")
            duplicated_dates = data[data['Date'].duplicated(keep=False)]['Date'].value_counts()
            print(f"Most common duplicated dates:\n{duplicated_dates.head()}")
            # Only drop duplicates if it won't result in too few rows
            data_no_duplicates = data.drop_duplicates(subset=['Date'], keep='last')
            if len(data_no_duplicates) >= self.min_rows:
                print(f"Dropping duplicates, keeping last occurrence. After dropping duplicates: {len(data_no_duplicates)} rows")
                data = data_no_duplicates
            else:
                print(f"Not dropping duplicates as it would leave only {len(data_no_duplicates)} rows (minimum required: {self.min_rows}).")
                if os.path.exists(self.data_file):
                    os.remove(self.data_file)  # Delete the problematic file
                print("Refetching...")
                return self.fetch_vix_data()  # Fetch fresh data, but avoid infinite loop

        # Convert 'Date' to datetime
        data['Date'] = pd.to_datetime(data['Date'])

        # Check for missing values in 'Close'
        if data['Close'].isna().any():
            missing_count = data['Close'].isna().sum()
            print(f"Warning: Found {missing_count} missing values in 'Close'. Dropping NaNs...")
            data = data.dropna(subset=['Close'])
            print(f"After dropping NaNs: {len(data)} rows")

        # Set index
        data = data.set_index('Date')
        print(f"After setting index: {len(data)} rows")

        if len(data) < 2:
            raise ValueError(f"Insufficient data: only {len(data)} row(s) available. Need at least 2 rows for analysis.")
        return data