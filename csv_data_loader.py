import pandas as pd


class CSVDataLoader:
    def __init__(self, filepath, start=None, end=None):
        """
        Initialize the CSVDataLoader with the path to the CSV file and optional date range.

        Args:
            filepath (str): Path to the CSV file (e.g., 'VIX_History (1).csv')
            start (str, optional): Start date for filtering (e.g., '1990-01-01')
            end (str, optional): End date for filtering (e.g., '2025-04-21')
        """
        self.filepath = filepath
        self.start = pd.to_datetime(start) if start else None
        self.end = pd.to_datetime(end) if end else None
        self.data = None

    def load_data(self):
        """
        Load the VIX data from the CSV file, rename columns, ensure correct data types,
        and add dummy columns for compatibility with Yahoo Finance data structure.

        Returns:
            pd.DataFrame: Loaded and processed DataFrame with columns 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'
        """
        # Load the CSV file
        self.data = pd.read_csv(self.filepath)

        # Rename columns to match expected format
        self.data = self.data.rename(columns={
            'DATE': 'Date',
            'OPEN': 'Open',
            'HIGH': 'High',
            'LOW': 'Low',
            'CLOSE': 'Close'
        })

        # Ensure numeric columns are of float type
        numeric_columns = ['Open', 'High', 'Low', 'Close']
        for col in numeric_columns:
            self.data[col] = pd.to_numeric(self.data[col], errors='coerce')

        # Convert Date column to datetime and set as index
        self.data['Date'] = pd.to_datetime(self.data['Date'])
        self.data = self.data.set_index('Date')

        # Drop any rows with NaN values in critical columns
        self.data = self.data.dropna(subset=['Open', 'High', 'Low', 'Close'])

        # Add dummy columns to match Yahoo Finance structure
        self.data['Adj Close'] = self.data['Close']  # VIX doesn't adjust, so use Close
        self.data['Volume'] = 0  # VIX has no volume, set to 0

        # Filter by date range if specified
        if self.start:
            self.data = self.data[self.data.index >= self.start]
        if self.end:
            self.data = self.data[self.data.index <= self.end]

        return self.data