import yfinance as yf
import pandas as pd


class YahooFinanceDataFetcher:
    def __init__(self, default_ticker="^VIX"):
        self.default_ticker = default_ticker

    def download_data(self, ticker=None, start=None, end=None):
        """
        Download data for the specified ticker and date range.
        If ticker is None, uses default ticker.
        """
        try:
            ticker_to_use = ticker if ticker is not None else self.default_ticker

            if ticker_to_use is None:
                raise ValueError("No ticker specified and no default ticker set")

            data = yf.download(ticker_to_use, start=start, end=end, progress=False, auto_adjust=False)

            # If the columns are MultiIndex, flatten them
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)

            return data

        except Exception as e:
            print(f"Failed to download data: {e}")
            return None