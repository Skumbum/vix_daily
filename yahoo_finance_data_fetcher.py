#yahoo_finance_data_fetcher.py
import yfinance as yf

class YahooFinanceDataFetcher:
    def __init__(self, default_ticker="^VIX"):
        self.default_ticker = default_ticker

    def download_data(self, ticker=None, start=None, end=None):
        """
        Download data for the specified ticker and date range.
        If ticker is None, uses default ticker.
        """
        try:
            # Use provided ticker or fall back to default
            ticker_to_use = ticker if ticker is not None else self.default_ticker

            # Verify we have a ticker to use
            if ticker_to_use is None:
                raise ValueError("No ticker specified and no default ticker set")

            data = yf.download(ticker_to_use, start=start, end=end, progress=False, auto_adjust=False)
            return data
        except Exception as e:
            print(f"Failed to download data: {e}")
            return None