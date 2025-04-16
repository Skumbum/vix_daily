#stats_vix_descriptive.py
import yfinance as yf

class VixStats:
    def __init__(self, start: str = None, end: str = None):
        self.ticker = "^VIX"
        self.start = start
        self.end = end
        self.data = None

        #Stats attributes
        self.row_count = None
        self.current_vix = None
        self.mean_close = None
        self.median_close = None
        self.mode_close = None
        self.mode_count_close = None
        self.std_dev_close = None
        self.z_score_close = None

        self.percentile_25 = None
        self.percentile_50 = None  # This is the same as median
        self.percentile_75 = None
        self.percentile_90 = None
        self.percentile_95 = None

        self.rolling_mean_7 = None
        self.rolling_mean_30 = None
        self.rsi = None

    def download_data(self):
        try:
            self.data = yf.download(self.ticker, start=self.start, end=self.end, progress=False, auto_adjust=False)

            # Update statistics attributes if download was successful
            if not self.data.empty:
                self.row_count = len(self.data["Close"])
                self.current_vix = round(self.data["Close"].iloc[-1].item(), 2)
                self.mean_close = round(self.data["Close"].mean().item(), 2)
                self.median_close = round(self.data["Close"].median().item(), 2)
                self.mode_close = round(self.data["Close"]["^VIX"].mode().iloc[0], 2)
                self.std_dev_close = round(self.data["Close"].std().item(), 2)

                self.percentile_25 = round(self.data[('Close', '^VIX')].quantile(0.25), 2)
                self.percentile_50 = round(self.data[('Close', '^VIX')].quantile(0.50), 2)  # Same as median
                self.percentile_75 = round(self.data[('Close', '^VIX')].quantile(0.75), 2)
                self.percentile_90 = round(self.data[('Close', '^VIX')].quantile(0.90), 2)
                self.percentile_95 = round(self.data[('Close', '^VIX')].quantile(0.95), 2)

                # Rolling Stats
                self.rolling_mean_7 = round(self.data[('Close', '^VIX')].rolling(window=7).mean().iloc[-1], 2)
                self.rolling_mean_30 = round(self.data[('Close', '^VIX')].rolling(window=30).mean().iloc[-1], 2)

        except Exception as e:
            print(f"Failed to download data: {e}")

    @property
    def get_row_count(self):
        return self.row_count

    @property
    def get_current_vix(self):
        return self.current_vix

    @property
    def get_mean(self):
        return self.mean_close

    @property
    def get_median(self):
        return self.median_close
    @property
    def get_mode(self):
        return self.mode_close

    @property
    def get_std_dev(self):
        return self.std_dev_close

    @property
    def get_z_score(self):
        return round(((self.current_vix - self.mean_close) / self.std_dev_close), 2)

    @property
    def get_percentile_25(self):
        return self.percentile_25

    @property
    def get_percentile_75(self):
        return self.percentile_75

    @property
    def get_percentile_90(self):
        return self.percentile_90

    @property
    def get_percentile_95(self):
        return self.percentile_95

    @property
    def current_percentile(self):
        """Returns the percentile of the current VIX value in the historical distribution"""
        if self.data is not None and not self.data.empty:
            # Calculate what percentage of values are below the current VIX
            percentile = (self.data[('Close', '^VIX')] < self.current_vix).mean() * 100
            return round(percentile, 1)
        return None

    @property
    def get_rolling_mean7(self):
        return self.rolling_mean_7

    @property
    def get_rolling_mean30(self):
        return self.rolling_mean_30

    @property
    def get_rsi(self):
        delta = self.data["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi_temp = 100 - (100 / (1 + rs))
        self.rsi = round(rsi_temp.iloc[-1], 2)
        return self.rsi