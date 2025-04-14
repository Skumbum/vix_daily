#stats_vix.py
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