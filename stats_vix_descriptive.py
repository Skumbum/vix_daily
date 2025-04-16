class VixStats:
    def __init__(self, data):
        self.data = data

        # Stats attributes
        self.row_count = None
        self.current_vix = None
        self.mean_close = None
        self.median_close = None
        self.mode_close = None
        self.std_dev_close = None

        self.percentile_25 = None
        self.percentile_50 = None
        self.percentile_75 = None
        self.percentile_90 = None
        self.percentile_95 = None

        self.rolling_mean_7 = None
        self.rolling_mean_30 = None
        self.rsi = None

        self.update_stats()

    def update_stats(self):
        try:
            if self.data is not None and not self.data.empty:
                close_prices = self.data["Close"]

                self.row_count = len(close_prices)
                self.current_vix = round(close_prices.iloc[-1], 2)
                self.mean_close = round(close_prices.mean(), 2)
                self.median_close = round(close_prices.median(), 2)
                self.mode_close = round(close_prices.mode().iloc[0], 2)
                self.std_dev_close = round(close_prices.std(), 2)

                self.percentile_25 = round(close_prices.quantile(0.25), 2)
                self.percentile_50 = round(close_prices.quantile(0.50), 2)
                self.percentile_75 = round(close_prices.quantile(0.75), 2)
                self.percentile_90 = round(close_prices.quantile(0.90), 2)
                self.percentile_95 = round(close_prices.quantile(0.95), 2)

                self.rolling_mean_7 = round(close_prices.rolling(window=7).mean().iloc[-1], 2)
                self.rolling_mean_30 = round(close_prices.rolling(window=30).mean().iloc[-1], 2)

                self.calculate_rsi(close_prices)

        except Exception as e:
            print(f"[VixStats] Failed to update stats: {e}")

    def calculate_rsi(self, close_prices, period=14):
        delta = close_prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        self.rsi = round(rsi.iloc[-1], 2) if not rsi.empty else None

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
        if None not in (self.current_vix, self.mean_close, self.std_dev_close):
            return round((self.current_vix - self.mean_close) / self.std_dev_close, 2)
        return None

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
        if self.data is not None and not self.data.empty and self.current_vix is not None:
            close_prices = self.data["Close"]
            percentile = (close_prices < self.current_vix).mean() * 100
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
        return self.rsi
