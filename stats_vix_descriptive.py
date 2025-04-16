class VixStats:
    def __init__(self, data):

        self.data = data

        # Stats attributes
        self.stats_row_count = None
        self.stats_current = None
        self.stats_mean = None
        self.stats_median = None
        self.stats_min = None
        self.stats_max = None
        self.stats_mode = None
        self.stats_std_dev = None
        # Percentiles
        self.stats_percentile_25 = None
        self.stats_percentile_50 = None
        self.stats_percentile_75 = None
        self.stats_percentile_90 = None
        self.stats_percentile_95 = None

        self.stats_rolling_mean_7 = None
        self.stats_rolling_mean_30 = None
        self.rsi = None

        self.update_stats()

    def update_stats(self):
        try:
            if self.data is not None and not self.data.empty:
                close_prices = self.data["Close"]

                self.stats_row_count = len(close_prices)
                self.stats_current = round(close_prices.iloc[-1], 2)
                self.stats_mean = round(close_prices.mean(), 2)
                self.stats_median = round(close_prices.median(), 2)
                self.stats_mode = round(close_prices.mode().iloc[0], 2)
                self.stats_min = round(close_prices.min(), 2)
                self.stats_max = round(close_prices.max(), 2)
                self.stats_std_dev = round(close_prices.std(), 2)

                self.stats_percentile_25 = round(close_prices.quantile(0.25), 2)
                self.stats_percentile_50 = round(close_prices.quantile(0.50), 2)
                self.stats_percentile_75 = round(close_prices.quantile(0.75), 2)
                self.stats_percentile_90 = round(close_prices.quantile(0.90), 2)
                self.stats_percentile_95 = round(close_prices.quantile(0.95), 2)

                self.stats_rolling_mean_7 = round(close_prices.rolling(window=7).mean().iloc[-1], 2)
                self.stats_rolling_mean_30 = round(close_prices.rolling(window=30).mean().iloc[-1], 2)

                self.ta_calculate_rsi(close_prices)

        except Exception as e:
            print(f"[VixStats] Failed to update stats: {e}")

    def ta_calculate_rsi(self, close_prices, period=14):
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
        return self.stats_row_count

    @property
    def get_current(self):
        return self.stats_current

    @property
    def get_mean(self):
        return self.stats_mean

    @property
    def get_median(self):
        return self.stats_median

    @property
    def get_mode(self):
        return self.stats_mode

    @property
    def get_min(self):
        return self.stats_min

    @property
    def get_max(self):
        return self.stats_max

    @property
    def get_std_dev(self):
        return self.stats_std_dev

    @property
    def get_z_score(self):
        if None not in (self.stats_current, self.stats_mean, self.stats_std_dev):
            return round((self.stats_current - self.stats_mean) / self.stats_std_dev, 2)
        return None

    @property
    def get_percentile_25(self):
        return self.stats_percentile_25
    @property
    def get_percentile_50(self):
        return self.stats_percentile_50

    @property
    def get_percentile_75(self):
        return self.stats_percentile_75

    @property
    def get_percentile_90(self):
        return self.stats_percentile_90

    @property
    def get_percentile_95(self):
        return self.stats_percentile_95

    @property
    def current_percentile(self):
        if self.data is not None and not self.data.empty and self.stats_current is not None:
            close_prices = self.data["Close"]
            percentile = (close_prices < self.stats_current).mean() * 100
            return round(percentile, 1)
        return None

    @property
    def get_rolling_mean7(self):
        return self.stats_rolling_mean_7

    @property
    def get_rolling_mean30(self):
        return self.stats_rolling_mean_30

    @property
    def get_ta_rsi(self):
        return self.rsi
