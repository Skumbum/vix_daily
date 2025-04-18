import numpy as np
import pandas as pd


class EmpiricalStatsDescriptive:
    def __init__(self, data):
        self.data = data
        # For simplicity, we'll use the 'Close' column for calculations if it exists
        if isinstance(data, pd.DataFrame) and 'Close' in data.columns:
            self.series = data['Close']
        elif isinstance(data, pd.DataFrame):
            self.series = data.iloc[:, 0]  # Use first column if 'Close' doesn't exist
        else:
            self.series = data  # Assume it's already a Series

        self.update_stats()

    def update_stats(self):
        # Compute basic stats
        self.mean = self.series.mean()
        self.std_dev = self.series.std()
        self.max = self.series.max()
        self.min = self.series.min()
        self.median = self.series.median()
        self.mode = self.series.mode().iloc[0] if not self.series.mode().empty else None
        self.variance = self.series.var()

        # Manually calculate MAD
        self.mad = self.calculate_mad(self.series)

        # Make sure we use scalar values for the z-score calculation
        current_value = self.series.iloc[-1]  # Get the most recent data point (last value)
        self.z_score = (current_value - self.mean) / self.std_dev if self.std_dev != 0 else None

        self.geometric_mean = self.calculate_geometric_mean(self.series)
        self.harmonic_mean = self.calculate_harmonic_mean(self.series)
        self.skewness = self.series.skew()
        self.kurtosis = self.series.kurt()
        self.iqr = self.series.quantile(0.75) - self.series.quantile(0.25)
        self.range = self.max - self.min
        self.cv = self.std_dev / self.mean if self.mean != 0 else None

    def calculate_mad(self, series):
        return np.mean(np.abs(series - series.mean()))

    def calculate_geometric_mean(self, data):
        # Filter out non-positive values to avoid errors with logarithms
        positive_data = data[data > 0]
        if len(positive_data) == 0:
            return None
        # Use logarithms to avoid overflow
        log_data = np.log(positive_data)
        return np.exp(np.mean(log_data))

    def calculate_harmonic_mean(self, data):
        # Harmonic mean can only be calculated on positive values
        positive_data = data[data > 0]
        if len(positive_data) == 0:
            return None
        return len(positive_data) / np.sum(1.0 / positive_data)

    # Other methods to return calculated statistics
    @property
    def get_row_count(self):
        return len(self.data)

    @property
    def get_current(self):
        return self.series.iloc[-1]

    @property
    def get_mean(self):
        return self.mean

    @property
    def get_median(self):
        return self.median

    @property
    def get_mode(self):
        return self.mode

    @property
    def get_min(self):
        return self.min

    @property
    def get_max(self):
        return self.max

    @property
    def get_std_dev(self):
        return self.std_dev

    @property
    def get_variance(self):
        return self.variance

    @property
    def get_mad(self):
        return self.mad

    @property
    def get_z_score(self):
        return self.z_score

    @property
    def get_geometric_mean(self):
        return self.geometric_mean

    @property
    def get_harmonic_mean(self):
        return self.harmonic_mean

    @property
    def get_skewness(self):
        return self.skewness

    @property
    def get_kurtosis(self):
        return self.kurtosis

    @property
    def get_iqr(self):
        return self.iqr

    @property
    def get_range(self):
        return self.range

    @property
    def get_cv(self):
        return self.cv

    @property
    def current_percentile(self):
        """Calculate the percentile rank of the current value within the dataset."""
        current_value = self.series.iloc[-1]  # Get the most recent data point
        return (self.series.rank(pct=True).iloc[-1]) * 100

    @property
    def get_percentile_25(self):
        return self.series.quantile(0.25)

    @property
    def get_percentile_50(self):
        return self.series.quantile(0.50)

    @property
    def get_percentile_75(self):
        return self.series.quantile(0.75)

    @property
    def get_percentile_90(self):
        return self.series.quantile(0.90)

    @property
    def get_percentile_95(self):
        return self.series.quantile(0.95)

    @property
    def get_rolling_mean7(self):
        return self.series.rolling(window=7).mean().iloc[-1]

    @property
    def get_rolling_mean30(self):
        return self.series.rolling(window=30).mean().iloc[-1]

    @property
    def get_30_day_volatility(self):
        return self.series.pct_change().rolling(window=30).std().iloc[-1]

    @property
    def get_stability_score(self):
        return 1 / (1 + self.get_cv) if self.get_cv is not None else None

    @property
    def get_rsi(self):
        delta = self.series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi_series = 100 - (100 / (1 + rs))
        return rsi_series.iloc[-1]