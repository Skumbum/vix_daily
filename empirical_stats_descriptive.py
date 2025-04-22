import numpy as np
import pandas as pd
from functools import cached_property


class EmpiricalStatsDescriptive:
    def __init__(self, data):
        self.data = data
        if isinstance(data, pd.DataFrame) and 'Close' in data.columns:
            self.series = data['Close']
        elif isinstance(data, pd.DataFrame):
            self.series = data.iloc[:, 0]
        else:
            self.series = data

        self.update_stats()

    def update_stats(self):
        self._mean = self.series.mean()
        self._std_dev = self.series.std()
        self._max = self.series.max()
        self._min = self.series.min()
        self._median = self.series.median()
        self._mode = self.series.mode().iloc[0] if not self.series.mode().empty else None
        self._variance = self.series.var()
        self._mad = self.calculate_mad(self.series)

        current_value = self.series.iloc[-1]
        self._z_score = (current_value - self._mean) / self._std_dev if self._std_dev != 0 else None

        self._geometric_mean = self.calculate_geometric_mean(self.series)
        self._harmonic_mean = self.calculate_harmonic_mean(self.series)
        self._skewness = self.series.skew()
        self._kurtosis = self.series.kurt()
        self._iqr = self.series.quantile(0.75) - self.series.quantile(0.25)
        self._range = self._max - self._min
        self._cv = self._std_dev / self._mean if self._mean != 0 else None

    def calculate_mad(self, series):
        return np.mean(np.abs(series - series.mean()))

    def calculate_geometric_mean(self, data):
        positive_data = data[data > 0]
        if len(positive_data) == 0:
            return None
        log_data = np.log(positive_data)
        return np.exp(np.mean(log_data))

    def calculate_harmonic_mean(self, data):
        positive_data = data[data > 0]
        if len(positive_data) == 0:
            return None
        return len(positive_data) / np.sum(1.0 / positive_data)

    @property
    def row_count(self):
        return len(self.data)

    @property
    def current(self):
        return self.series.iloc[-1]

    @property
    def mean(self): return self._mean

    @property
    def median(self): return self._median

    @property
    def mode(self): return self._mode

    @property
    def min(self): return self._min

    @property
    def max(self): return self._max

    @property
    def std_dev(self): return self._std_dev

    @property
    def variance(self): return self._variance

    @property
    def mad(self): return self._mad

    @property
    def z_score(self): return self._z_score

    @property
    def geometric_mean(self): return self._geometric_mean

    @property
    def harmonic_mean(self): return self._harmonic_mean

    @property
    def skewness(self): return self._skewness

    @property
    def kurtosis(self): return self._kurtosis

    @property
    def iqr(self): return self._iqr

    @property
    def range(self): return self._range

    @property
    def cv(self): return self._cv

    @property
    def current_percentile(self):
        return self.series.rank(pct=True).iloc[-1] * 100

    @property
    def percentile_25(self):
        return self.series.quantile(0.25)

    @property
    def percentile_50(self):
        return self.series.quantile(0.50)

    @property
    def percentile_75(self):
        return self.series.quantile(0.75)

    @property
    def percentile_90(self):
        return self.series.quantile(0.90)

    @property
    def percentile_95(self):
        return self.series.quantile(0.95)

    @property
    def rolling_mean_7(self):
        return self.series.rolling(window=7).mean().iloc[-1]

    @property
    def rolling_mean_30(self):
        return self.series.rolling(window=30).mean().iloc[-1]

    @property
    def volatility_30d(self):
        return self.series.pct_change().rolling(window=30).std().iloc[-1]

    @property
    def stability_score(self):
        return 1 / (1 + self.cv) if self.cv is not None else None

    @property
    def rsi(self):
        delta = self.series.diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        rsi_series = 100 - (100 / (1 + rs))
        return rsi_series.iloc[-1]

    def stats_summary(self):
        return {
            "row_count": self.row_count,
            "current": self.current,
            "mean": self.mean,
            "median": self.median,
            "mode": self.mode,
            "min": self.min,
            "max": self.max,
            "std_dev": self.std_dev,
            "variance": self.variance,
            "mad": self.mad,
            "z_score": self.z_score,
            "geometric_mean": self.geometric_mean,
            "harmonic_mean": self.harmonic_mean,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
            "iqr": self.iqr,
            "range": self.range,
            "cv": self.cv,
            "current_percentile": self.current_percentile,
            "percentile_25": self.percentile_25,
            "percentile_50": self.percentile_50,
            "percentile_75": self.percentile_75,
            "percentile_90": self.percentile_90,
            "percentile_95": self.percentile_95,
            "rolling_mean_7": self.rolling_mean_7,
            "rolling_mean_30": self.rolling_mean_30,
            "volatility_30d": self.volatility_30d,
            "stability_score": self.stability_score,
            "rsi": self.rsi
        }
