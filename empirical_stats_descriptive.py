import numpy as np
import pandas as pd


class EmpiricalStatsDescriptive:
    def __init__(self, data, column='Close'):
        self.data = data
        if isinstance(data, pd.DataFrame):
            if column in data.columns:
                self.series = data[column]
            else:
                self.series = data.iloc[:, 0]
        else:
            self.series = data

        self.update_stats()

    def update_stats(self):
        series = self.series
        self._mean = series.mean()
        self._std_dev = series.std()
        self._max = series.max()
        self._min = series.min()
        self._median = series.median()
        self._mode = series.mode().iloc[0] if not series.mode().empty else None
        self._variance = series.var()
        self._mad = self.calculate_mad(series)

        current_value = series.iloc[-1]
        self._z_score = (current_value - self._mean) / self._std_dev if self._std_dev != 0 else None

        self._geometric_mean = self.calculate_geometric_mean(series)
        self._harmonic_mean = self.calculate_harmonic_mean(series)
        self._skewness = series.skew()
        self._kurtosis = series.kurt()
        self._iqr = series.quantile(0.75) - series.quantile(0.25)
        self._range = self._max - self._min
        self._cv = self._std_dev / self._mean if self._mean != 0 else None

        self._current = current_value
        self._current_percentile = series.rank(pct=True).iloc[-1] * 100
        self._percentile_25 = series.quantile(0.25)
        self._percentile_50 = series.quantile(0.50)
        self._percentile_75 = series.quantile(0.75)
        self._percentile_90 = series.quantile(0.90)
        self._percentile_95 = series.quantile(0.95)

        self._rolling_mean_7 = series.rolling(window=7).mean().iloc[-1]
        self._rolling_mean_30 = series.rolling(window=30).mean().iloc[-1]
        self._volatility_30d = series.pct_change().rolling(window=30).std().iloc[-1]
        self._stability_score = 1 / (1 + self._cv) if self._cv is not None else None

        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        rsi_series = 100 - (100 / (1 + rs))
        self._rsi = rsi_series.iloc[-1]

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
    def current(self): return self._current
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
    def current_percentile(self): return self._current_percentile
    @property
    def percentile_25(self): return self._percentile_25
    @property
    def percentile_50(self): return self._percentile_50
    @property
    def percentile_75(self): return self._percentile_75
    @property
    def percentile_90(self): return self._percentile_90
    @property
    def percentile_95(self): return self._percentile_95
    @property
    def rolling_mean_7(self): return self._rolling_mean_7
    @property
    def rolling_mean_30(self): return self._rolling_mean_30
    @property
    def volatility_30d(self): return self._volatility_30d
    @property
    def stability_score(self): return self._stability_score
    @property
    def rsi(self): return self._rsi

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
