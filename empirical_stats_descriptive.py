import numpy as np
import pandas as pd


class EmpiricalStatsDescriptive:
    """Computes a wide range of descriptive statistics for a pandas Series or DataFrame, suitable for any financial asset."""

    def __init__(self, data, column='Close', weight_type='linear', exp_decay=0.9, hurst_min_lag=4, hurst_max_lag=100):
        """
        Initialize the EmpiricalStatsDescriptive class.

        Args:
            data (pd.DataFrame or pd.Series): Input time series data (e.g., prices, returns, yields).
            column (str): Column to extract from DataFrame, default is 'Close'.
            weight_type (str): Type of weighting for time-weighted mean ('linear', 'exponential', 'quadratic', default 'linear').
            exp_decay (float): Decay rate for exponential time-weighted mean (default 0.9).
            hurst_min_lag (int): Minimum sub-series length for Hurst Exponent (default 4).
            hurst_max_lag (int): Maximum sub-series length for Hurst Exponent (default 100).
        """
        self.data = data
        self.weight_type = weight_type
        self.exp_decay = exp_decay
        self.hurst_min_lag = hurst_min_lag
        self.hurst_max_lag = hurst_max_lag
        if isinstance(data, pd.DataFrame):
            if column in data.columns:
                self.series = data[column]
            else:
                self.series = data.iloc[:, 0]
        else:
            self.series = data

        self.update_stats()

    def update_stats(self):
        """
        Compute and update all statistical metrics.
        """
        series = self.series
        self._mean = series.mean()
        self._std_dev = series.std()
        self._max = series.max()
        self._min = series.min()
        self._median = series.median()
        self._mode = series.mode().iloc[0] if not series.mode().empty else None
        self._variance = series.var()
        self._mad = self.calculate_mad(series)
        self._midrange = (self._max + self._min) / 2
        self._trimmed_mean = self.calculate_trimmed_mean(series)
        self._time_weighted_mean = self.calculate_time_weighted_mean(series, weight_type=self.weight_type)
        self._raw_medad, self._medad = self.calculate_medad(series)
        self._entropy = self.calculate_entropy(series)
        self._hurst = self.calculate_hurst(series)
        self._cv = self._std_dev / self._mean if self._mean != 0 else None
        self._cv_percent = self._cv * 100 if self._cv is not None else None
        self._cv_abs_percent = (self._std_dev / abs(self._mean)) * 100 if self._mean != 0 else None
        self._sem = self._std_dev / np.sqrt(len(series)) if len(series) > 0 else None

        current_value = series.iloc[-1]
        self._z_score = (current_value - self._mean) / self._std_dev if self._std_dev != 0 else None

        self._geometric_mean = self.calculate_geometric_mean(series)
        self._harmonic_mean = self.calculate_harmonic_mean(series)
        self._skewness = series.skew()
        self._kurtosis = series.kurt()
        self._iqr = series.quantile(0.75) - series.quantile(0.25)
        self._range = self._max - self._min

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
        """
        Calculate the Mean Absolute Deviation (MAD).

        Args:
            series (pd.Series): The data series.

        Returns:
            float: The MAD value.
        """
        return np.mean(np.abs(series - series.mean()))

    def calculate_geometric_mean(self, data):
        """
        Calculate the geometric mean of a series, handling negative values by shifting.

        Args:
            data (pd.Series): The data series.

        Returns:
            float or None: The geometric mean or None if data is empty or invalid.
        """
        valid_data = data.dropna()
        if valid_data.empty:
            return None
        # Shift data to ensure all values are positive
        min_val = valid_data.min()
        shift = 1 - min_val if min_val <= 0 else 0
        shifted_data = valid_data + shift
        log_data = np.log(shifted_data)
        mean_log = np.mean(log_data)
        return np.exp(mean_log) - shift

    def calculate_harmonic_mean(self, data):
        """
        Calculate the harmonic mean of a series, handling negative values by shifting.

        Args:
            data (pd.Series): The data series.

        Returns:
            float or None: The harmonic mean or None if data is empty or invalid.
        """
        valid_data = data.dropna()
        if valid_data.empty:
            return None
        # Shift data to ensure all values are positive
        min_val = valid_data.min()
        shift = 1 - min_val if min_val <= 0 else 0
        shifted_data = valid_data + shift
        n = len(shifted_data)
        harmonic_sum = np.sum(1.0 / shifted_data)
        if harmonic_sum == 0:
            return None
        harmonic_mean_shifted = n / harmonic_sum
        return harmonic_mean_shifted - shift

    def calculate_trimmed_mean(self, series, trim_percent=0.1):
        """
        Calculate the trimmed mean of a series by removing a percentage of extreme values.

        Args:
            series (pd.Series): The data series.
            trim_percent (float): Fraction of data to trim from each end (default 0.1 for 10%).

        Returns:
            float or None: The trimmed mean or None if insufficient data remains after trimming.
        """
        if series.empty or series.isna().all():
            return None
        lower_bound = series.quantile(trim_percent)
        upper_bound = series.quantile(1 - trim_percent)
        trimmed_series = series[(series >= lower_bound) & (series <= upper_bound)]
        return trimmed_series.mean() if not trimmed_series.empty else None

    def calculate_time_weighted_mean(self, series, weight_type='linear'):
        """
        Calculate the time-weighted mean of a series, with weights based on recency.

        Args:
            series (pd.Series): The data series.
            weight_type (str): Type of weighting ('linear', 'exponential', 'quadratic', default 'linear').

        Returns:
            float or None: The time-weighted mean or None if the series is empty or all NaN.
        """
        if series.empty or series.isna().all():
            return None

        valid_series = series.dropna()
        if valid_series.empty:
            return None

        n = len(valid_series)
        if weight_type == 'linear':
            weights = np.arange(1, n + 1)
        elif weight_type == 'exponential':
            weights = np.array([self.exp_decay ** (n - i - 1) for i in range(n)])
        elif weight_type == 'quadratic':
            weights = np.array([i**2 for i in range(1, n + 1)])
        else:
            raise ValueError("weight_type must be 'linear', 'exponential', or 'quadratic'")

        weights = weights / weights.sum()
        return np.average(valid_series, weights=weights)

    def calculate_medad(self, series):
        """
        Calculate the raw and normalized Median Absolute Deviation (MedAD).

        Args:
            series (pd.Series): The data series.

        Returns:
            tuple: (raw_medad, normalized_medad) or (None, None) if the series is empty or all NaN.
        """
        if series.empty or series.isna().all():
            return None, None
        median = series.median()
        absolute_deviations = np.abs(series - median)
        raw_medad = absolute_deviations.median()
        normalized_medad = raw_medad * 1.4826 if raw_medad is not None else None
        return raw_medad, normalized_medad

    def calculate_entropy(self, series, bins='sturges'):
        """
        Calculate the Shannon Entropy of a series, using histogram-based discretization.

        Args:
            series (pd.Series): The data series.
            bins (str or int): Number of bins or binning method ('sturges' for Sturges' rule, or integer for fixed bins).

        Returns:
            float or None: The Shannon Entropy (in bits) or None if the series is empty, all NaN, or constant.
        """
        if series.empty or series.isna().all():
            return None

        valid_series = series.dropna()
        if valid_series.empty or valid_series.nunique() == 1:
            return 0.0

        n = len(valid_series)
        if bins == 'sturges':
            num_bins = int(np.ceil(np.log2(n) + 1))
        elif isinstance(bins, int) and bins > 0:
            num_bins = bins
        else:
            raise ValueError("bins must be 'sturges' or a positive integer")

        counts, _ = np.histogram(valid_series, bins=num_bins, density=False)
        probabilities = counts / counts.sum()
        non_zero_probs = probabilities[probabilities > 0]
        entropy = -np.sum(non_zero_probs * np.log2(non_zero_probs))
        return entropy

    def calculate_hurst(self, series):
        """
        Calculate the Hurst Exponent of a series using Rescaled Range (R/S) analysis.

        Args:
            series (pd.Series): The data series.

        Returns:
            float or None: The Hurst Exponent (0 to 1) or None if insufficient data.
        """
        valid_series = series.dropna()
        n = len(valid_series)
        if n < 50 or valid_series.empty:
            return None
        if n < 100:
            print(f"Warning: Small sample size ({n} points) may lead to unreliable Hurst Exponent.")

        valid_series = valid_series.values
        if n < self.hurst_min_lag * 2:
            return None

        lags = np.logspace(np.log10(self.hurst_min_lag), np.log10(min(self.hurst_min_lag * 25, n // 2)), num=10, dtype=int)
        lags = np.unique(lags)

        rs_values = []
        for lag in lags:
            num_subseries = n // lag
            if num_subseries < 2:
                continue
            rs_subseries = []
            for i in range(num_subseries):
                subseries = valid_series[i * lag : (i + 1) * lag]
                mean = np.mean(subseries)
                deviations = subseries - mean
                cumulative_dev = np.cumsum(deviations)
                r = np.max(cumulative_dev) - np.min(cumulative_dev)
                s = np.std(subseries, ddof=1)
                if s > 0 and r > 0:
                    rs_subseries.append(r / s)
            if rs_subseries:
                rs_values.append(np.mean(rs_subseries))

        if len(rs_values) < 2:
            return None

        log_lags = np.log(lags[:len(rs_values)])
        log_rs = np.log(rs_values)
        coeffs = np.polyfit(log_lags, log_rs, 1)
        hurst = coeffs[0]
        if hurst > 0.9:
            print(f"Warning: High Hurst Exponent ({hurst:.2f}) may indicate non-stationarity.")
        return hurst if 0 <= hurst <= 1 else None

    def analyze_returns(self, return_type='pct'):
        """
        Compute descriptive statistics for returns of the series.

        Args:
            return_type (str): Type of returns ('pct' for percentage change, 'log' for log returns, default 'pct').

        Returns:
            EmpiricalStatsDescriptive: A new instance with statistics for returns.
        """
        if self.series.empty or self.series.isna().all():
            return None
        if return_type == 'pct':
            returns = self.series.pct_change().dropna()
        elif return_type == 'log':
            returns = np.log(self.series / self.series.shift(1)).dropna()
        else:
            raise ValueError("return_type must be 'pct' or 'log'")
        return EmpiricalStatsDescriptive(returns, weight_type=self.weight_type, exp_decay=self.exp_decay,
                                        hurst_min_lag=self.hurst_min_lag, hurst_max_lag=self.hurst_max_lag)

    def get_time_weighted_means(self, series):
        """
        Calculate time-weighted means for all weight types.

        Args:
            series (pd.Series): The data series.

        Returns:
            dict: Time-weighted means for linear, exponential, quadratic weights.
        """
        return {
            'linear': self.calculate_time_weighted_mean(series, 'linear'),
            'exponential': self.calculate_time_weighted_mean(series, 'exponential'),
            'quadratic': self.calculate_time_weighted_mean(series, 'quadratic')
        }

    @property
    def row_count(self):
        """int: Number of data points."""
        return len(self.data)

    @property
    def current(self): """float: Most recent value in the series."""; return self._current
    @property
    def mean(self): """float: Arithmetic mean."""; return self._mean
    @property
    def median(self): """float: Median value."""; return self._median
    @property
    def mode(self): """float: Mode value."""; return self._mode
    @property
    def min(self): """float: Minimum value."""; return self._min
    @property
    def max(self): """float: Maximum value."""; return self._max
    @property
    def std_dev(self): """float: Standard deviation."""; return self._std_dev
    @property
    def variance(self): """float: Variance."""; return self._variance
    @property
    def mad(self): """float: Mean Absolute Deviation."""; return self._mad
    @property
    def midrange(self): """float: Midrange (average of max and min)."""; return self._midrange
    @property
    def trimmed_mean(self): """float: Trimmed mean (10% trimmed from each end)."""; return self._trimmed_mean
    @property
    def time_weighted_mean(self): """float: Time-weighted mean (based on specified weight type)."""; return self._time_weighted_mean
    @property
    def raw_medad(self): """float: Raw Median Absolute Deviation."""; return self._raw_medad
    @property
    def medad(self): """float: Normalized Median Absolute Deviation."""; return self._medad
    @property
    def entropy(self): """float: Shannon Entropy (in bits)."""; return self._entropy
    @property
    def hurst(self): """float: Hurst Exponent (0 to 1)."""; return self._hurst
    @property
    def z_score(self): """float: Z-score of the most recent value."""; return self._z_score
    @property
    def geometric_mean(self): """float: Geometric mean."""; return self._geometric_mean
    @property
    def harmonic_mean(self): """float: Harmonic mean."""; return self._harmonic_mean
    @property
    def skewness(self): """float: Skewness of the distribution."""; return self._skewness
    @property
    def kurtosis(self): """float: Kurtosis of the distribution."""; return self._kurtosis
    @property
    def iqr(self): """float: Interquartile range."""; return self._iqr
    @property
    def range(self): """float: Data range (max - min).""" ; return self._range
    @property
    def cv(self): """float: Coefficient of Variation (ratio)."""; return self._cv
    @property
    def cv_percent(self): """float: Coefficient of Variation (percentage)."""; return self._cv_percent
    @property
    def cv_abs_percent(self): """float: Coefficient of Variation using absolute mean (percentage)."""; return self._cv_abs_percent
    @property
    def sem(self): """float: Standard Error of the Mean."""; return self._sem
    @property
    def current_percentile(self): """float: Percentile rank of the current value."""; return self._current_percentile
    @property
    def percentile_25(self): """float: 25th percentile."""; return self._percentile_25
    @property
    def percentile_50(self): """float: 50th percentile (median).""" ; return self._percentile_50
    @property
    def percentile_75(self): """float: 75th percentile."""; return self._percentile_75
    @property
    def percentile_90(self): """float: 90th percentile."""; return self._percentile_90
    @property
    def percentile_95(self): """float: 95th percentile."""; return self._percentile_95
    @property
    def rolling_mean_7(self): """float: 7-day rolling mean."""; return self._rolling_mean_7
    @property
    def rolling_mean_30(self): """float: 30-day rolling mean."""; return self._rolling_mean_30
    @property
    def volatility_30d(self): """float: 30-day volatility (standard deviation of percent changes).""" ; return self._volatility_30d
    @property
    def stability_score(self): """float: Stability score, inverse of CV.""" ; return self._stability_score
    @property
    def rsi(self): """float: 14-day Relative Strength Index.""" ; return self._rsi

    def stats_summary(self):
        """
        Return a summary of all computed statistics, rounded to 2 decimal places.

        Returns:
            dict: Dictionary of descriptive statistics.
        """
        summary = {
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
            "midrange": self.midrange,
            "trimmed_mean": self.trimmed_mean,
            "time_weighted_mean": self.time_weighted_mean,
            "raw_medad": self.raw_medad,
            "medad": self.medad,
            "entropy": self.entropy,
            "hurst": self.hurst,
            "z_score": self.z_score,
            "geometric_mean": self.geometric_mean,
            "harmonic_mean": self.harmonic_mean,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
            "iqr": self.iqr,
            "range": self.range,
            "cv": self.cv,
            "cv_percent": self.cv_percent,
            "cv_abs_percent": self.cv_abs_percent,
            "sem": self.sem,
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
        return {k: round(v, 2) if isinstance(v, (int, float)) else v for k, v in summary.items()}