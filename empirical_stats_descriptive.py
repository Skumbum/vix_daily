import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
import warnings


class EmpiricalStatsDescriptive:
    def __init__(self, data, column='Close', weight_type='linear', exp_decay=0.9, hurst_min_lag=10, hurst_max_lag=30,
                 compute_price_metrics=True, rolling_short_window=7, rolling_long_window=30, rsi_period=14):
        self.data = data
        self.weight_type = weight_type
        self.exp_decay = exp_decay
        self.hurst_min_lag = hurst_min_lag
        self.hurst_max_lag = hurst_max_lag
        self.compute_price_metrics = compute_price_metrics
        self.rolling_short_window = rolling_short_window
        self.rolling_long_window = rolling_long_window
        self.rsi_period = rsi_period

        print(f"Inside __init__: data type = {type(data)}")
        if isinstance(data, pd.DataFrame):
            print(f"Inside __init__: data columns = {data.columns}")
            print(f"Inside __init__: data head = \n{data.head()}")
            if column not in data.columns:
                raise ValueError(f"Column '{column}' not found in data.columns")
            self.series = data[column]
            print(f"Inside __init__: data['{column}'] type = {type(self.series)}")
            print(f"Inside __init__: data['{column}'] head = \n{self.series.head()}")
        else:
            self.series = pd.Series(data)
            print(f"Inside __init__: Converted data to Series, series type = {type(self.series)}")

        if not isinstance(self.series, pd.Series):
            raise ValueError(f"Expected self.series to be a pandas Series, got {type(self.series)}")

        self.update_stats()

    def update_stats(self):
        series = self.series

        # Handle edge cases for the series
        if len(series) == 0 or series.isna().all().item():
            self._mean = None
            self._std_dev = None
            self._max = None
            self._min = None
            self._median = None
            self._mode = None
            self._variance = None
            self._mad = None
            self._midrange = None
            self._trimmed_mean = None
            self._time_weighted_mean = None
            self._raw_medad = None
            self._medad = None
            self._entropy = None
            self._hurst = None
            self._hurst_dfa = None
            self._cv = None
            self._cv_percent = None
            self._cv_abs_percent = None
            self._sem = None
            self._z_score = None
            self._geometric_mean = None
            self._harmonic_mean = None
            self._volatility_long = None
            self._stability_score = None
            self._rsi = None
            self._skewness = None
            self._kurtosis = None
            self._iqr = None
            self._range = None
            self._current = None
            self._current_percentile = None
            self._percentile_5 = None
            self._percentile_10 = None
            self._percentile_25 = None
            self._percentile_50 = None
            self._percentile_75 = None
            self._percentile_90 = None
            self._percentile_95 = None
            self._percentile_99 = None
            self._rolling_mean_short = None
            self._rolling_mean_long = None
            self._autocorrelation = None
            self._trend_slope = None
            self._adf_statistic = None
            self._adf_pvalue = None
            self._kpss_statistic = None
            self._kpss_pvalue = None
            self._kpss_warning = None
            return

        # Compute statistics with scalar extraction
        mean_val = series.mean().item()
        self._mean = float(mean_val) if not pd.isna(mean_val) else None

        std_val = series.std().item()
        self._std_dev = float(std_val) if not pd.isna(std_val) else None

        max_val = series.max().item()
        self._max = float(max_val) if not pd.isna(max_val) else None

        min_val = series.min().item()
        self._min = float(min_val) if not pd.isna(min_val) else None

        median_val = series.median().item()
        self._median = float(median_val) if not pd.isna(median_val) else None

        mode_series = series.mode()
        mode_val = mode_series.iloc[0].item() if not mode_series.empty else None
        self._mode = float(mode_val) if mode_val is not None and not pd.isna(mode_val) else None

        var_val = series.var().item()
        self._variance = float(var_val) if not pd.isna(var_val) else None

        self._mad = self.calculate_mad(series)
        self._midrange = (self._max + self._min) / 2 if self._max is not None and self._min is not None else None
        self._trimmed_mean = self.calculate_trimmed_mean(series)
        self._time_weighted_mean = self.calculate_time_weighted_mean(series, weight_type=self.weight_type)
        self._raw_medad, self._medad = self.calculate_medad(series)
        self._entropy = self.calculate_entropy(series)
        self._hurst = self.calculate_hurst(series)
        self._hurst_dfa = self.calculate_hurst_dfa(series)

        # Only compute CV if mean is not near zero
        mean_threshold = 1e-6  # Define a threshold for "near zero"
        if self._mean is not None and abs(self._mean) > mean_threshold:
            self._cv = self._std_dev / self._mean if self._std_dev is not None else None
            self._cv_percent = self._cv * 100 if self._cv is not None else None
            self._cv_abs_percent = (self._std_dev / abs(self._mean)) * 100 if self._std_dev is not None else None
        else:
            print(f"Warning: Mean ({self._mean}) is near zero, skipping CV calculations.")
            self._cv = None
            self._cv_percent = None
            self._cv_abs_percent = None

        self._sem = self._std_dev / np.sqrt(len(series)) if self._std_dev is not None and len(series) > 0 else None

        current_val = series.iloc[-1].item()
        self._current = float(current_val) if not pd.isna(current_val) else None
        self._z_score = (
                                    self._current - self._mean) / self._std_dev if self._std_dev is not None and self._std_dev != 0 and self._mean is not None and self._current is not None else None

        if self.compute_price_metrics:
            self._geometric_mean = self.calculate_geometric_mean(series)
            self._harmonic_mean = self.calculate_harmonic_mean(series)
            vol_long = series.pct_change().rolling(window=self.rolling_long_window).std().iloc[-1].item()
            self._volatility_long = float(vol_long) if not pd.isna(vol_long) else None
            self._stability_score = 1 / (1 + self._cv) if self._cv is not None else None

            delta = series.diff()
            gain = delta.where(delta > 0, 0).rolling(window=self.rsi_period).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=self.rsi_period).mean()
            rs = gain / loss
            rsi_series = 100 - (100 / (1 + rs))
            rsi_val = rsi_series.iloc[-1].item()
            self._rsi = float(rsi_val) if not pd.isna(rsi_val) else None
        else:
            self._geometric_mean = None
            self._harmonic_mean = None
            self._volatility_long = None
            self._stability_score = None
            self._rsi = None

        skew_val = series.skew().item()
        self._skewness = float(skew_val) if not pd.isna(skew_val) else None

        kurt_val = series.kurt().item()
        self._kurtosis = float(kurt_val) if not pd.isna(kurt_val) else None

        iqr_val = (series.quantile(0.75) - series.quantile(0.25)).item()
        self._iqr = float(iqr_val) if not pd.isna(iqr_val) else None

        self._range = self._max - self._min if self._max is not None and self._min is not None else None

        rank_val = series.rank(pct=True).iloc[-1].item()
        self._current_percentile = float(rank_val * 100) if not pd.isna(rank_val) else None

        q05 = series.quantile(0.05).item()
        self._percentile_5 = float(q05) if not pd.isna(q05) else None

        q10 = series.quantile(0.10).item()
        self._percentile_10 = float(q10) if not pd.isna(q10) else None

        q25 = series.quantile(0.25).item()
        self._percentile_25 = float(q25) if not pd.isna(q25) else None

        q50 = series.quantile(0.50).item()
        self._percentile_50 = float(q50) if not pd.isna(q50) else None

        q75 = series.quantile(0.75).item()
        self._percentile_75 = float(q75) if not pd.isna(q75) else None

        q90 = series.quantile(0.90).item()
        self._percentile_90 = float(q90) if not pd.isna(q90) else None

        q95 = series.quantile(0.95).item()
        self._percentile_95 = float(q95) if not pd.isna(q95) else None

        q99 = series.quantile(0.99).item()
        self._percentile_99 = float(q99) if not pd.isna(q99) else None

        roll_short = series.rolling(window=self.rolling_short_window).mean().iloc[-1].item()
        self._rolling_mean_short = float(roll_short) if not pd.isna(roll_short) else None

        roll_long = series.rolling(window=self.rolling_long_window).mean().iloc[-1].item()
        self._rolling_mean_long = float(roll_long) if not pd.isna(roll_long) else None

        self._autocorrelation = self.calculate_autocorrelation(series)
        self._trend_slope = self.calculate_trend_slope(series)

        self._adf_statistic, self._adf_pvalue = self.calculate_adf(series)
        self._kpss_statistic, self._kpss_pvalue, self._kpss_warning = self.calculate_kpss(series)

    def calculate_mad(self, series):
        if len(series) == 0 or series.isna().all().item():
            return None
        mad_val = np.mean(np.abs(series - series.mean())).item()
        return float(mad_val) if not pd.isna(mad_val) else None

    def calculate_geometric_mean(self, data):
        valid_data = data.dropna()
        if valid_data.empty:
            return None
        min_val = valid_data.min().item()
        min_val = float(min_val) if not pd.isna(min_val) else None
        shift = 1 - min_val if min_val is not None and min_val <= 0 else 0
        if shift > 0:
            print(
                f"Warning: Data contains negative or zero values (min: {min_val:.2f}). Shifted by {shift:.2f} for geometric mean calculation.")
        shifted_data = valid_data + shift
        log_data = np.log(shifted_data)
        mean_log = np.mean(log_data).item()
        result = np.exp(mean_log) - shift
        return float(result) if not pd.isna(mean_log) else None

    def calculate_harmonic_mean(self, data):
        valid_data = data.dropna()
        if valid_data.empty:
            return None
        min_val = valid_data.min().item()
        min_val = float(min_val) if not pd.isna(min_val) else None
        shift = 1 - min_val if min_val is not None and min_val <= 0 else 0
        if shift > 0:
            print(
                f"Warning: Data contains negative or zero values (min: {min_val:.2f}). Shifted by {shift:.2f} for harmonic mean calculation.")
        shifted_data = valid_data + shift
        n = len(shifted_data)
        harmonic_sum = np.sum(1.0 / shifted_data).item()
        harmonic_sum = float(harmonic_sum) if not pd.isna(harmonic_sum) else None
        if harmonic_sum == 0 or harmonic_sum is None:
            return None
        harmonic_mean_shifted = n / harmonic_sum
        return float(harmonic_mean_shifted - shift)

    def calculate_trimmed_mean(self, series, trim_percent=0.1):
        if len(series) == 0 or series.isna().all().item():
            return None
        lower_bound = series.quantile(trim_percent).item()
        lower_bound = float(lower_bound) if not pd.isna(lower_bound) else None
        upper_bound = series.quantile(1 - trim_percent).item()
        upper_bound = float(upper_bound) if not pd.isna(upper_bound) else None
        if lower_bound is None or upper_bound is None:
            return None
        trimmed_series = series[(series >= lower_bound) & (series <= upper_bound)]
        trimmed_mean = trimmed_series.mean().item()
        return float(trimmed_mean) if not trimmed_series.empty and not pd.isna(trimmed_mean) else None

    def calculate_time_weighted_mean(self, series, weight_type='linear'):
        if len(series) == 0 or series.isna().all().item():
            return None

        valid_series = series.dropna()
        if valid_series.empty:
            return None

        # Convert valid_series to a NumPy array to ensure consistent shape
        valid_series = valid_series.to_numpy()
        n = len(valid_series)

        # Generate weights based on the length of valid_series
        if weight_type == 'linear':
            weights = np.arange(1, n + 1)
        elif weight_type == 'exponential':
            weights = np.array([self.exp_decay ** (n - i - 1) for i in range(n)])
        elif weight_type == 'quadratic':
            weights = np.array([i ** 2 for i in range(1, n + 1)])
        else:
            raise ValueError("weight_type must be 'linear', 'exponential', or 'quadratic'")

        # Normalize weights
        weights = weights / weights.sum()

        # Ensure shapes match and specify axis
        avg = np.average(valid_series, weights=weights, axis=0).item()
        return float(avg) if not pd.isna(avg) else None

    def calculate_medad(self, series):
        if len(series) == 0 or series.isna().all().item():
            return None, None
        median = series.median().item()
        median = float(median) if not pd.isna(median) else None
        if median is None:
            return None, None
        absolute_deviations = np.abs(series - median)
        raw_medad = absolute_deviations.median().item()
        raw_medad = float(raw_medad) if not pd.isna(raw_medad) else None
        normalized_medad = raw_medad * 1.4826 if raw_medad is not None else None
        return raw_medad, normalized_medad

    def calculate_entropy(self, series, bins='sturges'):
        if len(series) == 0 or series.isna().all().item():
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
        entropy = -np.sum(non_zero_probs * np.log2(non_zero_probs)).item()
        return float(entropy) if not pd.isna(entropy) else None

    def calculate_hurst(self, series):
        valid_series = series.dropna()
        n = len(valid_series)
        if n < 50 or valid_series.empty:
            return None
        if n < 100:
            print(f"Warning: Small sample size ({n} points) may lead to unreliable Hurst Exponent (R/S).")

        valid_series = valid_series.to_numpy()
        if n < self.hurst_min_lag * 2:
            return None

        lags = np.logspace(np.log10(self.hurst_min_lag), np.log10(min(self.hurst_min_lag * 25, n // 2)), num=10,
                           dtype=int)
        lags = np.unique(lags)

        rs_values = []
        for lag in lags:
            num_subseries = n // lag
            if num_subseries < 2:
                continue
            rs_subseries = []
            for i in range(num_subseries):
                subseries = valid_series[i * lag: (i + 1) * lag]
                mean = np.mean(subseries).item()
                mean = float(mean) if not pd.isna(mean) else None
                if mean is None:
                    continue
                deviations = subseries - mean
                cumulative_dev = np.cumsum(deviations)
                r_max = np.max(cumulative_dev).item()
                r_min = np.min(cumulative_dev).item()
                r = float(r_max - r_min) if not pd.isna(r_max) and not pd.isna(r_min) else None
                s = np.std(subseries, ddof=1).item()
                s = float(s) if not pd.isna(s) else None
                if s is not None and s > 0 and r is not None and r > 0:
                    rs_subseries.append(r / s)
            if rs_subseries:
                rs_mean = np.mean(rs_subseries).item()
                rs_values.append(float(rs_mean) if not pd.isna(rs_mean) else None)

        if len(rs_values) < 2 or any(v is None for v in rs_values):
            return None

        log_lags = np.log(lags[:len(rs_values)])
        log_rs = np.log([v for v in rs_values if v is not None])
        coeffs = np.polyfit(log_lags, log_rs, 1)
        hurst = coeffs[0]
        hurst = float(hurst) if not pd.isna(hurst) else None
        if hurst is not None and hurst > 0.9:
            print(f"Warning: High Hurst Exponent (R/S) ({hurst:.2f}) may indicate non-stationarity.")
        return hurst if hurst is not None and 0 <= hurst <= 1 else None

    def calculate_hurst_dfa(self, series):
        valid_series = series.dropna()
        n = len(valid_series)
        if n < 50 or valid_series.empty:
            return None
        if n < 100:
            print(f"Warning: Small sample size ({n} points) may lead to unreliable Hurst Exponent (DFA).")

        valid_series = valid_series.to_numpy()
        mean = np.mean(valid_series).item()
        mean = float(mean) if not pd.isna(mean) else None
        if mean is None:
            return None
        y = np.cumsum(valid_series - mean)

        min_window = max(4, self.hurst_min_lag)
        max_window = min(n // 4, self.hurst_max_lag)
        if max_window < min_window * 2:
            return None
        scales = np.logspace(np.log10(min_window), np.log10(max_window), num=20, dtype=int)
        scales = np.unique(scales)

        fluctuations = []
        for scale in scales:
            num_subseries = n // scale
            if num_subseries < 2:
                continue
            f_squared = []
            for i in range(num_subseries):
                start = i * scale
                end = (i + 1) * scale
                window = y[start:end]
                if len(window) < scale:
                    continue
                x = np.arange(scale)
                coeffs = np.polyfit(x, window, 1)
                trend = np.polyval(coeffs, x)
                detrended = window - trend
                f_mean = np.mean(detrended ** 2).item()
                f = np.sqrt(f_mean)
                f = float(f) if not pd.isna(f_mean) else None
                if f is not None and f > 0:
                    f_squared.append(f)
            if f_squared:
                f_mean = np.mean(f_squared).item()
                fluctuations.append(float(f_mean) if not pd.isna(f_mean) else None)

        if len(fluctuations) < 2 or any(v is None for v in fluctuations):
            return None

        log_scales = np.log(scales[:len(fluctuations)])
        log_fluctuations = np.log([f for f in fluctuations if f is not None])
        coeffs = np.polyfit(log_scales, log_fluctuations, 1)
        hurst_dfa = coeffs[0]
        hurst_dfa = float(hurst_dfa) if not pd.isna(hurst_dfa) else None
        if hurst_dfa is not None:
            hurst_dfa = min(max(hurst_dfa, 0), 1)
            if hurst_dfa > 0.9:
                print(
                    f"Warning: High Hurst Exponent (DFA) ({hurst_dfa:.2f}) suggests strong trends or non-stationarity.")
        return hurst_dfa

    def calculate_autocorrelation(self, series, lag=1):
        # Ensure series is a pandas Series
        if not isinstance(series, pd.Series):
            raise ValueError(f"Expected series to be a pandas Series in calculate_autocorrelation, got {type(series)}")

        valid_series = series.dropna()
        if len(valid_series) < lag + 1:
            return None
        autocorr = valid_series.autocorr(lag=lag).item()
        return float(autocorr) if not pd.isna(autocorr) else None

    def calculate_trend_slope(self, series):
        valid_series = series.dropna()
        if len(valid_series) < 2:
            return None
        x = np.arange(len(valid_series))
        coeffs = np.polyfit(x, valid_series.to_numpy(), 1)
        slope = coeffs[0]
        return float(slope) if not pd.isna(slope) else None

    def calculate_adf(self, series):
        valid_series = series.dropna()
        if len(valid_series) < 2:
            return None, None
        try:
            maxlag = 10
            result = adfuller(valid_series, regression='ct', maxlag=maxlag, autolag=None)
            stat = float(result[0]) if not pd.isna(result[0]) else None
            pval = float(result[1]) if not pd.isna(result[1]) else None
            return stat, pval
        except Exception as e:
            print(f"Warning: ADF test failed: {str(e)}")
            return None, None

    def calculate_kpss(self, series):
        valid_series = series.dropna()
        if len(valid_series) < 2:
            return None, None, "Series too short for KPSS test"

        # Check for numerical issues
        if np.any(np.isinf(valid_series)) or np.any(np.isnan(valid_series)):
            return None, None, "KPSS test unreliable due to inf/nan values in the series"

        try:
            # Use a more conservative number of lags
            n_lags = int(4 * (len(valid_series) / 100) ** 0.25)
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = kpss(valid_series, regression='c', nlags=n_lags)
                kpss_statistic = float(result[0]) if not pd.isna(result[0]) else None
                kpss_pvalue = float(result[1]) if not pd.isna(result[1]) else None
                kpss_warning = None
                for warn in w:
                    if "InterpolationWarning" in str(warn.message):
                        kpss_warning = "KPSS p-value may be unreliable: test statistic outside lookup table range"
                        break
            return kpss_statistic, kpss_pvalue, kpss_warning
        except Exception as e:
            return None, None, f"KPSS test failed: {str(e)}"

    def analyse_returns(self, return_type='pct'):
        if len(self.series) == 0 or self.series.isna().all().item():
            return None
        if return_type == 'pct':
            returns = self.series.pct_change().dropna()
        elif return_type == 'log':
            # Check for zero values before computing log returns
            if np.any(self.series <= 0):
                raise ValueError("Series contains zero or negative values, cannot compute log returns")
            returns = np.log(self.series / self.series.shift(1)).dropna()
        else:
            raise ValueError("return_type must be 'pct' or 'log'")
        return EmpiricalStatsDescriptive(returns, weight_type=self.weight_type, exp_decay=self.exp_decay,
                                         hurst_min_lag=self.hurst_min_lag, hurst_max_lag=self.hurst_max_lag,
                                         compute_price_metrics=False,
                                         rolling_short_window=self.rolling_short_window,
                                         rolling_long_window=self.rolling_long_window,
                                         rsi_period=self.rsi_period)

    def get_time_weighted_means(self, series):
        return {
            'linear': self.calculate_time_weighted_mean(series, 'linear'),
            'exponential': self.calculate_time_weighted_mean(series, 'exponential'),
            'quadratic': self.calculate_time_weighted_mean(series, 'quadratic')
        }

    @property
    def row_count(self):
        return len(self.data)

    @property
    def current(self):
        return self._current

    @property
    def mean(self):
        return self._mean

    @property
    def median(self):
        return self._median

    @property
    def mode(self):
        return self._mode

    @property
    def min(self):
        return self._min

    @property
    def max(self):
        return self._max

    @property
    def std_dev(self):
        return self._std_dev

    @property
    def variance(self):
        return self._variance

    @property
    def mad(self):
        return self._mad

    @property
    def midrange(self):
        return self._midrange

    @property
    def trimmed_mean(self):
        return self._trimmed_mean

    @property
    def time_weighted_mean(self):
        return self._time_weighted_mean

    @property
    def raw_medad(self):
        return self._raw_medad

    @property
    def medad(self):
        return self._medad

    @property
    def entropy(self):
        return self._entropy

    @property
    def hurst(self):
        return self._hurst

    @property
    def hurst_dfa(self):
        return self._hurst_dfa

    @property
    def z_score(self):
        return self._z_score

    @property
    def geometric_mean(self):
        return self._geometric_mean

    @property
    def harmonic_mean(self):
        return self._harmonic_mean

    @property
    def skewness(self):
        return self._skewness

    @property
    def kurtosis(self):
        return self._kurtosis

    @property
    def iqr(self):
        return self._iqr

    @property
    def range(self):
        return self._range

    @property
    def cv(self):
        return self._cv

    @property
    def cv_percent(self):
        return self._cv_percent

    @property
    def cv_abs_percent(self):
        return self._cv_abs_percent

    @property
    def sem(self):
        return self._sem

    @property
    def current_percentile(self):
        return self._current_percentile

    @property
    def percentile_5(self):
        return self._percentile_5

    @property
    def percentile_10(self):
        return self._percentile_10

    @property
    def percentile_25(self):
        return self._percentile_25

    @property
    def percentile_50(self):
        return self._percentile_50

    @property
    def percentile_75(self):
        return self._percentile_75

    @property
    def percentile_90(self):
        return self._percentile_90

    @property
    def percentile_95(self):
        return self._percentile_95

    @property
    def percentile_99(self):
        return self._percentile_99

    @property
    def rolling_mean_7(self):
        return self._rolling_mean_short

    @property
    def rolling_mean_30(self):
        return self._rolling_mean_long

    @property
    def volatility_30d(self):
        return self._volatility_long

    @property
    def stability_score(self):
        return self._stability_score

    @property
    def rsi(self):
        return self._rsi

    @property
    def autocorrelation(self):
        return self._autocorrelation

    @property
    def trend_slope(self):
        return self._trend_slope

    @property
    def adf_statistic(self):
        return self._adf_statistic

    @property
    def adf_pvalue(self):
        return self._adf_pvalue

    @property
    def kpss_statistic(self):
        return self._kpss_statistic

    @property
    def kpss_pvalue(self):
        return self._kpss_pvalue

    @property
    def kpss_warning(self):
        return self._kpss_warning

    def stats_summary(self):
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
            "hurst_dfa": self.hurst_dfa,
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
            "percentile_5": self.percentile_5,
            "percentile_10": self.percentile_10,
            "percentile_25": self.percentile_25,
            "percentile_50": self.percentile_50,
            "percentile_75": self.percentile_75,
            "percentile_90": self.percentile_90,
            "percentile_95": self.percentile_95,
            "percentile_99": self.percentile_99,
            "rolling_mean_7": self.rolling_mean_7,
            "rolling_mean_30": self.rolling_mean_30,
            "volatility_30d": self.volatility_30d,
            "stability_score": self.stability_score,
            "rsi": self.rsi,
            "autocorrelation": self.autocorrelation,
            "trend_slope": self.trend_slope,
            "adf_statistic": self.adf_statistic,
            "adf_pvalue": self.adf_pvalue,
            "kpss_statistic": self.kpss_statistic,
            "kpss_pvalue": self.kpss_pvalue,
            "kpss_warning": self.kpss_warning
        }
        return {k: round(v, 2) if isinstance(v, (int, float)) else v for k, v in summary.items()}