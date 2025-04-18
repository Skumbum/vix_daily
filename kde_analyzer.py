import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity


class VIXKDEAnalyzer:
    """
    A class for analyzing VIX data using Kernel Density Estimation techniques.

    This class provides tools for loading VIX data from a DataFrame, performing KDE analysis,
    and generating statistical reports.
    """

    def __init__(self, data: pd.DataFrame, column: str = "Close", kernel: str = 'gaussian'):
        """
        Initialize the VIX KDE Analyzer.

        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame containing VIX time series data.
        column : str
            Column name containing VIX values (default: 'Close').
        kernel : str
            Kernel type to use for KDE (default: 'gaussian').
        """
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in data")
        self.data = data[[column]].dropna()
        self.column = column
        self.kernel = kernel
        self.bandwidth = None
        self.kde_model = None
        self.current_vix = self.data[column].iloc[-1]  # Most recent value
        self.fit()

    def optimize_bandwidth(self, bandwidth_range=None):
        """
        Find the optimal bandwidth using cross-validation.
        """
        data_array = self.data[self.column].values.reshape(-1, 1)
        if bandwidth_range is None:
            std = np.std(data_array)
            bandwidth_range = np.linspace(0.1 * std, 2 * std, 30)

        # Create a custom scorer for KDE
        def kde_scorer(estimator, X):
            return np.mean(estimator.score_samples(X))

        grid = GridSearchCV(
            KernelDensity(kernel=self.kernel),
            {'bandwidth': bandwidth_range},
            cv=5,
            scoring=kde_scorer  # Use our custom scorer
        )

        grid.fit(data_array)
        self.bandwidth = grid.best_params_['bandwidth']
        return self.bandwidth

    def fit(self):
        """
        Fit the KDE model to the VIX data.
        """
        data_array = self.data[self.column].values.reshape(-1, 1)
        if self.bandwidth is None:
            self.optimize_bandwidth()

        self.kde_model = KernelDensity(bandwidth=self.bandwidth, kernel=self.kernel)
        self.kde_model.fit(data_array)
        return self

    def estimate_pdf(self, x_range=None, points=1000):
        """
        Estimate the probability density function over a range.

        Parameters:
        -----------
        x_range : tuple or None
            Range of VIX values to estimate (min, max)
        points : int
            Number of points to evaluate

        Returns:
        --------
        tuple:
            (x_values, pdf_values)
        """
        if self.kde_model is None:
            raise ValueError("KDE model not fitted. Call fit() first.")

        data_array = self.data[self.column].values
        if x_range is None:
            min_val = max(0, data_array.min() - 0.1 * data_array.std())
            max_val = data_array.max() + 0.1 * data_array.std()
            x_range = (min_val, max_val)

        x = np.linspace(x_range[0], x_range[1], points).reshape(-1, 1)
        log_pdf = self.kde_model.score_samples(x)
        pdf = np.exp(log_pdf)

        return x.flatten(), pdf

    def calculate_percentile(self, value):
        """
        Calculate the percentile of a given VIX value.

        Parameters:
        -----------
        value : float
            VIX value

        Returns:
        --------
        float:
            Percentile (0-100)
        """
        return stats.percentileofscore(self.data[self.column], value)

    def calculate_probability(self, lower_bound, upper_bound=None, points=1000):
        """
        Calculate probability of VIX being in a specified range.

        Parameters:
        -----------
        lower_bound : float
            Lower bound of range
        upper_bound : float or None
            Upper bound of range. If None, calculates P(VIX >= lower_bound)
        points : int
            Number of integration points

        Returns:
        --------
        float:
            Probability estimate
        """
        if self.kde_model is None:
            raise ValueError("KDE model not fitted. Call fit() first.")

        data_array = self.data[self.column].values
        if upper_bound is None:
            upper_bound = float(data_array.max()) + 3 * float(data_array.std())

        x = np.linspace(lower_bound, upper_bound, points).reshape(-1, 1)
        log_dens = self.kde_model.score_samples(x)
        dens = np.exp(log_dens)
        dx = (upper_bound - lower_bound) / (points - 1)
        probability = np.sum(dens) * dx

        return probability

    def estimate_expected_return_period(self, threshold):
        """
        Estimate the expected return period (in days) for exceeding a threshold.

        Parameters:
        -----------
        threshold : float
            VIX threshold value

        Returns:
        --------
        float:
            Expected return period in days
        """
        prob = self.calculate_probability(threshold, None)
        if prob > 0:
            return 1.0 / prob
        else:
            return float('inf')

    def estimate_conditional_expectation(self, threshold, upper_limit=None, points=1000):
        """
        Calculate E[VIX | VIX > threshold].

        Parameters:
        -----------
        threshold : float
            Threshold value
        upper_limit : float or None
            Upper integration limit. If None, uses max + 3*std
        points : int
            Number of integration points

        Returns:
        --------
        float:
            Conditional expectation
        """
        if self.kde_model is None:
            raise ValueError("KDE model not fitted. Call fit() first.")

        data_array = self.data[self.column].values
        if upper_limit is None:
            upper_limit = float(data_array.max()) + 3 * float(data_array.std())

        x = np.linspace(threshold, upper_limit, points).reshape(-1, 1)
        dx = (upper_limit - threshold) / (points - 1)
        log_dens = self.kde_model.score_samples(x)
        dens = np.exp(log_dens)

        numerator = np.sum(x.flatten() * dens) * dx
        denominator = np.sum(dens) * dx

        if denominator > 0:
            return numerator / denominator
        else:
            return float('inf')

    def get_statistics(self):
        """
        Return core KDE statistics as text.

        Returns:
        --------
        str:
            Formatted string containing core statistics
        """
        data_array = self.data[self.column].values
        mean = np.mean(data_array)
        median = np.median(data_array)
        mode = data_array[np.argmax(np.exp(self.kde_model.score_samples(data_array.reshape(-1, 1))))]
        variance = np.var(data_array)
        skewness = stats.skew(data_array)
        kurtosis = stats.kurtosis(data_array)

        stats_text = f"""
==================================================
KDE STATISTICS for column: {self.column}
==================================================
Count: {len(data_array)}
Mean: {mean:.4f}
Median: {median:.4f}
Mode: {mode:.4f}
Variance: {variance:.4f}
Skewness: {skewness:.4f}
Kurtosis: {kurtosis:.4f}
Kernel: {self.kernel}
Bandwidth: {self.bandwidth:.4f}
"""
        return stats_text.strip()

    def get_extended_statistics(self):
        """
        Get extended KDE statistics including probabilities and return periods.

        Returns:
        --------
        str:
            Formatted string containing extended statistics
        """
        current_value = self.current_vix
        current_percentile = self.calculate_percentile(current_value)

        # Calculate probability intervals
        prob_below_15 = self.calculate_probability(0, 15) * 100
        prob_15_to_20 = self.calculate_probability(15, 20) * 100
        prob_20_to_30 = self.calculate_probability(20, 30) * 100
        prob_above_30 = self.calculate_probability(30, None) * 100

        # Calculate return periods
        return_period_30 = self.estimate_expected_return_period(30)
        return_period_40 = self.estimate_expected_return_period(40)
        return_period_50 = self.estimate_expected_return_period(50)

        # Calculate expected shortfall
        expected_above_30 = self.estimate_conditional_expectation(30)
        expected_above_40 = self.estimate_conditional_expectation(40)

        # Calculate quantiles
        data_array = self.data[self.column].values
        q1 = np.percentile(data_array, 1)
        q5 = np.percentile(data_array, 5)
        q95 = np.percentile(data_array, 95)
        q99 = np.percentile(data_array, 99)

        stats_text = f"""
==================================================
EXTENDED KDE STATISTICS
==================================================
Current Value: {current_value:.2f}
Current Percentile: {current_percentile:.2f}%

Probability Intervals:
- Below 15: {prob_below_15:.2f}%
- Between 15-20: {prob_15_to_20:.2f}%
- Between 20-30: {prob_20_to_30:.2f}%
- Above 30: {prob_above_30:.2f}%

Return Periods (Days):
- Exceeding 30: {return_period_30:.2f}
- Exceeding 40: {return_period_40:.2f}
- Exceeding 50: {return_period_50:.2f}

Expected Values:
- E[VIX | VIX > 30]: {expected_above_30:.2f}
- E[VIX | VIX > 40]: {expected_above_40:.2f}

Key Quantiles:
- 1%: {q1:.2f}
- 5%: {q5:.2f}
- 95%: {q95:.2f}
- 99%: {q99:.2f}
"""
        return stats_text.strip()