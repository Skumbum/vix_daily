import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.special import ndtr
from scipy.stats import linregress


class LogNormalAnalyser:
    def __init__(self, data: pd.DataFrame, column: str = "Close"):
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in data")
        self.data = data[[column]].dropna()
        self.column = column
        self.log_data = np.log(self.data[column])

        # Calculate core parameters once
        self.mu = self.log_data.mean()
        self.sigma = self.log_data.std()
        self.shape = self.sigma
        self.scale = np.exp(self.mu)

    def get_statistics(self):
        """Return log-normal fit statistics as text."""
        mean = stats.lognorm.mean(self.shape, scale=self.scale)
        median = stats.lognorm.median(self.shape, scale=self.scale)
        mode = np.exp(self.mu - self.sigma ** 2)
        variance = stats.lognorm.var(self.shape, scale=self.scale)
        skewness = stats.lognorm.stats(self.shape, scale=self.scale, moments='s')
        kurtosis = stats.lognorm.stats(self.shape, scale=self.scale, moments='k')

        stats_text = f"""
==================================================
LOG-NORMAL STATISTICS for column: {self.column}
==================================================
Log-Mean (μ): {self.mu:.4f}
Log-Std Dev (σ): {self.sigma:.4f}
Shape (σ): {self.shape:.4f}
Scale (e^μ): {self.scale:.4f}

Mean: {mean:.4f}
Median: {median:.4f}
Mode: {mode:.4f}
Variance: {variance:.4f}
Skewness: {skewness:.4f}
Kurtosis: {kurtosis:.4f}
"""
        return stats_text.strip()

    def get_probability_interval(self, lower_bound, upper_bound):
        """Calculate probability that value falls within [lower_bound, upper_bound]"""
        if lower_bound <= 0 or upper_bound <= 0:
            raise ValueError("Bounds must be positive for log-normal distribution")

        # P(a < X < b) = Φ((ln(b) - μ)/σ) - Φ((ln(a) - μ)/σ)
        cdf_upper = ndtr((np.log(upper_bound) - self.mu) / self.sigma)
        cdf_lower = ndtr((np.log(lower_bound) - self.mu) / self.sigma)
        return cdf_upper - cdf_lower

    def get_expected_shortfall(self, threshold):
        """Calculate expected value given that value exceeds threshold"""
        if threshold <= 0:
            raise ValueError("Threshold must be positive for log-normal distribution")

        # Convert to standard normal threshold
        z_threshold = (np.log(threshold) - self.mu) / self.sigma

        # Probability of exceeding threshold
        exceed_prob = 1 - ndtr(z_threshold)

        if exceed_prob == 0:  # Avoid division by zero
            return float('inf')

        # Correct formula: E[X | X > t] = e^(μ + σ²/2) * Φ(σ - z_threshold) / P(X > t)
        mean = np.exp(self.mu + (self.sigma ** 2) / 2)  # Mean of log-normal
        conditional_expectation = mean * ndtr(self.sigma - z_threshold) / exceed_prob
        return conditional_expectation

    def get_return_period(self, threshold):
        """Calculate average time between occurrences exceeding threshold"""
        if threshold <= 0:
            raise ValueError("Threshold must be positive for log-normal distribution")

        # P(X > threshold)
        exceed_prob = 1 - stats.lognorm.cdf(threshold, self.shape, scale=self.scale)

        if exceed_prob == 0:
            return float('inf')

        # Average return period is 1/p where p is probability of exceeding threshold
        return 1 / exceed_prob

    def get_confidence_interval(self, confidence=0.95):
        """Calculate interval containing values with specified confidence level"""
        alpha = 1 - confidence
        lower_quantile = alpha / 2
        upper_quantile = 1 - alpha / 2

        lower_bound = stats.lognorm.ppf(lower_quantile, self.shape, scale=self.scale)
        upper_bound = stats.lognorm.ppf(upper_quantile, self.shape, scale=self.scale)

        return lower_bound, upper_bound

    def get_survival_function(self, x):
        """Calculate probability of exceeding value x: P(X > x)"""
        if x <= 0:
            raise ValueError("Value must be positive for log-normal distribution")

        return 1 - stats.lognorm.cdf(x, self.shape, scale=self.scale)

    def get_quantile(self, p):
        """Calculate value at specified percentile p"""
        if not 0 <= p <= 1:
            raise ValueError("Percentile must be between 0 and 1")

        return stats.lognorm.ppf(p, self.shape, scale=self.scale)

    def get_half_life(self, start_value, target_value=None, lookback=252):
        """
        Robust half-life estimation with error handling
        Parameters:
        - start_value: Current value to measure reversion from
        - target_value: Reversion target (defaults to median)
        - lookback: Days of history to consider (default 1 year)
        """
        if start_value <= 0:
            return float('nan')

        try:
            if target_value is None:
                target_value = self.scale  # Log-normal median

            # Use most recent data
            recent_data = self.data[self.column].tail(lookback)
            log_data = np.log(recent_data)

            # Calculate daily log returns
            log_returns = log_data.diff().dropna()
            log_prices = log_data.shift(1).dropna()

            # Align the data
            common_index = log_returns.index.intersection(log_prices.index)
            log_returns = log_returns.loc[common_index]
            log_prices = log_prices.loc[common_index]

            # Perform linear regression: Δln(X) = α + β*ln(X_{t-1}) + ε
            slope, intercept, _, _, _ = linregress(log_prices, log_returns)

            # Calculate mean reversion parameters
            if slope >= 0:  # Non-mean-reverting case
                print("Warning: Non-mean-reverting behavior detected (slope >= 0)")
                return float('inf')

            theta = -slope
            half_life_days = np.log(2) / theta

            return half_life_days

        except Exception as e:
            print(f"Error calculating half-life: {e}")
            return float('nan')

    def get_volatility_of_volatility(self):
        """Calculate the volatility of volatility"""
        # For a log-normal, this is related to the variance of log returns
        return np.sqrt(np.exp(self.sigma ** 2) - 1)

    def get_extended_statistics(self):
        """Get extended log-normal statistics"""
        current_value = self.data[self.column].iloc[-1]

        # Calculate percentile of current value
        current_percentile = stats.lognorm.cdf(current_value, self.shape, scale=self.scale) * 100

        # Calculate probability intervals
        prob_below_10 = stats.lognorm.cdf(10, self.shape, scale=self.scale) * 100
        prob_10_to_20 = (stats.lognorm.cdf(20, self.shape, scale=self.scale) -
                         stats.lognorm.cdf(10, self.shape, scale=self.scale)) * 100
        prob_20_to_30 = (stats.lognorm.cdf(30, self.shape, scale=self.scale) -
                         stats.lognorm.cdf(20, self.shape, scale=self.scale)) * 100
        prob_above_30 = (1 - stats.lognorm.cdf(30, self.shape, scale=self.scale)) * 100

        # Calculate return periods
        return_period_30 = self.get_return_period(30)
        return_period_40 = self.get_return_period(40)
        return_period_50 = self.get_return_period(50)

        # Calculate expected shortfall
        expected_above_30 = self.get_expected_shortfall(30)
        expected_above_40 = self.get_expected_shortfall(40)

        # Calculate 95% confidence interval
        lower_ci, upper_ci = self.get_confidence_interval(0.95)

        # Calculate key quantiles
        q1 = self.get_quantile(0.01)
        q5 = self.get_quantile(0.05)
        q95 = self.get_quantile(0.95)
        q99 = self.get_quantile(0.99)

        # Calculate half-life if current value is above median
        if current_value > self.scale:
            half_life = self.get_half_life(current_value)
        else:
            half_life = None

        # Calculate volatility of volatility
        vol_of_vol = self.get_volatility_of_volatility()

        stats_text = f"""
==================================================
EXTENDED LOG-NORMAL STATISTICS
==================================================
Current Value: {current_value:.2f}
Current Percentile: {current_percentile:.2f}%

Probability Intervals:
- Below 10: {prob_below_10:.2f}%
- Between 10-20: {prob_10_to_20:.2f}%
- Between 20-30: {prob_20_to_30:.2f}%
- Above 30: {prob_above_30:.2f}%

Return Periods (Days):
- Exceeding 30: {return_period_30:.2f}
- Exceeding 40: {return_period_40:.2f}
- Exceeding 50: {return_period_50:.2f}

Expected Values:
- E[X | X > 30]: {expected_above_30:.2f}
- E[X | X > 40]: {expected_above_40:.2f}

95% Confidence Interval:
- Lower: {lower_ci:.2f}
- Upper: {upper_ci:.2f}

Key Quantiles:
- 1%: {q1:.2f}
- 5%: {q5:.2f}
- 95%: {q95:.2f}
- 99%: {q99:.2f}

Volatility of Volatility: {vol_of_vol:.4f}
"""
        if half_life is not None:
            stats_text += f"Estimated Half-Life to Median: {half_life:.2f} days\n"

        return stats_text.strip()