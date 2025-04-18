# File: log_normal_fit_tester.py

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from typing import Dict


class LogNormalFitTester:
    """A class to test and visualize the goodness-of-fit for a log-normal distribution
    using a LogNormalAnalyser instance."""

    def __init__(self, analyser):
        """Initialize with a LogNormalAnalyser instance."""
        if not hasattr(analyser, 'data') or not hasattr(analyser, 'log_data'):
            raise ValueError("Provided object must be a valid LogNormalAnalyser instance")
        self.analyser = analyser
        self.data = analyser.data
        self.log_data = analyser.log_data
        self.column = analyser.column
        self.mu = analyser.mu
        self.sigma = analyser.sigma
        self.shape = analyser.shape
        self.scale = analyser.scale

        # Validate data
        if (self.data[self.column] <= 0).any():
            raise ValueError("Data must be positive for log-normal distribution")

    def test_goodness_of_fit(self) -> Dict:
        """Perform statistical tests to validate log-normal assumption."""
        results = {}

        # Shapiro-Wilk Test (normality of log-transformed data)
        shapiro_stat, shapiro_p = stats.shapiro(self.log_data)
        results['shapiro_wilk'] = {'statistic': shapiro_stat, 'p_value': shapiro_p,
                                   'passed': shapiro_p > 0.05,
                                   'warning': len(self.log_data) > 5000}

        # Kolmogorov-Smirnov Test (data vs. fitted log-normal)
        ks_stat, ks_p = stats.ks_1samp(self.data[self.column],
                                       stats.lognorm.cdf, args=(self.shape, 0, self.scale))
        results['kolmogorov_smirnov'] = {'statistic': ks_stat, 'p_value': ks_p,
                                         'passed': ks_p > 0.05}

        # Anderson-Darling Test (normality of log-transformed data)
        ad_result = stats.anderson(self.log_data, dist='norm')
        ad_stat = ad_result.statistic
        ad_critical = ad_result.critical_values[2]  # 5% significance level
        results['anderson_darling'] = {'statistic': ad_stat,
                                       'critical_value_5%': ad_critical,
                                       'passed': ad_stat < ad_critical}

        # Chi-Square Test (observed vs. expected frequencies)
        bins = np.histogram(self.data[self.column], bins='auto')[1]
        observed_freq, _ = np.histogram(self.data[self.column], bins=bins)
        expected_prob = stats.lognorm.cdf(bins[1:], self.shape, scale=self.scale) - \
                        stats.lognorm.cdf(bins[:-1], self.shape, scale=self.scale)
        # Normalize expected frequencies to match observed sum
        expected_freq = expected_prob * np.sum(observed_freq) / np.sum(expected_prob)
        # Ensure no expected frequencies are too low (< 5)
        expected_freq = np.where(expected_freq < 5, 5, expected_freq)
        # Renormalize to maintain sum after adjusting low frequencies
        expected_freq = expected_freq * np.sum(observed_freq) / np.sum(expected_freq)
        chi2_stat, chi2_p = stats.chisquare(observed_freq, expected_freq)
        results['chi_square'] = {'statistic': chi2_stat, 'p_value': chi2_p,
                                 'passed': chi2_p > 0.05}

        # Information Criteria (AIC and BIC)
        log_likelihood = np.sum(stats.lognorm.logpdf(self.data[self.column],
                                                     self.shape, scale=self.scale))
        k = 2  # Number of parameters (mu, sigma)
        n = len(self.data)
        results['aic'] = 2 * k - 2 * log_likelihood
        results['bic'] = k * np.log(n) - 2 * log_likelihood

        # Overall conclusion
        tests_passed = sum(1 for test in ['shapiro_wilk', 'kolmogorov_smirnov',
                                          'anderson_darling', 'chi_square']
                           if results[test]['passed'])
        results['summary'] = {
            'tests_passed': tests_passed,
            'total_tests': 4,
            'conclusion': ('Log-normal assumption is likely valid'
                           if tests_passed >= 3 else
                           'Log-normal assumption may not be valid')
        }

        return results

    def get_goodness_of_fit_summary(self) -> str:
        """Generate a human-readable summary of goodness-of-fit tests."""
        results = self.test_goodness_of_fit()

        shapiro_warning = ("\n  Warning: Sample size > 5000; p-value may not be accurate"
                           if results['shapiro_wilk']['warning'] else "")

        summary = f"""
==================================================
GOODNESS-OF-FIT TEST RESULTS for {self.column}
==================================================
Shapiro-Wilk Test (Normality of Log-Transformed Data):
  Statistic: {results['shapiro_wilk']['statistic']:.4f}
  P-Value: {results['shapiro_wilk']['p_value']:.4f}
  Passed: {'Yes' if results['shapiro_wilk']['passed'] else 'No'}{shapiro_warning}

Kolmogorov-Smirnov Test (Data vs. Log-Normal):
  Statistic: {results['kolmogorov_smirnov']['statistic']:.4f}
  P-Value: {results['kolmogorov_smirnov']['p_value']:.4f}
  Passed: {'Yes' if results['kolmogorov_smirnov']['passed'] else 'No'}

Anderson-Darling Test (Normality of Log-Transformed Data):
  Statistic: {results['anderson_darling']['statistic']:.4f}
  Critical Value (5%): {results['anderson_darling']['critical_value_5%']:.4f}
  Passed: {'Yes' if results['anderson_darling']['passed'] else 'No'}

Chi-Square Test (Observed vs. Expected Frequencies):
  Statistic: {results['chi_square']['statistic']:.4f}
  P-Value: {results['chi_square']['p_value']:.4f}
  Passed: {'Yes' if results['chi_square']['passed'] else 'No'}

Information Criteria:
  AIC: {results['aic']:.2f}
  BIC: {results['bic']:.2f}

Summary:
  Tests Passed: {results['summary']['tests_passed']} / {results['summary']['total_tests']}
  Conclusion: {results['summary']['conclusion']}
"""
        return summary.strip()

    def plot_fit_comparison(self, filename: str = "log_normal_fit_plots.png"):
        """Create diagnostic plots for log-normal fit assessment and compare distributions."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()

        # 1. Histogram vs. Fitted PDFs
        data = self.data[self.column]
        hist, bins, _ = axes[0].hist(data, bins='auto', density=True, alpha=0.7,
                                     label='Data Histogram')
        x = np.linspace(min(bins), max(bins), 100)
        # Plot PDFs for all distributions
        distributions = {
            'lognorm': stats.lognorm(self.shape, scale=self.scale),
            'norm': stats.norm(loc=data.mean(), scale=data.std()),
            'gamma': stats.gamma(*stats.gamma.fit(data, floc=0)),
            'weibull': stats.weibull_min(*stats.weibull_min.fit(data, floc=0)),
            'expon': stats.expon(scale=data.mean()),
            'loglogistic': stats.fisk(*stats.fisk.fit(data, floc=0)),
            'pareto': stats.pareto(*stats.pareto.fit(data, floc=0))
        }
        colors = ['r', 'g', 'b', 'k', 'm', 'c', 'y']
        for (name, dist), color in zip(distributions.items(), colors):
            pdf = dist.pdf(x)
            axes[0].plot(x, pdf, f'{color}-', label=f'{name} PDF')
        axes[0].set_title('Histogram vs. Fitted PDFs')
        axes[0].legend()

        # 2. QQ Plot (log-transformed data vs. normal)
        stats.probplot(self.log_data, dist='norm', plot=axes[1])
        axes[1].set_title('QQ Plot (Log-Transformed Data)')

        # 3. Empirical vs. Theoretical CDF
        sorted_data = np.sort(data)
        n = len(sorted_data)
        empirical_cdf = np.arange(1, n + 1) / n
        theoretical_cdf = stats.lognorm.cdf(sorted_data, self.shape, scale=self.scale)
        axes[2].plot(sorted_data, empirical_cdf, 'b-', label='Empirical CDF')
        axes[2].plot(sorted_data, theoretical_cdf, 'r--', label='Log-Normal CDF')
        axes[2].set_title('Empirical vs. Log-Normal CDF')
        axes[2].legend()

        # 4. P-P Plot
        theoretical_cdf = stats.lognorm.cdf(sorted_data, self.shape, scale=self.scale)
        empirical_cdf = np.arange(1, n + 1) / n
        axes[3].scatter(theoretical_cdf, empirical_cdf, s=10)
        axes[3].plot([0, 1], [0, 1], 'r--')
        axes[3].set_title('P-P Plot (Log-Normal)')
        axes[3].set_xlabel('Theoretical Probabilities')
        axes[3].set_ylabel('Empirical Probabilities')

        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def compare_distributions(self) -> Dict:
        """Compare log-normal fit against other distributions."""
        data = self.data[self.column]
        distributions = {
            'lognorm': stats.lognorm(self.shape, scale=self.scale),
            'norm': stats.norm(loc=data.mean(), scale=data.std()),
            'gamma': stats.gamma(*stats.gamma.fit(data, floc=0)),
            'weibull': stats.weibull_min(*stats.weibull_min.fit(data, floc=0)),
            'expon': stats.expon(scale=data.mean()),
            'loglogistic': stats.fisk(*stats.fisk.fit(data, floc=0)),
            'pareto': stats.pareto(*stats.pareto.fit(data, floc=0))
        }

        results = {}
        n = len(data)
        for name, dist in distributions.items():
            # Log-likelihood
            log_likelihood = np.sum(dist.logpdf(data))

            # Number of parameters
            k = len(dist.stats(moments='mv'))  # Approximate based on moments
            if name == 'lognorm':
                k = 2  # shape, scale
            elif name == 'norm':
                k = 2  # mean, std
            elif name == 'gamma':
                k = 2  # a, scale (floc=0)
            elif name == 'weibull':
                k = 2  # c, scale (floc=0)
            elif name == 'expon':
                k = 1  # scale
            elif name == 'loglogistic':
                k = 2  # c, scale (floc=0)
            elif name == 'pareto':
                k = 2  # b, scale (floc=0)

            # AIC and BIC
            aic = 2 * k - 2 * log_likelihood
            bic = k * np.log(n) - 2 * log_likelihood

            # Kolmogorov-Smirnov Test
            ks_stat, ks_p = stats.ks_1samp(data, dist.cdf)

            results[name] = {
                'log_likelihood': log_likelihood,
                'aic': aic,
                'bic': bic,
                'ks_statistic': ks_stat,
                'ks_p_value': ks_p
            }

        # Rank distributions by AIC
        results['best_fit'] = min(results.items(), key=lambda x: x[1]['aic'])[0]

        return results