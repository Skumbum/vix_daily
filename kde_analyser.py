import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity


class KDEAnalyser:
    def __init__(self, data: pd.DataFrame, column: str = "Close", kernel: str = 'gaussian'):
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in data")
        self.data = data[[column]].dropna()
        self.column = column
        self.kernel = kernel
        self.bandwidth = None
        self.kde_model = None
        self.current = self.data[column].iloc[-1]  # Most recent value
        self.fit()

    def optimize_bandwidth(self, bandwidth_range=None):
        data_array = self.data[self.column].values.reshape(-1, 1)
        if bandwidth_range is None:
            std = np.std(data_array)
            bandwidth_range = np.linspace(0.1 * std, 2 * std, 30)

        def kde_scorer(estimator, X):
            return np.mean(estimator.score_samples(X))

        grid = GridSearchCV(
            KernelDensity(kernel=self.kernel),
            {'bandwidth': bandwidth_range},
            cv=5,
            scoring=kde_scorer
        )

        grid.fit(data_array)
        self.bandwidth = grid.best_params_['bandwidth']
        return self.bandwidth

    def fit(self):
        data_array = self.data[self.column].values.reshape(-1, 1)
        if self.bandwidth is None:
            self.optimize_bandwidth()

        self.kde_model = KernelDensity(bandwidth=self.bandwidth, kernel=self.kernel)
        self.kde_model.fit(data_array)
        return self

    def estimate_pdf(self, x_range=None, points=1000):
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
        return stats.percentileofscore(self.data[self.column], value)

    def calculate_probability(self, lower_bound, upper_bound=None, points=1000):
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
        prob = self.calculate_probability(threshold, None)
        return 1.0 / prob if prob > 0 else float('inf')

    def estimate_conditional_expectation(self, threshold, upper_limit=None, points=1000):
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

        return numerator / denominator if denominator > 0 else float('inf')

    def compute_statistics(self):
        """Compute basic KDE statistics and return as a dictionary."""
        data_array = self.data[self.column].values
        mean = np.mean(data_array)
        median = np.median(data_array)
        x_pdf, y_pdf = self.estimate_pdf()
        mode = x_pdf[np.argmax(y_pdf)]
        variance = np.var(data_array)
        skewness = stats.skew(data_array)
        kurtosis = stats.kurtosis(data_array)
        min_val = np.min(data_array)
        max_val = np.max(data_array)

        q25 = np.percentile(data_array, 25)
        q75 = np.percentile(data_array, 75)
        iqr = q75 - q25

        return {
            "Count": len(data_array),
            "Mean": mean,
            "Median": median,
            "Mode": mode,
            "Min": min_val,
            "Max": max_val,
            "Variance": variance,
            "Skewness": skewness,
            "Kurtosis": kurtosis,
            "Interquartile Range (IQR)": {
                "Value": iqr,
                "25th Percentile": q25,
                "75th Percentile": q75
            },
            "Kernel": self.kernel,
            "Bandwidth": self.bandwidth,
            "Z-Score (Current)": (self.current - mean) / np.std(data_array),
            "Current Value": self.current
        }

    def compute_extended_statistics(self):
        """Compute extended KDE statistics and return as a dictionary."""
        current_value = self.current
        current_percentile = self.calculate_percentile(current_value)

        prob_below_15 = self.calculate_probability(0, 15) * 100
        prob_15_to_20 = self.calculate_probability(15, 20) * 100
        prob_20_to_30 = self.calculate_probability(20, 30) * 100
        prob_above_30 = self.calculate_probability(30, None) * 100

        return_period_30 = self.estimate_expected_return_period(30)
        return_period_40 = self.estimate_expected_return_period(40)
        return_period_50 = self.estimate_expected_return_period(50)

        expected_above_30 = self.estimate_conditional_expectation(30)
        expected_above_40 = self.estimate_conditional_expectation(40)

        data_array = self.data[self.column].values
        q1 = np.percentile(data_array, 1)
        q5 = np.percentile(data_array, 5)
        q95 = np.percentile(data_array, 95)
        q99 = np.percentile(data_array, 99)

        return {
            "Current Value": current_value,
            "Current Percentile": f"{current_percentile:.2f}%",
            "Probability Intervals": {
                "Below 15": f"{prob_below_15:.2f}%",
                "Between 15-20": f"{prob_15_to_20:.2f}%",
                "Between 20-30": f"{prob_20_to_30:.2f}%",
                "Above 30": f"{prob_above_30:.2f}%"
            },
            "Return Periods (Days)": {
                "Exceeding 30": return_period_30,
                "Exceeding 40": return_period_40,
                "Exceeding 50": return_period_50
            },
            "Expected Values": {
                "E[Data | Data > 30]": expected_above_30,
                "E[Data | Data > 40]": expected_above_40
            },
            "Key Quantiles": {
                "1%": q1,
                "5%": q5,
                "95%": q95,
                "99%": q99
            }
        }

    def analyze(self):
        """Return a dictionary of KDE analysis results for main.py to process."""
        return {
            f"KDE STATISTICS for column: {self.column}": self.compute_statistics(),
            "EXTENDED KDE STATISTICS": self.compute_extended_statistics()
        }

    def plot_kde_with_histogram(self, bins=30, filename="Plot/kde_plot.png"):
        # Create subdirectory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        data_array = self.data[self.column].values
        x_vals, pdf_vals = self.estimate_pdf()

        plt.figure(figsize=(12, 6))
        plt.hist(data_array, bins=bins, density=True, alpha=0.4, color='grey', label='Histogram')
        plt.plot(x_vals, pdf_vals, color='blue', lw=2, label='KDE')
        plt.fill_between(x_vals, pdf_vals, alpha=0.2, color='blue')

        mean = np.mean(data_array)
        median = np.median(data_array)
        mode = x_vals[np.argmax(pdf_vals)]
        current = self.current

        q25 = np.percentile(data_array, 25)
        q75 = np.percentile(data_array, 75)

        plt.axvline(mean, color='red', linestyle='--', label=f"Mean: {mean:.2f}")
        plt.axvline(median, color='green', linestyle='--', label=f"Median: {median:.2f}")
        plt.axvline(mode, color='purple', linestyle='--', label=f"Mode: {mode:.2f}")
        plt.axvline(current, color='black', linestyle='-', label=f"Current: {current:.2f}")

        # Plot shaded interquartile range
        plt.axvspan(q25, q75, color='orange', alpha=0.2, label='Interquartile Range (25%-75%)')

        z_score = (current - mean) / np.std(data_array)
        plt.title(f"KDE and Histogram (Z-Score: {z_score:.2f})")
        plt.xlabel(self.column)
        plt.ylabel("Density")
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def plot_boxplot_with_mean_iqr(self, filename="Plot/boxplot_with_mean_iqr.png"):
        # Create subdirectory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        data_array = self.data[self.column].values

        # Calculate key statistics
        mean = np.mean(data_array)
        q25 = np.percentile(data_array, 25)
        q75 = np.percentile(data_array, 75)
        iqr = q75 - q25

        # Create the box plot
        plt.figure(figsize=(12, 6))
        plt.boxplot(data_array, vert=False, patch_artist=True, boxprops=dict(facecolor='lightblue', color='black'),
                    whiskerprops=dict(color='black'), flierprops=dict(markerfacecolor='red', marker='o'))

        # Add the mean and IQR markers
        #plt.axvline(mean, color='red', linestyle='--', label=f"Mean: {mean:.2f}")
        #plt.axvspan(q25, q75, color='orange', alpha=0.2, label=f"IQR: {q25:.2f} - {q75:.2f}")

        # Set labels and title
        plt.xlabel(self.column)
        plt.title(f"Boxplot IQR")
        #plt.legend()
        plt.tight_layout()

        # Save the plot
        plt.savefig(filename)
        plt.close()