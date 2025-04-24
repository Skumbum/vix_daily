from yahoo_finance_data_fetcher import YahooFinanceDataFetcher
from empirical_stats_descriptive import EmpiricalStatsDescriptive
from log_normal_analyser import LogNormalAnalyser
from kde_analyzer import KDEAnalyzer
from mean_reversion_analyser import MeanReversionAnalyser
from range_analyser import RangeAnalyser
from bar_range_analyser import BarRange
from csv_data_loader import CSVDataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def print_dict(d, indent=0):
    for key, value in d.items():
        if isinstance(value, dict):  # If the value is a dictionary, recurse
            print(f"{' ' * indent}{key}:")
            print_dict(value, indent + 2)
        else:  # If the value is a float or other type, print it
            print(f"{' ' * indent}{key}: {value:.2f}" if isinstance(value, (np.float64, float)) else f"{' ' * indent}{key}: {value}")


def main():
    # Fetch VIX data
    #vix_data = YahooFinanceDataFetcher().download_data("^VIX", start="2024-01-01")
    yfinance_data = YahooFinanceDataFetcher().download_data("^VIX")
    empirical_stats = EmpiricalStatsDescriptive(yfinance_data)
    # csv_loader = CSVDataLoader('VIX_History (1).csv')
    # yfinance_data = csv_loader.load_data()
    # empirical_stats = EmpiricalStatsDescriptive(yfinance_data)

    # Get stats summary
    stats = empirical_stats.stats_summary()

    # Basic Statistics
    print("\n" + "=" * 50)
    print("BASIC STATISTICS")
    print("=" * 50)
    basic_stats = {
        "Row Count": stats["row_count"],
        "Current": stats["current"],
        "Mean": stats["mean"],
        "Median": stats["median"],
        "Mode": stats["mode"],
        "Trimmed Mean": stats["trimmed_mean"],
        "Min": stats["min"],
        "Max": stats["max"],
        "Range": stats["range"],
        "Mid Range": stats["midrange"]
    }
    print_dict(basic_stats)

    # Dispersion Statistics
    print("\n" + "=" * 50)
    print("DISPERSION STATISTICS")
    print("=" * 50)
    dispersion_stats = {
        "Standard Deviation": stats["std_dev"],
        "Variance": stats["variance"],
        "Coefficient of Variation (Ratio)": stats["cv"],
        "Coefficient of Variation (Percent)": stats["cv_percent"],
        "Coefficient of Variation (Abs Percent)": stats["cv_abs_percent"],
        "Standard Error of Mean": stats["sem"],
        "Mean Absolute Deviation": stats["mad"],
        "Raw Median Absolute Deviation": stats["raw_medad"],
        "Normalized Median Absolute Deviation": stats["medad"]
    }
    print_dict(dispersion_stats)

    # Percentile Statistics
    print("\n" + "=" * 50)
    print("PERCENTILE STATISTICS")
    print("=" * 50)
    percentile_stats = {
        "Current Percentile": stats["current_percentile"],
        "Percentile 25": stats["percentile_25"],
        "Percentile 50": stats["percentile_50"],
        "Percentile 75": stats["percentile_75"],
        "Percentile 90": stats["percentile_90"],
        "Percentile 95": stats["percentile_95"],
        "IQR": stats["iqr"]
    }
    print_dict(percentile_stats)

    # Distributional Statistics
    print("\n" + "=" * 50)
    print("DISTRIBUTIONAL STATISTICS")
    print("=" * 50)
    distributional_stats = {
        "Shannon Entropy (bits)": stats["entropy"],
        "Z-Score": stats["z_score"],
        "Geometric Mean": stats["geometric_mean"],
        "Harmonic Mean": stats["harmonic_mean"],
        "Skewness": stats["skewness"],
        "Kurtosis": stats["kurtosis"],
        "Hurst Exponent": stats["hurst"]
    }
    print_dict(distributional_stats)

    # Time-Series Statistics
    print("\n" + "=" * 50)
    print("TIME-SERIES STATISTICS")
    print("=" * 50)
    time_series_stats = {
        "Rolling 7 Day Mean": stats["rolling_mean_7"],
        "Rolling 30 Day Mean": stats["rolling_mean_30"],
        "30-Day Volatility": stats["volatility_30d"],
        "Stability Score": stats["stability_score"],
        "RSI": stats["rsi"],
        "Time Weighted Means": {
            "Linear": stats["time_weighted_mean"],
            "Exponential": empirical_stats.get_time_weighted_means(empirical_stats.series)["exponential"],
            "Quadratic": empirical_stats.get_time_weighted_means(empirical_stats.series)["quadratic"]
        }
    }
    print_dict(time_series_stats)

    # Returns Analysis (Log Returns)
    print("\n" + "=" * 50)
    print("LOG RETURNS ANALYSIS")
    print("=" * 50)
    returns_stats = empirical_stats.analyze_returns(return_type="log").stats_summary()
    returns_summary = {
        "Row Count": returns_stats["row_count"],
        "Mean": returns_stats["mean"],
        "Standard Deviation": returns_stats["std_dev"],
        "Coefficient of Variation (Abs Percent)": returns_stats["cv_abs_percent"],
        "Standard Error of Mean": returns_stats["sem"],
        "Skewness": returns_stats["skewness"],
        "Kurtosis": returns_stats["kurtosis"],
        "Hurst Exponent": returns_stats["hurst"]
    }
    print_dict(returns_summary)

    # Create LogNormalAnalyser for VIX data and print basic statistics
    print("\n" + "=" * 50)
    print("LOG-NORMAL ANALYSIS")
    print("=" * 50)
    vix_analyser = LogNormalAnalyser(yfinance_data, column="Close")
    print(vix_analyser.get_statistics())
    print(vix_analyser.get_extended_statistics())

    kde_analyzer = KDEAnalyzer(yfinance_data, column="Close")
    lognorm_analyzer = LogNormalAnalyser(yfinance_data, column="Close")

    # KDE Analysis
    print("\n" + "=" * 50)
    print("KDE ANALYSIS")
    print("=" * 50)
    print(kde_analyzer.get_statistics())
    print(kde_analyzer.get_extended_statistics())

    kde_analyzer.plot_kde_with_histogram()
    kde_analyzer.plot_boxplot_with_mean_iqr()

    # Create both analyzers
    mr_analyzer = MeanReversionAnalyser(yfinance_data["Close"].values, mean=kde_analyzer.data[kde_analyzer.column].mean())

    # Get distribution insights from KDE
    kde_stats = kde_analyzer.get_statistics()
    kde_extended = kde_analyzer.get_extended_statistics()

    # Get time-based insights from mean reversion
    mr_params = mr_analyzer.estimate_ar_parameters()
    mr_summary = mr_analyzer.summary()

    # Cross-validate the results
    expected_time_to_mean = mr_analyzer.estimate_time_to_mean()
    return_periods = {
        "30": kde_analyzer.estimate_expected_return_period(30),
        "40": kde_analyzer.estimate_expected_return_period(40),
        "mean": 1 / kde_analyzer.calculate_probability(mr_analyzer.mean * 0.99, mr_analyzer.mean * 1.01)
    }

    # Compare the approaches
    print(f"\nKDE return period to mean: {return_periods['mean']:.2f} days")
    print(f"Mean reversion time estimate: {expected_time_to_mean:.2f} days\n")

    print("\n" + "=" * 50)
    print("MEAN REVERSION ANALYSIS")
    print("=" * 50)
    print_dict(mr_analyzer.summary())
    mr_analyzer.plot_simulated_paths()

    # Range Analysis
    print("\n" + "=" * 50)
    print("RANGE ANALYSIS")
    print("=" * 50)

    # Define quantile-based ranges and labels
    ranges = [(0, 13.85), (13.85, 22.83), (22.83, 33.20), (33.20, 100)]
    labels = ["Low", "Moderate", "High", "Extreme"]

    # Initialize RangeAnalyser
    analyser = RangeAnalyser(yfinance_data["Close"], ranges, labels)

    # Output results
    analyser.histogram_output()
    analyser.plot_histogram()
    print("\nTransition Matrix:")
    print(analyser.build_transition_matrix())
    analyser.plot_transition_matrix()
    print("\nAverage Durations:")
    print(analyser.compute_average_durations())
    analyser.plot_durations()
    print("\nExtreme Events:")
    print(analyser.track_extreme_events())

    # Instantiate BarRange analyser
    analyser = BarRange(yfinance_data)

    # Get summary statistics (including ATR)
    stats = analyser.summary_statistics()
    print("\n")

    # Print the statistics using the existing print_dict function
    print("=" * 50)
    print("BAR RANGE SUMMARY STATISTICS")
    print("=" * 50)
    print_dict(stats)

    print("\n")

    # Print Text-based Histogram
    range_histogram_output = analyser.range_histogram_text(bins=100, cols=5)
    print(range_histogram_output)

    # Plot and save the histogram
    analyser.plot_histogram(bins=100)


if __name__ == "__main__":
    main()