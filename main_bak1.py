from yahoo_finance_data_fetcher import YahooFinanceDataFetcher
from empirical_stats_descriptive import EmpiricalStatsDescriptive
from log_normal_analyser import LogNormalAnalyser
from kde_analyser import KDEAnalyzer
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
            print(f"{' ' * indent}{key}: {value:.2f}" if isinstance(value, np.float64) else f"{' ' * indent}{key}: {value}")

def main():
    # Fetch VIX data
    #vix_data = YahooFinanceDataFetcher().download_data("^VIX", start="2024-01-01")
    yfinance_data = YahooFinanceDataFetcher().download_data("^VIX")
    empirical_stats = EmpiricalStatsDescriptive(yfinance_data)
    # csv_loader = CSVDataLoader('VIX_History (1).csv')
    # yfinance_data = csv_loader.load_data()
    # empirical_stats = EmpiricalStatsDescriptive(yfinance_data)

    # Print basic statistics with a header
    print("\n" + "=" * 50)
    print("BASIC STATISTICS")
    print("=" * 50)
    print(f"Row Count: {empirical_stats.row_count}")
    print(f"Current: {empirical_stats.current:.2f}")
    print(f"Mean: {empirical_stats.mean:.2f}")
    print(f"Median: {empirical_stats.median:.2f}")
    print(f"Mode: {empirical_stats.mode:.2f}")
    print(f"Trimmed Mean: {empirical_stats.trimmed_mean:.2f}")
    print(f"Min: {empirical_stats.min:.2f}")
    print(f"Max: {empirical_stats.max:.2f}")
    print(f"Range: {empirical_stats.range:.2f}")
    print(f"Mid Range: {empirical_stats.midrange:.2f}")
    print(f"Standard Deviation: {empirical_stats.std_dev:.2f}")
    print(f"Variance: {empirical_stats.variance:.2f}")

    # Print percentiles with a header
    print("\n" + "=" * 50)
    print("PERCENTILE STATISTICS")
    print("=" * 50)
    print(f"Current Percentile: {empirical_stats.current_percentile:.2f}")
    print(f"Percentile 25: {empirical_stats.percentile_25:.2f}")
    print(f"Percentile 50: {empirical_stats.percentile_50:.2f}")
    print(f"Percentile 75: {empirical_stats.percentile_75:.2f}")
    print(f"Percentile 90: {empirical_stats.percentile_90:.2f}")
    print(f"Percentile 95: {empirical_stats.percentile_95:.2f}")
    print(f"IQR: {empirical_stats.iqr:.2f}")

    # Print advanced statistics with a header
    print("\n" + "=" * 50)
    print("ADVANCED STATISTICS")
    print("=" * 50)
    print(f"Shannon Entropy: {empirical_stats.entropy:.2f} bits")
    print(f"MAD: {empirical_stats.mad:.2f}")
    print(f"Raw Median Absolute Deviation: {empirical_stats.raw_medad:.2f}")
    print(f"Normalized Median Absolute Deviation: {empirical_stats.medad:.2f}")
    print(f"Z-Score: {empirical_stats.z_score:.2f}")
    print(f"Geometric Mean: {empirical_stats.geometric_mean:.2f}")
    print(f"Harmonic Mean: {empirical_stats.harmonic_mean:.2f}")
    print(f"Skewness: {empirical_stats.skewness:.2f}")
    print(f"Kurtosis: {empirical_stats.kurtosis:.2f}")
    print(f"Coefficient of Variation (ratio): {empirical_stats.cv:.2f}")
    print(f"Coefficient of Variation (percent): {empirical_stats.cv_percent:.2f}%")
    print(f"Hurst Exponent: {empirical_stats.hurst:.2f}")

    # Print time-series related statistics with a header
    print("\n" + "=" * 50)
    print("TIME-SERIES STATISTICS")
    print("=" * 50)
    print(f"Rolling 7 Day Mean: {empirical_stats.rolling_mean_7:.2f}")
    print(f"Rolling 30 Day Mean: {empirical_stats.rolling_mean_30:.2f}")
    print(f"30-Day Volatility: {empirical_stats.volatility_30d:.2f}")
    print(f"Stability Score: {empirical_stats.stability_score:.2f}")
    # Assuming empirical_stats is an instance of EmpiricalStatsDescriptive
    print(f"Time Weighted Mean (Linear): {empirical_stats.time_weighted_mean:.2f}")
    print(
        f"Time Weighted Mean (Quadratic): {empirical_stats.calculate_time_weighted_mean(empirical_stats.series, weight_type='quadratic'):.2f}")
    print(f"RSI: {empirical_stats.rsi:.2f}\n")

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

    #print(mr_analyzer.summary())
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
