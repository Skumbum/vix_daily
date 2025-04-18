from yahoo_finance_data_fetcher import YahooFinanceDataFetcher
from empirical_stats_descriptive import EmpiricalStatsDescriptive
from log_normal_analyser import LogNormalAnalyser
from kde_analyzer import KDEAnalyzer
from mean_reversion_analyser import MeanReversionAnalyser
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    # Fetch VIX data
    #vix_data = YahooFinanceDataFetcher().download_data("^VIX", start="2024-01-01")
    vix_data = YahooFinanceDataFetcher().download_data()
    vix_stats = EmpiricalStatsDescriptive(vix_data)

    # Print basic statistics with a header
    print("\n" + "=" * 50)
    print("BASIC STATISTICS")
    print("=" * 50)
    print(f"Row Count: {vix_stats.get_row_count}")
    print(f"Current: {vix_stats.get_current:.2f}")
    print(f"Mean: {vix_stats.get_mean:.2f}")
    print(f"Median: {vix_stats.get_median:.2f}")
    print(f"Mode: {vix_stats.get_mode:.2f}")
    print(f"Min: {vix_stats.get_min:.2f}")
    print(f"Max: {vix_stats.get_max:.2f}")
    print(f"Range: {vix_stats.get_range:.2f}")
    print(f"Standard Deviation: {vix_stats.get_std_dev:.2f}")
    print(f"Variance: {vix_stats.get_variance:.2f}")

    # Print percentiles with a header
    print("\n" + "=" * 50)
    print("PERCENTILE STATISTICS")
    print("=" * 50)
    print(f"Current Percentile: {vix_stats.current_percentile:.2f}")
    print(f"Percentile 25: {vix_stats.get_percentile_25:.2f}")
    print(f"Percentile 50: {vix_stats.get_percentile_50:.2f}")
    print(f"Percentile 75: {vix_stats.get_percentile_75:.2f}")
    print(f"Percentile 90: {vix_stats.get_percentile_90:.2f}")
    print(f"Percentile 95: {vix_stats.get_percentile_95:.2f}")
    print(f"IQR: {vix_stats.get_iqr:.2f}")

    # Print advanced statistics with a header
    print("\n" + "=" * 50)
    print("ADVANCED STATISTICS")
    print("=" * 50)
    print(f"MAD: {vix_stats.get_mad:.2f}")
    print(f"Z-Score: {vix_stats.get_z_score:.2f}")
    print(f"Geometric Mean: {vix_stats.get_geometric_mean:.2f}")
    print(f"Harmonic Mean: {vix_stats.get_harmonic_mean:.2f}")
    print(f"Skewness: {vix_stats.get_skewness:.2f}")
    print(f"Kurtosis: {vix_stats.get_kurtosis:.2f}")
    print(f"Coefficient of Variation (CV): {vix_stats.get_cv:.2f}")

    # Print time-series related statistics with a header
    print("\n" + "=" * 50)
    print("TIME-SERIES STATISTICS")
    print("=" * 50)
    print(f"Rolling 7 Day Mean: {vix_stats.get_rolling_mean7:.2f}")
    print(f"Rolling 30 Day Mean: {vix_stats.get_rolling_mean30:.2f}")
    print(f"30-Day Volatility: {vix_stats.get_30_day_volatility:.2f}")
    print(f"Stability Score: {vix_stats.get_stability_score:.2f}")
    print(f"RSI: {vix_stats.get_rsi:.2f}\n")

    # Create LogNormalAnalyser for VIX data and print basic statistics
    print("\n" + "=" * 50)
    print("LOG-NORMAL ANALYSIS FOR VIX DATA")
    print("=" * 50)
    vix_analyser = LogNormalAnalyser(vix_data, column="Close")
    print(vix_analyser.get_statistics())
    print(vix_analyser.get_extended_statistics())


    kde_analyzer = KDEAnalyzer(vix_data, column="Close")
    lognorm_analyzer = LogNormalAnalyser(vix_data, column="Close")

    # KDE Analysis
    print("\n" + "=" * 50)
    print("KDE ANALYSIS FOR VIX DATA")
    print("=" * 50)
    print(kde_analyzer.get_statistics())
    print(kde_analyzer.get_extended_statistics())


    kde_analyzer.plot_kde_with_histogram()
    kde_analyzer.plot_boxplot_with_mean_iqr()
    #kde_analyzer.save_kde_plot1("kde_plot1.png")

    # Create both analyzers
    mr_analyzer = MeanReversionAnalyser(vix_data["Close"].values, mean=kde_analyzer.data[kde_analyzer.column].mean())

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

    print(mr_analyzer.summary())


if __name__ == "__main__":
    main()