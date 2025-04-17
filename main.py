from yahoo_finance_data_fetcher import YahooFinanceDataFetcher
from stats_descriptive import StatsDescriptive
from log_normal_analyser import LogNormalAnalyser

def main():
    vix_data = YahooFinanceDataFetcher().download_data()
    vix_stats = StatsDescriptive(vix_data)

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


    fetcher = YahooFinanceDataFetcher()
    df = fetcher.download_data()
    analyser = LogNormalAnalyser(df, column="Close")
    print(analyser.get_statistics())
    print(analyser.get_extended_statistics())


if __name__ == "__main__":
    main()