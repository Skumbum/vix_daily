#main.py

from yahoo_finance_data_fetcher import YahooFinanceDataFetcher
from stats_vix_descriptive import StatsDescriptive

def main():
    vix_data = YahooFinanceDataFetcher().download_data()
    vix_stats = StatsDescriptive(vix_data)

    print(f"Row Count: {vix_stats.get_row_count}")
    print(f"Current Vix: {vix_stats.get_current}")
    print(f"Mean: {vix_stats.get_mean}")
    print(f"Median: {vix_stats.get_median}")
    print(f"Mode: {vix_stats.get_mode}")
    print(f"Min: {vix_stats.get_min}")
    print(f"Max: {vix_stats.get_max}")
    print(f"Std Dev: {vix_stats.get_std_dev}")
    print(f"Variance: {vix_stats.get_variance}")
    print(f"MAD (Mean Absolute Deviation): {vix_stats.get_mad}")
    print(f"Z-Score: {vix_stats.get_z_score}\n")

    print(f"Geometric Mean: {vix_stats.get_geom_mean}")
    print(f"Harmonic Mean: {vix_stats.get_harm_mean}")
    print(f"Skewness: {vix_stats.get_skewness}")
    print(f"Kurtosis: {vix_stats.get_kurtosis}")
    print(f"IQR (Interquartile Range): {vix_stats.get_iqr}\n")

    print(f"Current Percentile: {vix_stats.current_percentile}")
    print(f"Percentile 25: {vix_stats.get_percentile_25}")
    print(f"Percentile 50: {vix_stats.get_percentile_50}")
    print(f"Percentile 75: {vix_stats.get_percentile_75}")
    print(f"Percentile 90: {vix_stats.get_percentile_90}")
    print(f"Percentile 95: {vix_stats.get_percentile_95}\n")

    print(f"Rolling 7 Day Mean: {vix_stats.get_rolling_mean7}")
    print(f"Rolling 30 Day Mean: {vix_stats.get_rolling_mean30}")
    print(f"30-Day Volatility: {vix_stats.get_volatility_30d}")
    print(f"Stability Score: {vix_stats.get_stability_score}")
    print(f"RSI: {vix_stats.get_ta_rsi}")

if __name__ == "__main__":
    main()
