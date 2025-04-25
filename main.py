from turtledemo.penrose import start

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from yahoo_finance_data_fetcher import YahooFinanceDataFetcher
from csv_data_loader import CSVDataLoader
from empirical_stats_descriptive import EmpiricalStatsDescriptive
from range_analyser import RangeAnalyser
from extreme_event_analyser import ExtremeEventAnalyser
from volatility_analyser import VolatilityAnalyser
from bar_range_analyser import BarRange
from kde_analyser import KDEAnalyser
from log_normal_analyser import LogNormalAnalyser
from mean_reversion_analyser import MeanReversionAnalyser

def generate_report(ticker, days, price_stats, returns_stats, range_analysis, extreme_analysis, volatility_analysis,
                    bar_range_analysis, kde_analysis, log_normal_analysis, mean_reversion_analysis):
    print("\n=== VIX Descriptive Statistics Report ===")
    print(f"Ticker: {ticker}")
    print(f"Period: {days} days\n")

    # Summary Statistics
    print("--- Summary Statistics ---")
    print("Price:")
    for key in ['row_count', 'current', 'mean', 'median', 'mode', 'min', 'max', 'std_dev', 'variance', 'mad', 'midrange']:
        value = price_stats.get(key)
        print(f"  {key}: {value:.2f}" if isinstance(value, (int, float)) else f"  {key}: {value}")
    print("Log Returns:")
    for key in ['row_count', 'current', 'mean', 'median', 'mode', 'min', 'max', 'std_dev', 'variance', 'mad', 'midrange']:
        value = returns_stats.get(key)
        print(f"  {key}: {value:.2f}" if isinstance(value, (int, float)) else f"  {key}: {value}")

    # Distribution Characteristics
    print("\n--- Distribution Characteristics ---")
    print("Price:")
    for key in ['skewness', 'kurtosis', 'entropy', 'geometric_mean', 'harmonic_mean']:
        value = price_stats.get(key)
        print(f"  {key}: {value:.2f}" if isinstance(value, (int, float)) else f"  {key}: {value}")
    print("Log Returns:")
    for key in ['skewness', 'kurtosis', 'entropy', 'geometric_mean', 'harmonic_mean']:
        value = returns_stats.get(key)
        print(f"  {key}: {value:.2f}" if isinstance(value, (int, float)) else f"  {key}: {value}")
    print("KDE Parameters:")
    for key in ['Kernel', 'Bandwidth']:
        value = kde_analysis[f"KDE STATISTICS for column: Close"].get(key)
        print(f"  {key}: {value}")
    print("Log-Normal Parameters:")
    for key in ['Log-Mean (μ)', 'Log-Std Dev (σ)', 'Shape (σ)', 'Scale (e^μ)', 'Mean', 'Median', 'Mode', 'Variance', 'Skewness', 'Kurtosis']:
        value = log_normal_analysis[f"LOG-NORMAL STATISTICS for column: Close"].get(key)
        print(f"  {key}: {value:.2f}" if isinstance(value, (int, float)) else f"  {key}: {value}")

    # Percentiles and Ranges
    print("\n--- Percentiles and Ranges ---")
    print("Price Percentiles:")
    for key in ['current_percentile', 'percentile_5', 'percentile_10', 'percentile_25', 'percentile_50', 'percentile_75', 'percentile_90', 'percentile_95', 'percentile_99', 'iqr', 'range']:
        value = price_stats.get(key)
        print(f"  {key}: {value:.2f}" if isinstance(value, (int, float)) else f"  {key}: {value}")
    print("Log Returns Percentiles:")
    for key in ['current_percentile', 'percentile_5', 'percentile_10', 'percentile_25', 'percentile_50', 'percentile_75', 'percentile_90', 'percentile_95', 'percentile_99', 'iqr', 'range']:
        value = returns_stats.get(key)
        print(f"  {key}: {value:.2f}" if isinstance(value, (int, float)) else f"  {key}: {value}")
    print("Range Analysis:")
    for key, value in range_analysis['counts'].items():
        print(f"  {key}: {value}")
    print(f"  Current Range: {range_analysis['current_range']}")
    print("KDE Percentiles and Intervals:")
    print(f"  Current Percentile: {kde_analysis['EXTENDED KDE STATISTICS']['Current Percentile']}")
    for key, value in kde_analysis['EXTENDED KDE STATISTICS']['Key Quantiles'].items():
        print(f"  {key}: {value:.2f}")
    print("Log-Normal Percentiles and Intervals:")
    print(f"  Current Percentile: {log_normal_analysis['EXTENDED LOG-NORMAL STATISTICS']['Current Percentile']}")
    for key, value in log_normal_analysis['EXTENDED LOG-NORMAL STATISTICS']['Key Quantiles'].items():
        print(f"  {key}: {value:.2f}")
    for key, value in log_normal_analysis['EXTENDED LOG-NORMAL STATISTICS']['95% Confidence Interval'].items():
        print(f"  {key}: {value:.2f}")

    # Volatility and Risk Metrics
    print("\n--- Volatility and Risk Metrics ---")
    print("Price Variability:")
    for key in ['cv', 'cv_percent', 'cv_abs_percent', 'sem', 'volatility_30d', 'rolling_mean_7', 'rolling_mean_30']:
        value = price_stats.get(key)
        print(f"  {key}: {value:.2f}" if isinstance(value, (int, float)) else f"  {key}: {value}")
    print("Log Returns Variability:")
    for key in ['cv', 'cv_percent', 'cv_abs_percent', 'sem', 'volatility_30d', 'rolling_mean_7', 'rolling_mean_30']:
        value = returns_stats.get(key)
        print(f"  {key}: {value:.2f}" if isinstance(value, (int, float)) else f"  {key}: {value}")
    print("Volatility Analysis:")
    print(f"  Rolling Volatility (30-day): {volatility_analysis['rolling_volatility']:.4f}")
    print(f"  Annualized Volatility: {volatility_analysis['annualized_volatility']:.4f}")
    print("Extreme Events:")
    print(f"  Count of Extreme Events (>40): {extreme_analysis['count']}")
    if extreme_analysis['count'] > 0:
        print("  Extreme Event Dates and Values:")
        for date, value in zip(extreme_analysis['dates'], extreme_analysis['values']):
            print(f"    Date: {date}, Value: {value:.2f}")
    print("Bar Range Analysis:")
    for key, value in bar_range_analysis['Summary Statistics'].items():
        print(f"  {key}: {value:.2f}")
    print("  Daily Range Distribution:")
    for range_str, count in bar_range_analysis['Range Histogram']['Daily Range Distribution'].items():
        print(f"    {range_str}: {count}")
    print("  Range Histogram Summary:")
    for key, value in bar_range_analysis['Range Histogram']['Summary'].items():
        print(f"    {key}: {value}")

    # Stationarity and Dynamics
    print("\n--- Stationarity and Dynamics ---")
    print("Price Dynamics:")
    for key in ['hurst', 'hurst_dfa', 'z_score', 'autocorrelation', 'trend_slope', 'adf_statistic', 'adf_pvalue', 'kpss_statistic', 'kpss_pvalue', 'kpss_warning', 'rsi', 'stability_score']:
        value = price_stats.get(key)
        print(f"  {key}: {value:.2f}" if isinstance(value, (int, float)) else f"  {key}: {value}")
    print("Log Returns Dynamics:")
    for key in ['hurst', 'hurst_dfa', 'z_score', 'autocorrelation', 'trend_slope', 'adf_statistic', 'adf_pvalue', 'kpss_statistic', 'kpss_pvalue', 'kpss_warning', 'rsi', 'stability_score']:
        value = returns_stats.get(key)
        print(f"  {key}: {value:.2f}" if isinstance(value, (int, float)) else f"  {key}: {value}")

    # Mean Reversion Properties
    print("\n--- Mean Reversion Properties ---")
    print("Mean Reversion Analysis:")
    print(f"  Current Value: {mean_reversion_analysis['current_value']:.2f}")
    print(f"  Mean: {mean_reversion_analysis['mean']:.2f}")
    print(f"  Distance to Mean: {mean_reversion_analysis['distance_to_mean']:.2f}")
    print(f"  Percent Deviation: {mean_reversion_analysis['percent_deviation']:.2f}%")
    print("  AR Model:")
    print(f"    Phi: {mean_reversion_analysis['ar_model']['phi']:.2f}")
    print(f"    Half-Life: {mean_reversion_analysis['ar_model']['half_life']:.2f} days")
    print(f"    Time to Mean Estimate: {mean_reversion_analysis['ar_model']['time_to_mean_estimate']:.2f} days")
    print("  OU Model:")
    print(f"    Theta: {mean_reversion_analysis['ou_model']['theta']:.2f}")
    print(f"    Mu: {mean_reversion_analysis['ou_model']['mu']:.2f}")
    print(f"    Sigma: {mean_reversion_analysis['ou_model']['sigma']:.2f}")
    print(f"    Half-Life: {mean_reversion_analysis['ou_model']['half_life']:.2f} days")
    print(f"    Time to Mean Estimate: {mean_reversion_analysis['ou_model']['time_to_mean_estimate']:.2f} days")
    print("Log-Normal Half-Life:")
    print(f"  Estimated Half-Life to Median: {log_normal_analysis['EXTENDED LOG-NORMAL STATISTICS']['Estimated Half-Life to Median']}")
    print("\nStationarity Warning: High Hurst exponent detected in price series. Mean reversion results may be unreliable.")

    # Probability and Return Periods
    print("\n--- Probability and Return Periods ---")
    print("KDE Probabilities:")
    for key, value in kde_analysis['EXTENDED KDE STATISTICS']['Probability Intervals'].items():
        print(f"  {key}: {value}")
    print("KDE Return Periods (Days):")
    if 'Return Periods (Days)' in kde_analysis['EXTENDED KDE STATISTICS']:
        for key, value in kde_analysis['EXTENDED KDE STATISTICS']['Return Periods (Days)'].items():
            print(f"  {key}: {value:.2f}")
    else:
        print("  Warning: Return Periods not available in KDE analysis.")
    print("KDE Expected Values:")
    for key, value in kde_analysis['EXTENDED KDE STATISTICS']['Expected Values'].items():
        print(f"  {key}: {value:.2f}")
    print("Log-Normal Probabilities:")
    for key, value in log_normal_analysis['EXTENDED LOG-NORMAL STATISTICS']['Probability Intervals'].items():
        print(f"  {key}: {value}")
    print("Log-Normal Return Periods (Days):")
    if 'Return Periods (Days)' in log_normal_analysis['EXTENDED LOG-NORMAL STATISTICS']:
        for key, value in log_normal_analysis['EXTENDED LOG-NORMAL STATISTICS']['Return Periods (Days)'].items():
            print(f"  {key}: {value:.2f}")
    else:
        print("  Warning: Return Periods not available in Log-Normal analysis.")
    print("Log-Normal Expected Values:")
    for key, value in log_normal_analysis['EXTENDED LOG-NORMAL STATISTICS']['Expected Values'].items():
        print(f"  {key}: {value:.2f}")

    print("\n--- Generating Plots ---")
    print("Plots saved: histogram.png, transition_matrix.png, durations.png, bar_range_histogram.png, kde_plot.png, boxplot_with_mean_iqr.png, mr_sim_paths.png")

def main(create_report=False, csv_filepath=None):
    ticker = "^VIX"
    print(f"Analysing ticker: {ticker}")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=5 * 365)


    # Load data
    if csv_filepath:
        print(f"Loading data from CSV: {csv_filepath}")
        loader = CSVDataLoader(csv_filepath, start=start_date, end=end_date)
        data = loader.load_data()
    else:
        print("Fetching data from Yahoo Finance")
        fetcher = YahooFinanceDataFetcher()  # Use default ticker "^VIX"
        data = fetcher.download_data(ticker=ticker, start=start_date, end=end_date)

    if data is None or data.empty:
        raise ValueError("Failed to fetch data or data is empty")

    print(f"Data type: {type(data)}")
    print(f"Data columns: {data.columns}")
    print(f"Close column type: {type(data['Close'])}")
    print(f"Data head: \n{data.head()}")
    print(f"Data tail: \n{data.tail()}")

    # Initialize analyzers
    price_analyser = EmpiricalStatsDescriptive(data)
    returns_analyser = price_analyser.analyse_returns(return_type='log')

    vix_ranges = [(0, 15), (15, 20), (20, 25), (25, 30), (30, 40), (40, float('inf'))]
    vix_labels = ['Low', 'Moderate', 'Elevated', 'High', 'Very High', 'Extreme']
    range_analyser = RangeAnalyser(data['Close'], vix_ranges, vix_labels)
    range_analyser.plot_histogram()
    range_analyser.plot_transition_matrix()
    range_analyser.plot_durations()

    extreme_analyser = ExtremeEventAnalyser(data['Close'], threshold=40)
    volatility_analyser = VolatilityAnalyser(data['Close'], window=30)
    bar_range_analyser = BarRange(data)
    bar_range_analyser.plot_histogram()

    kde_analyser = KDEAnalyser(data, column='Close')
    kde_analyser.plot_kde_with_histogram()
    kde_analyser.plot_boxplot_with_mean_iqr()

    log_normal_analyser = LogNormalAnalyser(data, column='Close')
    mean_reversion_analyser = MeanReversionAnalyser(data['Close'])
    mean_reversion_analyser.plot_simulated_paths()

    # Collect statistics
    stats_dict = {
        'price_stats': price_analyser.stats_summary(),
        'returns_stats': returns_analyser.stats_summary(),
        'range_analysis': range_analyser.range_summary(),
        'extreme_analysis': extreme_analyser.extreme_summary(),
        'volatility_analysis': volatility_analyser.volatility_summary(),
        'bar_range_analysis': bar_range_analyser.get_bar_range_statistics(),
        'kde_analysis': kde_analyser.analyze(),
        'log_normal_analysis': log_normal_analyser.analyze(),
        'mean_reversion_analysis': mean_reversion_analyser.summary()
    }

    if create_report:
        generate_report(
            ticker,
            (end_date - start_date).days,
            stats_dict['price_stats'],
            stats_dict['returns_stats'],
            stats_dict['range_analysis'],
            stats_dict['extreme_analysis'],
            stats_dict['volatility_analysis'],
            stats_dict['bar_range_analysis'],
            stats_dict['kde_analysis'],
            stats_dict['log_normal_analysis'],
            stats_dict['mean_reversion_analysis']
        )

    return stats_dict

if __name__ == "__main__":
    stats_dict = main(create_report=True, csv_filepath=None)