---
title: "VixStats Class Documentation"
description: "The VixStats class retrieves and analyzes historical data for the CBOE Volatility Index (VIX) from Yahoo Finance, providing several statistical measures and indicators."
---



The `VixStats` class is designed to retrieve and analyze historical data for the **CBOE Volatility Index (VIX)** from Yahoo Finance. It provides several descriptive statistical measures and rolling indicators that can be used to gain insights into the behavior of the VIX, which is often used as a gauge for market volatility.

## Attributes

- **ticker (str)**: The stock symbol for the VIX (`^VIX`), used to fetch data from Yahoo Finance.
- **start (str)**: The start date for the data retrieval in the format `YYYY-MM-DD`.
- **end (str)**: The end date for the data retrieval in the format `YYYY-MM-DD`.
- **data (DataFrame)**: The historical VIX data fetched from Yahoo Finance, containing the "Close" prices and other associated columns.

### Statistical Attributes

- **row_count (int)**: The number of rows in the dataset (i.e., the number of trading days in the selected period).
- **current_vix (float)**: The most recent (latest) closing VIX value.
- **mean_close (float)**: The mean (average) of the closing prices of VIX over the specified date range.
- **median_close (float)**: The median closing price of VIX.
- **mode_close (float)**: The mode (most frequent value) of the closing prices.
- **std_dev_close (float)**: The standard deviation of the closing prices.
- **z_score_close (float)**: The Z-score, which measures how far the most recent closing value is from the mean in terms of standard deviations.
- **percentile_25 (float)**: The 25th percentile (lower quartile) of the closing prices.
- **percentile_50 (float)**: The 50th percentile (median) of the closing prices.
- **percentile_75 (float)**: The 75th percentile (upper quartile) of the closing prices.
- **percentile_90 (float)**: The 90th percentile of the closing prices.
- **percentile_95 (float)**: The 95th percentile of the closing prices.
- **rolling_mean_7 (float)**: The 7-day rolling average of the closing prices.
- **rolling_mean_30 (float)**: The 30-day rolling average of the closing prices.
- **rsi (float)**: The 14-day Relative Strength Index (RSI), which is used to identify whether the VIX is overbought or oversold.

## Methods

### download_data()

This method fetches the historical VIX data from Yahoo Finance between the provided start and end dates. It calculates various statistical measures (mean, median, mode, percentiles, rolling averages, etc.) and stores them as attributes for later use. If there is an error during data retrieval, it catches the exception and prints an error message.

## Properties

The class provides several properties that return computed statistical values:

- **get_row_count**: Returns the number of rows (trading days) in the dataset.
- **get_current_vix**: Returns the most recent closing value of the VIX.
- **get_mean**: Returns the mean of the closing prices.
- **get_median**: Returns the median of the closing prices.
- **get_mode**: Returns the mode (most frequent value) of the closing prices.
- **get_std_dev**: Returns the standard deviation of the closing prices.
- **get_z_score**: Returns the Z-score for the most recent VIX closing value.
- **get_percentile_25**: Returns the 25th percentile of the closing prices.
- **get_percentile_75**: Returns the 75th percentile of the closing prices.
- **get_percentile_90**: Returns the 90th percentile of the closing prices.
- **get_percentile_95**: Returns the 95th percentile of the closing prices.
- **current_percentile**: Returns the percentile of the most recent VIX value within the historical distribution (i.e., what percentage of the historical closing values are lower than the current VIX).
- **get_rolling_mean7**: Returns the 7-day rolling mean of the closing prices.
- **get_rolling_mean30**: Returns the 30-day rolling mean of the closing prices.
- **get_rsi**: Returns the 14-day Relative Strength Index (RSI) value.
