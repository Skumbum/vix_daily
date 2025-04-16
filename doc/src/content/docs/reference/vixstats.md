---
title: "VixStats Class Documentation"
description: "The VixStats class retrieves and analyzes historical data for the CBOE Volatility Index (VIX) from Yahoo Finance, providing several statistical measures and indicators."
---

# VixStats Class Documentation

The `VixStats` class is designed to retrieve and analyze historical data for the **CBOE Volatility Index (VIX)** from Yahoo Finance. It provides several descriptive statistical measures and rolling indicators that can be used to gain insights into the behavior of the VIX, which is often used as a gauge for market volatility.

## Attributes

- **data** (`DataFrame`): The historical VIX data fetched from Yahoo Finance, containing the "Close" prices and other associated columns.

### Statistical Attributes

- **row_count** (`int`): The number of rows in the dataset (i.e., the number of trading days in the selected period).
- **current_vix** (`float`): The most recent (latest) closing VIX value.
- **mean_close** (`float`): The mean (average) of the closing prices of VIX over the specified date range.
- **median_close** (`float`): The median closing price of VIX.
- **mode_close** (`float`): The mode (most frequent value) of the closing prices.
- **std_dev_close** (`float`): The standard deviation of the closing prices.
- **percentile_25** (`float`): The 25th percentile (lower quartile) of the closing prices.
- **percentile_50** (`float`): The 50th percentile (median) of the closing prices.
- **percentile_75** (`float`): The 75th percentile (upper quartile) of the closing prices.
- **percentile_90** (`float`): The 90th percentile of the closing prices.
- **percentile_95** (`float`): The 95th percentile of the closing prices.
- **rolling_mean_7** (`float`): The 7-day rolling average of the closing prices.
- **rolling_mean_30** (`float`): The 30-day rolling average of the closing prices.
- **rsi** (`float`): The 14-day Relative Strength Index (RSI), which is used to identify whether the VIX is overbought or oversold.

## Methods

### `update_stats()`

This method calculates various statistical measures (mean, median, mode, percentiles, rolling averages, etc.) from the VIX closing prices and stores them as attributes for later use.

### `calculate_rsi()`

This method calculates the **Relative Strength Index (RSI)**, a momentum oscillator used to identify overbought or oversold conditions, based on the 14-day period. It computes the gains and losses, averages them, and uses them to calculate the RSI.

## Properties

The class provides several properties that return computed statistical values:

- **get_row_count**: Returns the number of rows (trading days) in the dataset.
- **get_current_vix**: Returns the most recent closing value of the VIX.
- **get_mean**: Returns the mean of the closing prices.
- **get_median**: Returns the median of the closing prices.
- **get_mode**: Returns the mode (most frequent value) of the closing prices.
- **get_std_dev**: Returns the standard deviation of the closing prices.
- **get_z_score**: Returns the Z-score for the most recent VIX closing value, which indicates how many standard deviations away it is from the mean.
- **get_percentile_25**: Returns the 25th percentile of the closing prices.
- **get_percentile_75**: Returns the 75th percentile of the closing prices.
- **get_percentile_90**: Returns the 90th percentile of the closing prices.
- **get_percentile_95**: Returns the 95th percentile of the closing prices.
- **current_percentile**: Returns the percentile of the most recent VIX value within the historical distribution (i.e., what percentage of the historical closing values are lower than the current VIX).
- **get_rolling_mean7**: Returns the 7-day rolling mean of the closing prices.
- **get_rolling_mean30**: Returns the 30-day rolling mean of the closing prices.
- **get_rsi**: Returns the 14-day Relative Strength Index (RSI) value.

## Usage Example

Here’s how you might use the `VixStats` class:

```python
import yfinance as yf
import pandas as pd

# Retrieve historical VIX data
vix_data = yf.download("^VIX", start="2020-01-01", end="2025-01-01")

# Instantiate VixStats class with the data
vix_stats = VixStats(vix_data)

# Access calculated statistics
print(f"Current VIX: {vix_stats.get_current_vix}")
print(f"7-day Rolling Mean: {vix_stats.get_rolling_mean7}")
print(f"14-day RSI: {vix_stats.get_rsi}")
print(f"Percentile 25: {vix_stats.get_percentile_25}")
