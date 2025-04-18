---
title: StatsDescriptive Class Documentation
---

# StatsDescriptive Class Documentation

## Overview

The **StatsDescriptive** class provides statistical analysis for a given dataset. It calculates a range of descriptive statistics and rolling indicators that help to understand the data's distribution, trends, and volatility. This class is designed to work with time-series data (e.g., stock prices, market indices, etc.) and provides insights through various measures, including central tendency, variability, and distribution shape.

## Features and Attributes

### 1. **Row Count**
   - **Description:** The total number of data points (rows) in the dataset.
   - **Usage:** `get_row_count`

### 2. **Current**
   - **Description:** The latest value in the dataset.
   - **Usage:** `get_current`

### 3. **Mean**
   - **Description:** The arithmetic average of the dataset.
   - **Usage:** `get_mean`

### 4. **Median**
   - **Description:** The middle value when the data points are sorted in ascending order.
   - **Usage:** `get_median`

### 5. **Mode**
   - **Description:** The value that appears most frequently in the dataset.
   - **Usage:** `get_mode`

### 6. **Min**
   - **Description:** The smallest value in the dataset.
   - **Usage:** `get_min`

### 7. **Max**
   - **Description:** The largest value in the dataset.
   - **Usage:** `get_max`

### 8. **Standard Deviation (Std Dev)**
   - **Description:** A measure of how spread out the data points are from the mean.
   - **Usage:** `get_std_dev`

### 9. **Variance**
   - **Description:** The square of the standard deviation, indicating the spread of the data.
   - **Usage:** `get_variance`

### 10. **MAD (Mean Absolute Deviation)**
   - **Description:** The average of the absolute differences between each data point and the mean.
   - **Usage:** `get_mad`

### 11. **Z-Score**
   - **Description:** A measure of how many standard deviations a data point is from the mean.
   - **Usage:** `get_z_score`

### 12. **Geometric Mean**
   - **Description:** The nth root of the product of all data points, used for multiplicative processes.
   - **Usage:** `get_geometric_mean`

### 13. **Harmonic Mean**
   - **Description:** The reciprocal of the arithmetic mean of the reciprocals of the data points. Useful for rates like average speed.
   - **Usage:** `get_harmonic_mean`

### 14. **Skewness**
   - **Description:** A measure of the asymmetry of the distribution.
   - **Usage:** `get_skewness`

### 15. **Kurtosis**
   - **Description:** A measure of the "tailedness" of the distribution.
   - **Usage:** `get_kurtosis`

### 16. **IQR (Interquartile Range)**
   - **Description:** The range between the 25th and 75th percentiles, representing the spread of the middle 50% of the data.
   - **Usage:** `get_iqr`

### 17. **Range**
   - **Description:** The difference between the maximum and minimum values in the dataset.
   - **Usage:** `get_range`

### 18. **Coefficient of Variation (CV)**
   - **Description:** The ratio of the standard deviation to the mean, representing the relative variability of the data.
   - **Usage:** `get_cv`

### 19. **Current Percentile**
   - **Description:** The percentile rank of the current value within the dataset.
   - **Usage:** `current_percentile`

### 20. **Percentiles**
   - **Description:** 
     - 25th Percentile: The value below which 25% of the data lies.
     - 50th Percentile: The median value.
     - 75th Percentile: The value below which 75% of the data lies.
     - 90th Percentile: The value below which 90% of the data lies.
     - 95th Percentile: The value below which 95% of the data lies.
   - **Usage:** 
     - `get_percentile_25`
     - `get_percentile_50`
     - `get_percentile_75`
     - `get_percentile_90`
     - `get_percentile_95`

### 21. **Rolling Means**
   - **Description:** 
     - 7-Day Rolling Mean: The moving average over a 7-day window.
     - 30-Day Rolling Mean: The moving average over a 30-day window.
   - **Usage:** 
     - `get_rolling_mean7`
     - `get_rolling_mean30`

### 22. **Volatility**
   - **Description:** The variability in the data over a defined window, typically a 30-day window.
   - **Usage:** `get_30_day_volatility`

### 23. **Stability Score**
   - **Description:** A measure of market stability, calculated based on volatility and standard deviation.
   - **Usage:** `get_stability_score`

### 24. **RSI (Relative Strength Index)**
   - **Description:** A momentum oscillator used to identify overbought or oversold conditions in the data.
   - **Usage:** `get_rsi`

## Methods

### `update_stats`
   - **Description:** This method computes the descriptive statistics based on the input dataset and updates the class attributes with the results.

### `calculate_mad`
   - **Description:** A helper function that calculates the Mean Absolute Deviation for the dataset.

### `calculate_geometric_mean`
   - **Description:** A helper function that calculates the geometric mean for the dataset, filtering out non-positive values.

### `calculate_harmonic_mean`
   - **Description:** A helper function that calculates the harmonic mean for the dataset, using only positive values.

## Example Usage

```python
from empirical_financial_stats import StatsDescriptive

data = [25, 30, 35, 40, 45, 50]  # Sample data
stats = StatsDescriptive(data)

print(f"Row Count: {stats.get_row_count}")
print(f"Current: {stats.get_current}")
print(f"Mean: {stats.get_mean}")
print(f"Median: {stats.get_median}")
print(f"Mode: {stats.get_mode}")
```