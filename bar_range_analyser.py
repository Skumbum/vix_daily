import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class BarRange:
    def __init__(self, data, bins=50):
        # Initialize with the data and calculate bar ranges (high - low)
        self.data = data
        self.bins = bins
        self.data['Range'] = self.data['High'] - self.data['Low']
        self.data['ATR'] = self.calculate_atr()

    def calculate_atr(self, period=14):
        """Calculate the Average True Range (ATR) for the data."""
        high_low = self.data['High'] - self.data['Low']
        high_close = abs(self.data['High'] - self.data['Close'].shift())
        low_close = abs(self.data['Low'] - self.data['Close'].shift())

        true_range = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = true_range.max(axis=1)

        atr = true_range.rolling(window=period).mean()
        return atr

    def summary_statistics(self):
        """Return a dictionary of summary statistics."""
        stats = {
            "Mean Daily Range": self.data['Range'].mean(),
            "Median Daily Range": self.data['Range'].median(),
            "Max Daily Range": self.data['Range'].max(),
            "Min Daily Range": self.data['Range'].min(),
            "Standard Deviation": self.data['Range'].std(),
            "Average True Range (ATR)": self.data['ATR'].iloc[-1]  # Latest ATR value
        }
        return stats

    def compute_range_histogram(self):
        """
        Compute the range histogram and return as a dictionary.

        Returns:
            dict: Histogram data and summary statistics.
        """
        # Calculate histogram
        range_hist, bin_edges = np.histogram(self.data['Range'], bins=self.bins)
        total_count = len(self.data['Range'])

        # Build histogram data as a dictionary
        histogram_data = {}
        for i in range(len(range_hist)):
            range_str = f"{bin_edges[i]:.2f}-{bin_edges[i + 1]:.2f}"
            histogram_data[range_str] = range_hist[i]

        # Summary stats
        zero_bins = sum(1 for count in range_hist if count == 0)
        summary = {
            "Total Counts": total_count,
            "Bins with Zero Counts": f"{zero_bins} ({zero_bins / self.bins * 100:.2f}%)"
        }

        return {
            "Daily Range Distribution": histogram_data,
            "Summary": summary
        }

    def get_bar_range_statistics(self):
        """Return a dictionary of all bar range statistics."""
        return {
            "Summary Statistics": self.summary_statistics(),
            "Range Histogram": self.compute_range_histogram()
        }

    def plot_histogram(self, bins=10, filename="Plot/bar_range_histogram.png"):
        """Plot a histogram of the daily bar ranges and save it."""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.figure(figsize=(10, 6))
        plt.hist(self.data['Range'], bins=bins, color='skyblue', edgecolor='black')
        plt.title("Histogram of Daily Bar Ranges")
        plt.xlabel("Range")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)
        plt.savefig(filename)
        plt.close()