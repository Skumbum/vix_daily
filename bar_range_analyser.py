import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class BarRange:
    def __init__(self, data):
        # Initialize with the data and calculate bar ranges (high - low)
        self.data = data
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

    def range_histogram_text(self, bins=100, cols=5):
        """
        Return a formatted string of the range histogram in a multi-column text format,
        including a summary of total counts and bins with zero counts.

        Args:
            bins (int): Number of bins for the histogram (default: 100).
            cols (int): Number of bins to display per row (default: 5).

        Returns:
            str: Formatted string containing the histogram and summary.
        """
        # Calculate histogram
        range_hist, bin_edges = np.histogram(self.data['Range'], bins=bins)
        total_count = len(self.data['Range'])

        # Build the output string
        output = []
        output.append("Daily Range Distribution (All Bins):")
        output.append(f"{'Range (Count)':<20}" * cols)
        output.append("-" * (20 * cols))

        # Display bins in a multi-column format
        for i in range(0, len(range_hist), cols):
            row_bins = range_hist[i:i + cols]
            row_edges = bin_edges[i:i + cols + 1]
            row_strs = []
            for j in range(len(row_bins)):
                count = row_bins[j]
                range_str = f"{row_edges[j]:.2f}-{row_edges[j + 1]:.2f}"
                display_str = f"{range_str} ({count})"
                row_strs.append(f"{display_str:<20}")
            output.append("".join(row_strs))

        # Summary stats
        zero_bins = sum(1 for count in range_hist if count == 0)
        output.append("")
        output.append("Summary:")
        output.append(f"Total Counts: {total_count}")
        output.append(f"Bins with Zero Counts: {zero_bins} ({zero_bins / bins * 100:.2f}%)")

        # Join all lines into a single string
        return "\n".join(output)

    def plot_histogram(self, bins=10):
        """Plot a histogram of the daily bar ranges and save it."""
        plt.figure(figsize=(10, 6))
        plt.hist(self.data['Range'], bins=bins, color='skyblue', edgecolor='black')
        plt.title("Histogram of Daily Bar Ranges")
        plt.xlabel("Range")
        plt.ylabel("Frequency")
        plt.grid(True, alpha=0.3)
        plt.savefig("bar_range_histogram.png")
        plt.close()