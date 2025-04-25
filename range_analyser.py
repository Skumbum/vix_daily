import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class RangeAnalyser:
    def __init__(self, data, ranges, labels):
        if len(ranges) != len(labels):
            raise ValueError("Number of ranges must match number of labels")

        # Handle DataFrame input by extracting the first column as a Series
        if isinstance(data, pd.DataFrame):
            print(f"RangeAnalyser: Input is a DataFrame with columns {data.columns}")
            if len(data.columns) == 1:
                data = data.iloc[:, 0]
            else:
                raise ValueError(f"Expected DataFrame with 1 column, got {len(data.columns)} columns")
        # Convert to Series if not already
        if not isinstance(data, pd.Series):
            data = pd.Series(data)

        if data.empty:
            raise ValueError("Data cannot be empty")
        if data.isna().any():
            raise ValueError("Data contains missing values")

        self.data = data
        self.ranges = ranges
        self.labels = labels
        self.categorized_data = self.categorize_data()
        self.range_counts = self.calculate_range_counts()

    def categorize_data(self):
        bins = [r[0] for r in self.ranges] + [self.ranges[-1][1]]  # Include upper bound of last range
        return pd.cut(self.data, bins=bins, labels=self.labels, include_lowest=True, right=False)

    def calculate_range_counts(self):
        counts = self.categorized_data.value_counts().reindex(self.labels, fill_value=0)
        return counts.sort_index()

    def get_range_distribution(self):
        total = self.range_counts.sum()
        if total == 0:
            return pd.Series(index=self.labels, data=0)
        distribution = self.range_counts / total
        return distribution.sort_index()

    @property
    def current_range(self):
        if self.data.empty:
            return None
        latest_value = float(self.data.iloc[-1])  # Ensure scalar
        print(f"Debug current_range: latest_value = {latest_value}, type = {type(latest_value)}")
        for i, (low, high) in enumerate(self.ranges):
            print(f"Debug current_range: low = {low}, high = {high}, range = {self.labels[i]}")
            # Handle infinite upper bound explicitly
            if np.isinf(high):
                if latest_value >= low:
                    return self.labels[i]
            else:
                if low <= latest_value <= high:
                    return self.labels[i]
        return None

    def histogram_output(self):
        """Output the histogram of the data ranges."""
        histogram = self.categorized_data.value_counts().reindex(self.labels, fill_value=0)
        for label in self.labels:
            print(f"{label}: {histogram.get(label, 0)}")

    def plot_histogram(self, filename="Plot/ra_histogram.png"):
        """Plot a histogram of the data ranges and save it as a PNG file."""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        histogram = self.categorized_data.value_counts().reindex(self.labels, fill_value=0)
        plt.figure(figsize=(8, 6))
        plt.bar(histogram.index, histogram.values, color='skyblue', edgecolor='black')
        plt.title("Histogram of VIX Ranges")
        plt.xlabel("Range")
        plt.ylabel("Count")
        plt.grid(True, alpha=0.3)
        plt.savefig(filename)
        plt.close()

    def plot_transition_matrix(self, filename="Plot/ra_transition_matrix.png"):
        """Plot a heatmap of the transition matrix and save it as a PNG file."""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        matrix = self.build_transition_matrix()
        plt.figure(figsize=(8, 6))
        sns.heatmap(matrix, annot=True, xticklabels=self.labels, yticklabels=self.labels,
                    cmap='Blues', fmt='.3f')
        plt.title("VIX Range Transition Matrix")
        plt.xlabel("Next Range")
        plt.ylabel("Current Range")
        plt.savefig(filename)
        plt.close()

    def plot_durations(self, filename="Plot/ra_durations.png"):
        """Plot a bar chart of average durations in each range and save it as a PNG file."""
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        durations = self.compute_average_durations()
        plt.figure(figsize=(8, 6))
        plt.bar(durations.keys(), durations.values(), color='lightgreen', edgecolor='black')
        plt.title("Average Duration in Each VIX Range")
        plt.xlabel("Range")
        plt.ylabel("Average Duration (Days)")
        plt.grid(True, alpha=0.3)
        plt.savefig(filename)
        plt.close()

    def build_transition_matrix(self):
        """Build a transition matrix of range states."""
        transition_matrix = np.zeros((len(self.labels), len(self.labels)))
        for i in range(1, len(self.categorized_data)):
            current_label = self.categorized_data.iloc[i - 1]
            next_label = self.categorized_data.iloc[i]
            current_index = self.labels.index(current_label)
            next_index = self.labels.index(next_label)
            transition_matrix[current_index, next_index] += 1

        # Normalize the matrix, handling zero-sum rows
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        # Avoid division by zero by setting zero-sum rows to uniform distribution
        transition_matrix = np.where(row_sums == 0, 1 / len(self.labels), transition_matrix / row_sums)
        return transition_matrix

    def compute_average_durations(self):
        """Compute the average durations spent in each range."""
        durations = {label: [] for label in self.labels}  # Store list of durations for each label
        current_label = self.categorized_data.iloc[0]
        duration = 1
        for i in range(1, len(self.categorized_data)):
            next_label = self.categorized_data.iloc[i]
            if next_label == current_label:
                duration += 1
            else:
                durations[current_label].append(duration)
                current_label = next_label
                duration = 1
        durations[current_label].append(duration)  # Append final duration

        # Compute average duration per label
        avg_durations = {label: np.mean(durations[label]) if durations[label] else 0
                         for label in self.labels}
        return avg_durations

    def track_extreme_events(self):
        """Track the extreme events in the data ranges."""
        return self.categorized_data.value_counts().reindex(self.labels, fill_value=0).to_dict()

    def range_summary(self):
        summary = {
            'counts': self.range_counts.to_dict(),
            'distribution': self.get_range_distribution().to_dict(),
            'current_range': self.current_range
        }
        return summary