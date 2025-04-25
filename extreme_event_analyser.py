import pandas as pd

class ExtremeEventAnalyser:
    def __init__(self, data, threshold=40):
        if not isinstance(data, pd.Series):
            raise ValueError("Expected data to be a pandas Series")
        self.data = data
        self.threshold = threshold

    def analyze_extremes(self):
        extreme_events = self.data[self.data > self.threshold]
        summary = {
            'count': len(extreme_events),
            'dates': extreme_events.index.tolist(),
            'values': extreme_events.tolist()
        }
        return summary

    def extreme_summary(self):
        return self.analyze_extremes()