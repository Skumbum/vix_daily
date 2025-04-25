import pandas as pd
import numpy as np

class VolatilityAnalyser:
    def __init__(self, data, window=30):
        if not isinstance(data, pd.Series):
            raise ValueError("Expected data to be a pandas Series")
        self.data = data
        self.window = window

    def calculate_volatility(self):
        # Compute log returns
        log_returns = np.log(self.data / self.data.shift(1)).dropna()
        # Rolling standard deviation of log returns
        rolling_vol = log_returns.rolling(window=self.window).std()
        # Annualize the volatility (assuming 252 trading days)
        annualized_vol = rolling_vol * np.sqrt(252)
        # Latest values
        latest_rolling_vol = rolling_vol.iloc[-1] if not rolling_vol.empty else None
        latest_annualized_vol = annualized_vol.iloc[-1] if not annualized_vol.empty else None
        return {
            'rolling_volatility': latest_rolling_vol,
            'annualized_volatility': latest_annualized_vol
        }

    def volatility_summary(self):
        return self.calculate_volatility()