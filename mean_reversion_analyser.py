import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import minimize
import warnings


class MeanReversionAnalyser:
    """
    A class for analyzing mean reversion properties of time series data
    and estimating time to return to mean value.
    """

    def __init__(self, time_series, mean=None, dt=1):
        """
        Initialize with time series data

        Parameters:
        -----------
        time_series : array-like
            The time series data (typically closing prices)
        mean : float, optional
            Custom mean value (if None, calculated from time_series)
        dt : float, optional
            Time step between observations (default=1 for daily data)
        """
        self.time_series = np.array(time_series)
        self.mean = mean if mean is not None else np.mean(self.time_series)
        self.current_value = self.time_series[-1]
        self.dt = dt
        self.n = len(time_series)

        # Initialize parameters
        self.phi = None
        self.half_life = None
        self.ou_theta = None
        self.ou_mu = None
        self.ou_sigma = None

    def estimate_ar_parameters(self):
        """
        Estimate AR(1) coefficient and related parameters

        Returns:
        --------
        tuple: (phi, half_life, mean)
            phi: AR(1) coefficient
            half_life: Half-life in time units
            mean: Long-term mean
        """
        # Lag-1 regression to get phi
        X = self.time_series[:-1]
        y = self.time_series[1:]
        X = sm.add_constant(X)

        try:
            model = sm.OLS(y, X).fit()
            constant = model.params[0]
            self.phi = model.params[1]  # AR coefficient

            # Calculate implied long-term mean
            implied_mean = constant / (1 - self.phi)

            # Calculate half-life
            self.half_life = -np.log(2) / np.log(abs(self.phi)) if self.phi < 1 else np.inf

            return self.phi, self.half_life, implied_mean
        except:
            warnings.warn("AR parameter estimation failed. Using fallback values.")
            self.phi = 0.9  # Fallback value
            self.half_life = -np.log(2) / np.log(0.9)
            return self.phi, self.half_life, self.mean

    def estimate_ou_parameters(self):
        """
        Estimate Ornstein-Uhlenbeck process parameters using maximum likelihood

        Returns:
        --------
        tuple: (theta, mu, sigma)
            theta: Mean reversion rate
            mu: Long-term mean
            sigma: Volatility parameter
        """
        # Calculate returns
        returns = np.diff(self.time_series)
        levels = self.time_series[:-1]

        # Define negative log-likelihood function for O-U process
        def neg_log_likelihood(params):
            theta, mu, sigma = params
            if theta <= 0 or sigma <= 0:  # Ensure valid parameters
                return 1e10

            # O-U process likelihood
            expected_returns = theta * (mu - levels) * self.dt
            variance = sigma ** 2 * self.dt
            loglik = -np.sum(norm.logpdf(returns, loc=expected_returns, scale=np.sqrt(variance)))
            return loglik

        # Initial guess for parameters
        initial_guess = [0.1, self.mean, np.std(returns) / np.sqrt(self.dt)]

        # Optimize to find parameters
        try:
            result = minimize(neg_log_likelihood, initial_guess,
                              method='L-BFGS-B',
                              bounds=[(1e-8, None), (None, None), (1e-8, None)])

            self.ou_theta, self.ou_mu, self.ou_sigma = result.x
            return self.ou_theta, self.ou_mu, self.ou_sigma
        except:
            warnings.warn("O-U parameter estimation failed. Using fallback values.")
            self.ou_theta = 0.1
            self.ou_mu = self.mean
            self.ou_sigma = np.std(returns) / np.sqrt(self.dt)
            return self.ou_theta, self.ou_mu, self.ou_sigma

    def estimate_time_to_mean(self, current_value=None, threshold=0.05, method='ar'):
        """
        Estimate expected time to return within threshold of mean

        Parameters:
        -----------
        current_value : float, optional
            Starting value (default is last value in series)
        threshold : float, optional
            How close to the mean (as fraction of distance)
        method : str, optional
            Estimation method: 'ar' or 'ou'

        Returns:
        --------
        float: Estimated time units to reach mean within threshold
        """
        if current_value is None:
            current_value = self.current_value

        distance = abs(current_value - self.mean)
        target_distance = distance * threshold

        if method == 'ar':
            if self.phi is None:
                self.estimate_ar_parameters()

            # Time to cover (1-threshold) of the distance
            if self.phi >= 1:
                return np.inf  # Not mean-reverting
            t = np.log(target_distance / distance) / np.log(self.phi)
            return max(0, t)

        elif method == 'ou':
            if self.ou_theta is None:
                self.estimate_ou_parameters()

            # O-U expected time calculation
            t = -np.log(threshold) / self.ou_theta
            return max(0, t)

        else:
            raise ValueError("Method must be 'ar' or 'ou'")

    def simulate_paths(self, current_value=None, periods=60, n_simulations=1000, method='ar'):
        """
        Monte Carlo simulation of potential future paths

        Parameters:
        -----------
        current_value : float, optional
            Starting value (default is last value in series)
        periods : int, optional
            Number of periods to simulate
        n_simulations : int, optional
            Number of simulation paths
        method : str, optional
            Simulation method: 'ar' or 'ou'

        Returns:
        --------
        np.ndarray: Matrix of simulated paths (n_simulations × periods)
        """
        if current_value is None:
            current_value = self.current_value

        # Initialize the paths
        paths = np.zeros((n_simulations, periods + 1))
        paths[:, 0] = current_value

        if method == 'ar':
            if self.phi is None:
                self.estimate_ar_parameters()

            # Calculate innovation standard deviation
            innov_std = np.std(self.time_series[1:] - self.mean - self.phi * (self.time_series[:-1] - self.mean))

            # Generate AR(1) paths
            for t in range(1, periods + 1):
                innovations = np.random.normal(0, innov_std, n_simulations)
                paths[:, t] = self.mean + self.phi * (paths[:, t - 1] - self.mean) + innovations

        elif method == 'ou':
            if self.ou_theta is None:
                self.estimate_ou_parameters()

            # Generate O-U paths
            for t in range(1, periods + 1):
                drift = self.ou_theta * (self.ou_mu - paths[:, t - 1]) * self.dt
                diffusion = self.ou_sigma * np.sqrt(self.dt) * np.random.normal(0, 1, n_simulations)
                paths[:, t] = paths[:, t - 1] + drift + diffusion

        else:
            raise ValueError("Method must be 'ar' or 'ou'")

        return paths

    def first_passage_distribution(self, current_value=None, periods=60, n_simulations=10000, method='ar'):
        """
        Calculate probability distribution of first passage time to mean

        Parameters:
        -----------
        current_value : float, optional
            Starting value (default is last value in series)
        periods : int, optional
            Maximum periods to simulate
        n_simulations : int, optional
            Number of simulation paths
        method : str, optional
            Simulation method: 'ar' or 'ou'

        Returns:
        --------
        dict: First passage time distribution statistics
        """
        if current_value is None:
            current_value = self.current_value

        # Simulate paths
        paths = self.simulate_paths(current_value, periods, n_simulations, method)

        # For each path, find first time it crosses the mean
        first_passage_times = np.full(n_simulations, periods + 1)

        if current_value > self.mean:
            # Looking for crossings from above
            for i in range(n_simulations):
                crossings = np.where(paths[i, :] <= self.mean)[0]
                if len(crossings) > 0:
                    first_passage_times[i] = crossings[0]
        else:
            # Looking for crossings from below
            for i in range(n_simulations):
                crossings = np.where(paths[i, :] >= self.mean)[0]
                if len(crossings) > 0:
                    first_passage_times[i] = crossings[0]

        # Calculate statistics
        median_time = np.median(first_passage_times[first_passage_times <= periods])
        mean_time = np.mean(first_passage_times[first_passage_times <= periods])

        # Percentage that hit the mean within the period
        hit_percentage = 100 * np.sum(first_passage_times <= periods) / n_simulations

        # Calculate percentiles for the distribution
        percentiles = {}
        for p in [10, 25, 50, 75, 90]:
            percentiles[f'p{p}'] = np.percentile(first_passage_times[first_passage_times <= periods], p)

        return {
            'median_time': median_time,
            'mean_time': mean_time,
            'hit_percentage': hit_percentage,
            'percentiles': percentiles,
            'raw_times': first_passage_times
        }

    def plot_distribution(self, passage_times, max_periods=None):
        """
        Plot histogram of first passage times

        Parameters:
        -----------
        passage_times : array-like or dict
            First passage times or output from first_passage_distribution
        max_periods : int, optional
            Maximum periods to include in plot
        """
        if isinstance(passage_times, dict):
            times = passage_times['raw_times']
        else:
            times = passage_times

        if max_periods is not None:
            times = times[times <= max_periods]

        plt.figure(figsize=(10, 6))
        plt.hist(times, bins=30, alpha=0.7)
        plt.axvline(np.median(times), color='red', linestyle='--', label=f'Median: {np.median(times):.2f}')
        plt.axvline(np.mean(times), color='green', linestyle='--', label=f'Mean: {np.mean(times):.2f}')
        plt.title('Distribution of First Passage Times to Mean')
        plt.xlabel('Time Periods')
        plt.ylabel('Frequency')
        plt.legend()
        plt.grid(True, alpha=0.3)
        return plt.gcf()

    def plot_simulated_paths(self, paths=None, n_paths=10, periods=60):
        """
        Plot sample of simulated paths

        Parameters:
        -----------
        paths : array-like, optional
            Previously simulated paths
        n_paths : int, optional
            Number of paths to display
        periods : int, optional
            Number of periods to simulate if paths not provided
        """
        if paths is None:
            paths = self.simulate_paths(periods=periods)

        plt.figure(figsize=(12, 6))

        # Plot a sample of paths
        for i in range(min(n_paths, paths.shape[0])):
            plt.plot(paths[i], alpha=0.5, linewidth=1)

        # Add mean line
        plt.axhline(y=self.mean, color='r', linestyle='-', label=f'Mean: {self.mean:.2f}')

        # Add last known value
        plt.scatter(0, paths[0, 0], color='black', s=50, label=f'Current: {paths[0, 0]:.2f}')

        plt.title('Simulated Mean Reversion Paths')
        plt.xlabel('Time Periods')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        return plt.gcf()

    def summary(self, current_value=None):
        """
        Return comprehensive mean reversion analysis summary

        Parameters:
        -----------
        current_value : float, optional
            Starting value (default is last value in series)

        Returns:
        --------
        dict: Summary statistics and estimates
        """
        if current_value is None:
            current_value = self.current_value

        # Ensure parameters are calculated
        if self.phi is None:
            self.estimate_ar_parameters()

        if self.ou_theta is None:
            self.estimate_ou_parameters()

        # Calculate time to mean estimates
        time_to_mean_ar = self.estimate_time_to_mean(current_value, method='ar')
        time_to_mean_ou = self.estimate_time_to_mean(current_value, method='ou')

        # Calculate first passage distributions
        passage_dist_ar = self.first_passage_distribution(current_value, method='ar')
        passage_dist_ou = self.first_passage_distribution(current_value, method='ou')

        # Compile results
        return {
            'current_value': current_value,
            'mean': self.mean,
            'distance_to_mean': current_value - self.mean,
            'percent_deviation': 100 * (current_value - self.mean) / self.mean,
            'ar_model': {
                'phi': self.phi,
                'half_life': self.half_life,
                'time_to_mean_estimate': time_to_mean_ar,
                'passage_distribution': {
                    'median': passage_dist_ar['median_time'],
                    'mean': passage_dist_ar['mean_time'],
                    'hit_percentage': passage_dist_ar['hit_percentage'],
                    'percentiles': passage_dist_ar['percentiles']
                }
            },
            'ou_model': {
                'theta': self.ou_theta,
                'mu': self.ou_mu,
                'sigma': self.ou_sigma,
                'half_life': np.log(2) / self.ou_theta,
                'time_to_mean_estimate': time_to_mean_ou,
                'passage_distribution': {
                    'median': passage_dist_ou['median_time'],
                    'mean': passage_dist_ou['mean_time'],
                    'hit_percentage': passage_dist_ou['hit_percentage'],
                    'percentiles': passage_dist_ou['percentiles']
                }
            }
        }

