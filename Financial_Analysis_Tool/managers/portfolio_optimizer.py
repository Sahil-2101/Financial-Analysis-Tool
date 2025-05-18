"""
Portfolio optimization module for calculating optimal portfolio weights.
"""

import numpy as np
import scipy.optimize as sco

class PortfolioOptimizer:
    @staticmethod
    def optimize_portfolio(returns, cov_matrix):
        """
        Optimize portfolio weights to minimize risk for a given target return.

        Args:
            returns (numpy.ndarray): Expected returns of the assets.
            cov_matrix (numpy.ndarray): Covariance matrix of the asset returns.

        Returns:
            dict: A dictionary containing the optimal weights and the optimization result.
        """
        # Objective function: Portfolio variance (risk)
        def portfolio_variance(weights, cov_matrix):
            return weights.T @ cov_matrix @ weights

        # Constraints: Weights must sum to 1
        constraints = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1}  # Sum of weights equals 1
        ]

        # Bounds: Weights must be between 0 and 1 (no short-selling)
        bounds = [(0, 1) for _ in range(len(returns))]

        # Initial guess: Equal allocation
        initial_weights = np.ones(len(returns)) / len(returns)

        # Optimization using SLSQP method
        result = sco.minimize(
            portfolio_variance,
            x0=initial_weights,
            args=(cov_matrix,),
            method="SLSQP",
            constraints=constraints,
            bounds=bounds
        )

        if result.success:
            return {
                "Optimal Weights": result.x,
                "Optimization Result": result
            }
        else:
            raise ValueError(f"Optimization failed: {result.message}") 