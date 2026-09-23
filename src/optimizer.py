import numpy as np
from scipy.optimize import minimize

def optimize_portfolio(expected_returns, cov_matrix, max_weight=0.50):
    """
    Performs Mean-Variance Optimization using SLSQP with institutional constraints:
    - Long-only mandate (weights between 0.0 and max_weight)
    - Fully invested constraint (weights sum to 1.0)
    """
    num_assets = len(expected_returns)

    # Objective function: Minimize negative Sharpe ratio (or variance for MVP)
    def portfolio_variance(weights):
        return np.dot(weights.T, np.dot(cov_matrix, weights))

    # Constraints
    constraints = (
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}  # Weights sum to 1
    )

    # Bounds: Long-only with max concentration cap per sector
    bounds = tuple((0.0, max_weight) for _ in range(num_assets))

    # Initial equal weights guess
    init_guess = np.array([1.0 / num_assets] * num_assets)

    result = minimize(
        portfolio_variance, 
        init_guess, 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints
    )

    if not result.success:
        raise ValueError(f"Optimization failed: {result.message}")

    return result.x
