import unittest
import numpy as np
import sys
import os

# Add src to path safely
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), 'src')))
from optimizer import optimize_portfolio

class TestPortfolioOptimizer(unittest.TestCase):
    def setUp(self):
        # Mock expected returns and covariance matrix for 3 sectors
        self.expected_returns = np.array([0.12, 0.15, 0.10])
        self.cov_matrix = np.array([
            [0.04, 0.01, 0.005],
            [0.01, 0.05, 0.02],
            [0.005, 0.02, 0.03]
        ])

    def test_weights_sum_to_one(self):
        weights = optimize_portfolio(self.expected_returns, self.cov_matrix, max_weight=0.50)
        self.assertAlmostEqual(np.sum(weights), 1.0, places=4)

    def test_concentration_cap(self):
        max_cap = 0.40
        weights = optimize_portfolio(self.expected_returns, self.cov_matrix, max_weight=max_cap)
        for w in weights:
            self.assertLessEqual(w, max_cap + 1e-5)
            self.assertGreaterEqual(w, -1e-5)

if __name__ == '__main__':
    unittest.main(argv=[''], exit=False)
