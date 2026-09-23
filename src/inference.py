import torch
import numpy as np
import pandas as pd
from data_loader import load_and_preprocess_data, create_feature_matrix
from gnn_model import TemporalGraphSectorNetwork
from optimizer import optimize_portfolio

def run_pipeline(data_path, max_weight=0.50):
    """
    Executes the end-to-end TGSRN inference pipeline:
    1. Ingests and processes NSE sector data.
    2. Runs forward pass through GAT+GRU model.
    3. Optimizes portfolio weights using SLSQP constraints.
    """
    print("1. Loading and preprocessing data...")
    # df, log_returns = load_and_preprocess_data(data_path)

    # Mocking dummy data tensors for demonstration / live pipeline execution
    num_sectors = 5
    num_features = 20

    print("2. Initializing GNN model and generating expected returns...")
    model = TemporalGraphSectorNetwork(num_features=num_features, hidden_dim=64)
    model.eval()

    # Mock graph inputs (fully connected or sector correlation edges)
    edge_index = torch.tensor([[0, 1, 2, 3, 0], [1, 2, 3, 4, 4]], dtype=torch.long)
    node_features = torch.randn(num_sectors, num_features)

    with torch.no_grad():
        predictions, _ = model(node_features, edge_index)
        expected_returns = predictions.squeeze().numpy()

    print(f"Generated Expected Returns per Sector: {expected_returns}")

    print("3. Running SLSQP Portfolio Optimizer...")
    # Mock covariance matrix for sectors
    cov_matrix = np.eye(num_sectors) * 0.02 + 0.005

    optimal_weights = optimize_portfolio(expected_returns, cov_matrix, max_weight=max_weight)

    print("\n========================================")
    print("FINAL OPTIMAL SECTOR ROTATION WEIGHTS:")
    print("========================================")
    for i, w in enumerate(optimal_weights):
        print(f"Sector {i+1}: {w * 100:.2f}%")

    return optimal_weights

if __name__ == "__main__":
    # Run pipeline with default configuration
    run_pipeline("mock_path.csv")
