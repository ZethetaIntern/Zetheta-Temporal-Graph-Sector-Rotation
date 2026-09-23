import pandas as pd
import numpy as np

def load_and_preprocess_data(file_path):
    """
    Loads NSE sectoral index price data, handles missing values, 
    and computes log returns for the sector rotation network.
    """
    # Read CSV data (expecting Date index and sector columns)
    df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')

    # Handle missing values using forward fill then backward fill
    df = df.fillna(method='ffill').fillna(method='bfill')

    # Calculate daily log returns
    log_returns = np.log(df / df.shift(1)).dropna()

    return df, log_returns

def create_feature_matrix(log_returns, window_size=20):
    """
    Generates rolling feature windows for temporal graph inputs.
    """
    features = []
    for i in range(window_size, len(log_returns)):
        window = log_returns.iloc[i-window_size:i].values
        features.append(window)
    return np.array(features)
