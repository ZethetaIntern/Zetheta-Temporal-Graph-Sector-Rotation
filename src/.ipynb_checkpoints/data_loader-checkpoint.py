import os
import pandas as pd

def load_validated_dataset(filename):
    """Loads a validated dataset from the production_ready_hmm_datasets directory."""
    path = os.path.join('production_ready_hmm_datasets', filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset {filename} not found at {path}")
    df = pd.read_csv(path)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.sort_values('Date').reset_index(drop=True)
    return df
