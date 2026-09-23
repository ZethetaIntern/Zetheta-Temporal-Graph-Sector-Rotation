"""
Data Ingestion Pipeline for NSE Sectoral Indices (Bank, IT, Pharma).
"""
import pandas as pd
import numpy as np

def load_nse_sector_data(filepath='nse_sectors_data.csv'):
    try:
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        return df
    except FileNotFoundError:
        dates = pd.date_range(start='2018-01-01', end='2026-09-01', freq='B')
        np.random.seed(42)
        random_walks = np.cumsum(np.random.normal(0.0003, 0.015, size=(len(dates), 3)), axis=0) + 100
        return pd.DataFrame(random_walks, index=dates, columns=['NIFTY BANK', 'NIFTY IT', 'NIFTY PHARMA'])
