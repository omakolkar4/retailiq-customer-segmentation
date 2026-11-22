from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def prepare_clustering_features(rfm_df: pd.DataFrame) -> Tuple[np.ndarray, StandardScaler, list]:
    """
    Standardize customer-level RFM features for unsupervised clustering:
    1. Extract Recency, Frequency, Monetary.
    2. Handle invalid/non-positive values via clipping to positive lower bounds.
    3. Apply log1p transformation to correct heavy right-skewness.
    4. Apply StandardScaler to give equal geometric weight to all three dimensions.
    
    Returns:
        scaled_data: (n_samples, 3) float array
        scaler: fitted StandardScaler instance
        feature_names: ['Recency', 'Frequency', 'Monetary']
    """
    feature_names = ["Recency", "Frequency", "Monetary"]
    data = rfm_df[feature_names].copy()
    
    # Ensure numerical stability: Recency >= 0, Frequency >= 1, Monetary >= 0.01
    data["Recency"] = data["Recency"].clip(lower=0)
    data["Frequency"] = data["Frequency"].clip(lower=1)
    data["Monetary"] = data["Monetary"].clip(lower=0.01)
    
    # Log1p transformation for right-skewed distributions
    log_transformed = np.log1p(data)
    
    # Standardize to zero mean and unit variance
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(log_transformed)
    
    return scaled_data, scaler, feature_names
