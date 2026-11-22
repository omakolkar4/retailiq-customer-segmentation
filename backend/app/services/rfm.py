from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

def compute_rfm(clean_df: pd.DataFrame, reference_date: pd.Timestamp) -> pd.DataFrame:
    """Aggregate cleaned transaction lines to customer-level Recency, Frequency, Monetary."""
    rfm = clean_df.groupby("CustomerID").agg(
        LastPurchase=("InvoiceDate", "max"),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalAmount", "sum"),
    )
    rfm["Recency"] = (reference_date - rfm["LastPurchase"]).dt.days
    rfm = rfm.reset_index()
    rfm = rfm[["CustomerID", "Recency", "Frequency", "Monetary"]]
    rfm = rfm.replace([np.inf, -np.inf], np.nan).dropna()
    rfm = rfm[(rfm["Recency"] >= 0) & (rfm["Frequency"] > 0) & (rfm["Monetary"] > 0)]
    rfm["CustomerID"] = rfm["CustomerID"].astype(int)

    if len(rfm) < 4:
        raise ValueError("At least 4 customers with valid RFM values are required for customer segmentation.")

    return rfm

def prepare_rfm_features(rfm_df: pd.DataFrame) -> Tuple[np.ndarray, StandardScaler]:
    """Applies log1p transformation to reduce skewness followed by StandardScaler."""
    features = ["Recency", "Frequency", "Monetary"]
    model_data = rfm_df[features].copy()
    model_log = np.log1p(model_data)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(model_log)
    return scaled, scaler
