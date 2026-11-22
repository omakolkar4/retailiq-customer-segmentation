import io
import os
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np

from app.core.config import DEFAULT_DATA_FILE, FALLBACK_EXCEL

ALIASES = {
    "CustomerID": ["CustomerID", "Customer ID", "customer_id", "customerid"],
    "InvoiceNo": ["InvoiceNo", "Invoice No", "Invoice", "invoice_no", "invoiceno"],
    "InvoiceDate": ["InvoiceDate", "Invoice Date", "date", "invoice_date"],
    "Quantity": ["Quantity", "quantity", "Qty"],
    "UnitPrice": ["UnitPrice", "Unit Price", "Price", "unit_price", "price"],
}

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map common Online Retail column names to a consistent canonical schema."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    rename = {}
    for canonical, candidates in ALIASES.items():
        for candidate in candidates:
            if candidate in df.columns:
                rename[candidate] = canonical
                break
    df = df.rename(columns=rename)

    required = ["CustomerID", "InvoiceNo", "InvoiceDate", "Quantity", "UnitPrice"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns: {', '.join(missing)}. "
            "Expected transaction data with CustomerID, InvoiceNo, InvoiceDate, Quantity, and UnitPrice."
        )

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    for col in ["Quantity", "UnitPrice", "CustomerID"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["InvoiceNo"] = df["InvoiceNo"].astype(str).str.strip()
    return df

def clean_and_profile_data(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Clean raw transactions and compute comprehensive data quality diagnostics."""
    norm_df = normalize_columns(raw_df)
    total_raw = len(norm_df)

    missing_cust = int(norm_df["CustomerID"].isna().sum())
    missing_other = int(norm_df[["InvoiceNo", "InvoiceDate", "Quantity", "UnitPrice"]].isna().any(axis=1).sum())
    
    # Non-null subset
    valid_nulls = norm_df.dropna(subset=["CustomerID", "InvoiceNo", "InvoiceDate", "Quantity", "UnitPrice"]).copy()
    
    # Cancellations (InvoiceNo starts with 'C')
    is_cancelled = valid_nulls["InvoiceNo"].str.upper().str.startswith("C")
    cancelled_count = int(is_cancelled.sum())
    non_cancelled = valid_nulls[~is_cancelled].copy()

    # Negative or zero quantities/prices
    invalid_qty_price = (non_cancelled["Quantity"] <= 0) | (non_cancelled["UnitPrice"] <= 0)
    invalid_qty_price_count = int(invalid_qty_price.sum())
    
    clean_df = non_cancelled[~invalid_qty_price].copy()
    clean_df["TotalAmount"] = clean_df["Quantity"] * clean_df["UnitPrice"]

    if clean_df.empty:
        raise ValueError("No valid transaction rows remain after cleaning.")

    ref_date = clean_df["InvoiceDate"].max() + pd.Timedelta(days=1)
    min_date = clean_df["InvoiceDate"].min()
    max_date = clean_df["InvoiceDate"].max()

    countries_count = int(clean_df["Country"].nunique()) if "Country" in clean_df.columns else 1
    total_revenue = float(clean_df["TotalAmount"].sum())
    total_invoices = int(clean_df["InvoiceNo"].nunique())
    total_customers = int(clean_df["CustomerID"].nunique())
    avg_order_value = float(total_revenue / total_invoices) if total_invoices > 0 else 0.0

    summary = {
        "total_raw_rows": total_raw,
        "cleaned_rows": len(clean_df),
        "total_customers": total_customers,
        "date_min": min_date.strftime("%Y-%m-%d") if pd.notna(min_date) else None,
        "date_max": max_date.strftime("%Y-%m-%d") if pd.notna(max_date) else None,
        "reference_date": ref_date.strftime("%Y-%m-%d"),
        "total_revenue": round(total_revenue, 2),
        "total_invoices": total_invoices,
        "avg_order_value": round(avg_order_value, 2),
        "countries_count": countries_count,
        "missing_customer_ids": missing_cust,
        "cancelled_orders": cancelled_count,
        "negative_or_zero_qty_price": invalid_qty_price_count,
    }

    return clean_df, summary

class DataStore:
    """Thread-safe in-memory singleton holding current active dataset state."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DataStore, cls).__new__(cls)
            cls._instance.raw_df = None
            cls._instance.clean_df = None
            cls._instance.rfm_df = None
            cls._instance.summary = None
            cls._instance.filename = "online_retail.csv"
            cls._instance.reference_date = None
            cls._instance.cached_models = {}
            cls._instance.load_default()
        return cls._instance

    def load_default(self):
        """Loads default dataset from csv or excel."""
        if DEFAULT_DATA_FILE.exists():
            print(f"Loading default dataset from {DEFAULT_DATA_FILE}...")
            df = pd.read_csv(DEFAULT_DATA_FILE)
            self.set_dataset(df, filename="online_retail.csv")
        elif FALLBACK_EXCEL.exists():
            print(f"Loading fallback dataset from {FALLBACK_EXCEL}...")
            df = pd.read_excel(FALLBACK_EXCEL)
            self.set_dataset(df, filename="Online Retail.xlsx")
        else:
            print("No default dataset found on disk yet.")

    def set_dataset(self, df: pd.DataFrame, filename: str = "uploaded_file.csv"):
        """Process and set new active dataset."""
        from app.services.rfm import compute_rfm
        from app.services.clustering import clear_clustering_cache
        from app.services.model_evaluation import clear_model_eval_cache
        from app.services.validation import clear_validation_cache

        clear_clustering_cache()
        clear_model_eval_cache()
        clear_validation_cache()

        self.raw_df = df
        self.filename = filename
        self.clean_df, self.summary = clean_and_profile_data(df)
        self.summary["filename"] = filename
        self.reference_date = pd.to_datetime(self.summary["reference_date"])
        self.rfm_df = compute_rfm(self.clean_df, self.reference_date)
        print(f"Dataset '{filename}' loaded: {len(self.clean_df):,} clean transactions, {len(self.rfm_df):,} customers.")

data_store = DataStore()
