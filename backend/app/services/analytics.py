from typing import Dict, Any, List
import pandas as pd
import numpy as np

def compute_dashboard_kpis(clean_df: pd.DataFrame, rfm_df: pd.DataFrame, result_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute high-level executive retail KPIs from real transaction and segmentation data."""
    total_cust = int(rfm_df["CustomerID"].nunique())
    total_rev = float(clean_df["TotalAmount"].sum())
    avg_spend = float(rfm_df["Monetary"].mean()) if total_cust > 0 else 0.0
    median_spend = float(rfm_df["Monetary"].median()) if total_cust > 0 else 0.0
    avg_freq = float(rfm_df["Frequency"].mean()) if total_cust > 0 else 0.0
    total_orders = int(clean_df["InvoiceNo"].nunique())
    aov = float(total_rev / total_orders) if total_orders > 0 else 0.0
    
    # Repeat rate: customers with > 1 invoice
    repeat_customers = int((rfm_df["Frequency"] > 1).sum())
    retention_rate = round(float(repeat_customers / total_cust * 100), 1) if total_cust > 0 else 0.0

    # At-risk metrics
    at_risk_df = result_df[result_df["Persona"].str.contains("At-Risk", case=False, na=False)]
    at_risk_count = len(at_risk_df)
    at_risk_revenue = float(at_risk_df["Monetary"].sum())

    n_segments = int(result_df["Cluster"].nunique())

    return {
        "total_customers": total_cust,
        "total_revenue": round(total_rev, 2),
        "avg_customer_spend": round(avg_spend, 2),
        "avg_frequency": round(avg_freq, 2),
        "median_spend": round(median_spend, 2),
        "total_orders": total_orders,
        "avg_order_value": round(aov, 2),
        "customer_segments_count": n_segments,
        "retention_rate": retention_rate,
        "at_risk_count": at_risk_count,
        "at_risk_revenue": round(at_risk_revenue, 2),
    }

def compute_monthly_trends(clean_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Aggregate revenue, invoices, and active buyers by month."""
    df = clean_df.copy()
    df["YearMonth"] = df["InvoiceDate"].dt.to_period("M").astype(str)

    monthly = df.groupby("YearMonth").agg(
        revenue=("TotalAmount", "sum"),
        orders=("InvoiceNo", "nunique"),
        active_customers=("CustomerID", "nunique")
    ).reset_index()

    monthly["revenue"] = monthly["revenue"].round(2)
    return monthly.rename(columns={"YearMonth": "month"}).to_dict(orient="records")

def compute_country_breakdown(clean_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Compute revenue and customer geographic distribution."""
    if "Country" not in clean_df.columns:
        return [{"country": "United Kingdom", "revenue": float(clean_df["TotalAmount"].sum()), "orders": clean_df["InvoiceNo"].nunique(), "customers": clean_df["CustomerID"].nunique()}]

    country_grp = clean_df.groupby("Country").agg(
        revenue=("TotalAmount", "sum"),
        orders=("InvoiceNo", "nunique"),
        customers=("CustomerID", "nunique")
    ).reset_index().sort_values("revenue", ascending=False)

    country_grp["revenue"] = country_grp["revenue"].round(2)
    return country_grp.rename(columns={"Country": "country"}).head(10).to_dict(orient="records")

def compute_top_products(clean_df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Identify top performing stock items by revenue and quantity."""
    if "Description" not in clean_df.columns:
        return []

    prod_grp = clean_df.groupby(["StockCode", "Description"]).agg(
        total_quantity=("Quantity", "sum"),
        total_revenue=("TotalAmount", "sum"),
        unique_buyers=("CustomerID", "nunique")
    ).reset_index().sort_values("total_revenue", ascending=False)

    prod_grp["total_revenue"] = prod_grp["total_revenue"].round(2)
    return prod_grp.head(10).to_dict(orient="records")
