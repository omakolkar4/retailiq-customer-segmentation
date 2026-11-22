from fastapi import APIRouter, HTTPException
from typing import Optional

from app.services.data_loader import data_store
from app.services.clustering import fit_clustering_model
from app.services.analytics import (
    compute_dashboard_kpis,
    compute_monthly_trends,
    compute_country_breakdown,
    compute_top_products,
)

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])

@router.get("/kpis")
def get_kpis(k: int = 4):
    if data_store.rfm_df is None or data_store.clean_df is None:
        raise HTTPException(status_code=404, detail="No active dataset.")
    
    customers_df, profiles_df, _ = fit_clustering_model(data_store.rfm_df, algorithm="kmeans", k=k)
    kpis = compute_dashboard_kpis(data_store.clean_df, data_store.rfm_df, customers_df)
    return kpis

@router.get("/trends")
def get_trends():
    if data_store.clean_df is None:
        raise HTTPException(status_code=404, detail="No active dataset.")
    return compute_monthly_trends(data_store.clean_df)

@router.get("/countries")
def get_countries():
    if data_store.clean_df is None:
        raise HTTPException(status_code=404, detail="No active dataset.")
    return compute_country_breakdown(data_store.clean_df)

@router.get("/top-products")
def get_products():
    if data_store.clean_df is None:
        raise HTTPException(status_code=404, detail="No active dataset.")
    return compute_top_products(data_store.clean_df)
