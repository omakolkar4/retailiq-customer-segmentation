from fastapi import APIRouter, HTTPException, Query
from app.services.data_loader import data_store
from app.services.clustering import fit_clustering_model
from app.services.marketing import generate_marketing_intelligence

router = APIRouter(prefix="/marketing", tags=["Marketing Intelligence"])

@router.get("/strategies")
def get_strategies(k: int = Query(4, ge=2, le=10)):
    """Return tailored marketing blueprints, campaign plans, and revenue share per segment."""
    if data_store.rfm_df is None:
        raise HTTPException(status_code=404, detail="No active dataset.")

    _, profiles_df, _ = fit_clustering_model(data_store.rfm_df, algorithm="kmeans", k=k)
    strategies = generate_marketing_intelligence(profiles_df, data_store.rfm_df)

    return {
        "k": k,
        "strategies": strategies,
        "total_revenue": float(profiles_df["Total_Revenue"].sum()),
        "total_customers": int(profiles_df["Total_Customers"].sum()),
    }
