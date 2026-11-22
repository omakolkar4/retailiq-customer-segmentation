from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from app.services.data_loader import data_store
from app.services.clustering import fit_clustering_model

router = APIRouter(prefix="/segmentation", tags=["Segmentation"])

@router.get("/profiles")
def get_segment_profiles(k: int = Query(4, ge=2, le=10), algorithm: str = "kmeans"):
    if data_store.rfm_df is None:
        raise HTTPException(status_code=404, detail="No active dataset.")
    
    customers_df, profiles_df, meta = fit_clustering_model(data_store.rfm_df, algorithm=algorithm, k=k)
    return {
        "algorithm": meta.get("algorithm", algorithm),
        "k": meta.get("k", k),
        "profiles": profiles_df.reset_index().to_dict(orient="records"),
        "total_customers": len(customers_df),
        "total_revenue": float(customers_df["Monetary"].sum()),
        "explained_variance": meta.get("explained_variance_ratio", [0.5, 0.3]),
    }

@router.get("/pca")
def get_pca_scatter(
    k: int = Query(4, ge=2, le=10),
    algorithm: str = "kmeans",
    sample_limit: int = Query(1500, ge=100, le=5000)
):
    """Return 2D PCA projected points sampled for responsive scatter visualization."""
    if data_store.rfm_df is None:
        raise HTTPException(status_code=404, detail="No active dataset.")
    
    customers_df, _, meta = fit_clustering_model(data_store.rfm_df, algorithm=algorithm, k=k)
    
    if len(customers_df) > sample_limit:
        sample_df = customers_df.sample(sample_limit, random_state=42)
    else:
        sample_df = customers_df

    cols = ["CustomerID", "Recency", "Frequency", "Monetary", "Cluster", "Persona", "PCA1", "PCA2"]
    points = sample_df[cols].to_dict(orient="records")

    return {
        "points": points,
        "sample_size": len(points),
        "total_size": len(customers_df),
        "explained_variance": meta.get("explained_variance_ratio", [0.5, 0.3]),
    }
