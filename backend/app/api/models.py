from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.services.data_loader import data_store
from app.services.model_evaluation import evaluate_models, clear_model_eval_cache
from app.services.stability_analysis import run_multi_model_stability, clear_stability_cache
from app.services.clustering import fit_clustering_model, clear_clustering_cache

router = APIRouter(prefix="/models", tags=["Model Laboratory"])

class BenchmarkRequest(BaseModel):
    k: Optional[int] = 4
    force_refresh: Optional[bool] = False

class ValidationRequest(BaseModel):
    k: Optional[int] = 4
    n_trials: Optional[int] = 5
    subsample_ratio: Optional[float] = 0.8

@router.get("/comparison")
def get_model_comparison(k: int = Query(4, ge=2, le=10)):
    """
    Academically rigorous comparative benchmark of all SIX clustering algorithms:
    K-Means, Agglomerative Hierarchical, GMM, BIRCH, DBSCAN, and OPTICS.
    """
    if data_store.rfm_df is None:
        raise HTTPException(status_code=404, detail="No active dataset.")
    
    return evaluate_models(data_store.rfm_df, k=k)

@router.post("/benchmark")
def run_model_benchmark(payload: BenchmarkRequest):
    """
    Trigger full 6-model benchmark with optional cache busting.
    """
    if data_store.rfm_df is None:
        raise HTTPException(status_code=404, detail="No active dataset.")
    
    if payload.force_refresh:
        clear_model_eval_cache()
        clear_stability_cache()
        clear_clustering_cache()

    return evaluate_models(data_store.rfm_df, k=payload.k or 4)

@router.get("/recommendation")
def get_model_recommendation(k: int = Query(4, ge=2, le=10)):
    """
    Return dynamic algorithmic selection recommendation with multi-criteria rankings.
    """
    if data_store.rfm_df is None:
        raise HTTPException(status_code=404, detail="No active dataset.")
    
    res = evaluate_models(data_store.rfm_df, k=k)
    return {
        "recommended_model": res["recommended_model"],
        "recommendation_reason": res["recommendation_reason"],
        "composite_rank_summary": res["composite_rank_summary"],
    }

@router.post("/validation")
def get_multi_model_validation(payload: ValidationRequest):
    """
    Execute bootstrap stability testing across all 6 models with Adjusted Rand Index (ARI).
    """
    if data_store.rfm_df is None:
        raise HTTPException(status_code=404, detail="No active dataset.")
        
    return run_multi_model_stability(
        data_store.rfm_df,
        k=payload.k or 4,
        n_trials=payload.n_trials or 5,
        subsample_ratio=payload.subsample_ratio or 0.8
    )

@router.get("/profiles/{algorithm}")
def get_algorithm_cluster_profiles(algorithm: str, k: int = Query(4, ge=2, le=10)):
    """
    Retrieve cluster profiles, persona mapping, and RFM centroid metrics for a specific algorithm.
    Supported: kmeans, hierarchical, gmm, birch, dbscan, optics.
    """
    if data_store.rfm_df is None:
        raise HTTPException(status_code=404, detail="No active dataset.")
        
    valid_algos = ["kmeans", "hierarchical", "agglomerative", "gmm", "gaussian_mixture", "birch", "dbscan", "optics"]
    if algorithm.lower() not in valid_algos:
        raise HTTPException(status_code=400, detail=f"Unsupported algorithm '{algorithm}'. Must be one of {valid_algos}")
        
    result_df, profiles, meta = fit_clustering_model(data_store.rfm_df, algorithm=algorithm, k=k)
    return {
        "algorithm": meta.get("algorithm", algorithm),
        "family": meta.get("family"),
        "parameters": meta.get("parameters"),
        "profiles": profiles.reset_index().to_dict(orient="records"),
        "pca_sample": result_df[["CustomerID", "Cluster", "Persona", "PCA1", "PCA2"]].head(1000).to_dict(orient="records"),
    }
