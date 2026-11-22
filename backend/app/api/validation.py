from fastapi import APIRouter, HTTPException, Query
from app.services.data_loader import data_store
from app.services.validation import run_clustering_validation

router = APIRouter(prefix="/validation", tags=["Clustering Validation"])

@router.get("/stability")
def get_clustering_stability(
    k: int = Query(4, ge=2, le=8),
    n_iterations: int = Query(10, ge=3, le=20),
    subsample_ratio: float = Query(0.8, ge=0.5, le=0.95),
):
    """Run bootstrap resampling stability evaluation and cross-sample consensus analysis."""
    if data_store.rfm_df is None:
        raise HTTPException(status_code=404, detail="No active dataset.")
    
    return run_clustering_validation(
        data_store.rfm_df,
        k=k,
        n_iterations=n_iterations,
        subsample_ratio=subsample_ratio
    )
