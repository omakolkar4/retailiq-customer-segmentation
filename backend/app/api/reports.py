from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import Response, StreamingResponse
from typing import Any
import io
import pandas as pd

from app.services.data_loader import data_store
from app.services.clustering import fit_clustering_model
from app.services.analytics import compute_dashboard_kpis
from app.services.model_evaluation import evaluate_models
from app.services.validation import run_clustering_validation
from app.services.report_generator import generate_pdf_report

router = APIRouter(prefix="/reports", tags=["Reports & Exports"])

@router.get("/pdf")
def export_pdf_report(k: Any = 4):
    """Generate and download a comprehensive multi-page executive PDF report."""
    if hasattr(k, "default"):
        k = k.default
    k = int(k)
    if data_store.rfm_df is None or data_store.clean_df is None:
        raise HTTPException(status_code=404, detail="No active dataset.")

    customers_df, profiles_df, _ = fit_clustering_model(data_store.rfm_df, algorithm="kmeans", k=k)
    kpis = compute_dashboard_kpis(data_store.clean_df, data_store.rfm_df, customers_df)
    models_data = evaluate_models(data_store.rfm_df, k=k)
    validation_data = run_clustering_validation(data_store.rfm_df, k=k, n_iterations=5)

    pdf_bytes = generate_pdf_report(
        kpis=kpis,
        profiles_df=profiles_df,
        customers_df=customers_df,
        models_data=models_data,
        validation_data=validation_data,
        filename=data_store.filename,
    )

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": 'attachment; filename="RetailIQ_Executive_Report.pdf"'},
    )

@router.get("/csv/customers")
def export_customers_csv(k: Any = 4):
    """Export customer-level RFM, cluster assignments, personas, and CLV to CSV."""
    if hasattr(k, "default"):
        k = k.default
    k = int(k)
    if data_store.rfm_df is None:
        raise HTTPException(status_code=404, detail="No active dataset.")

    customers_df, _, _ = fit_clustering_model(data_store.rfm_df, algorithm="kmeans", k=k)
    csv_str = customers_df.to_csv(index=False)

    return Response(
        content=csv_str,
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="retailiq_customer_segments.csv"'},
    )

@router.get("/csv/segments")
def export_segments_csv(k: Any = 4):
    """Export segment-level profiles and summary statistics to CSV."""
    if hasattr(k, "default"):
        k = k.default
    k = int(k)
    if data_store.rfm_df is None:
        raise HTTPException(status_code=404, detail="No active dataset.")

    _, profiles_df, _ = fit_clustering_model(data_store.rfm_df, algorithm="kmeans", k=k)
    csv_str = profiles_df.reset_index().to_csv(index=False)

    return Response(
        content=csv_str,
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="retailiq_segment_profiles.csv"'},
    )

@router.get("/csv/models")
def export_models_csv(k: Any = 4):
    """Export model comparison benchmark metrics to CSV."""
    if hasattr(k, "default"):
        k = k.default
    k = int(k)
    if data_store.rfm_df is None:
        raise HTTPException(status_code=404, detail="No active dataset.")

    eval_data = evaluate_models(data_store.rfm_df, k=k)
    df_models = pd.DataFrame(eval_data["models"])
    csv_str = df_models.to_csv(index=False)

    return Response(
        content=csv_str,
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="retailiq_model_benchmarks.csv"'},
    )
