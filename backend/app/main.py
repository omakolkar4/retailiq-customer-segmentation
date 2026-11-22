from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import threading
from app.services.model_evaluation import evaluate_models

from app.core.config import PROJECT_NAME, VERSION, API_V1_PREFIX, CORS_ORIGINS
from app.services.data_loader import data_store
from app.api import (
    dataset,
    dashboard,
    segmentation,
    models,
    validation,
    customers,
    marketing,
    reports,
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Ensure default dataset is loaded
    if data_store.clean_df is None:
        data_store.load_default()
    # Pre-warm model evaluation cache in background thread
    if data_store.rfm_df is not None:
        threading.Thread(target=evaluate_models, args=(data_store.rfm_df, 4), daemon=True).start()
    yield

app = FastAPI(
    title=PROJECT_NAME,
    version=VERSION,
    description="Enterprise-grade Customer Intelligence & Retail Analytics API powered by RFM Analysis and Machine Learning Clustering.",
    lifespan=lifespan,
)

# CORS configuration for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS + ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount API Routers
app.include_router(dataset.router, prefix=API_V1_PREFIX)
app.include_router(dashboard.router, prefix=API_V1_PREFIX)
app.include_router(segmentation.router, prefix=API_V1_PREFIX)
app.include_router(models.router, prefix=API_V1_PREFIX)
app.include_router(validation.router, prefix=API_V1_PREFIX)
app.include_router(customers.router, prefix=API_V1_PREFIX)
app.include_router(marketing.router, prefix=API_V1_PREFIX)
app.include_router(reports.router, prefix=API_V1_PREFIX)

@app.get("/")
def root():
    return {
        "status": "online",
        "platform": "RetailIQ — Customer Intelligence Platform",
        "version": VERSION,
        "docs_url": "/docs",
        "dataset_loaded": data_store.clean_df is not None,
        "active_dataset": data_store.filename,
        "total_customers": len(data_store.rfm_df) if data_store.rfm_df is not None else 0,
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
