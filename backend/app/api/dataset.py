from fastapi import APIRouter, UploadFile, File, HTTPException
import pandas as pd
import io

from app.services.data_loader import data_store

router = APIRouter(prefix="/dataset", tags=["Dataset Management"])

@router.get("/summary")
def get_dataset_summary():
    """Return summary diagnostics and quality metrics of the active dataset."""
    if data_store.summary is None:
        raise HTTPException(status_code=404, detail="No active dataset loaded.")
    return data_store.summary

@router.get("/preview")
def get_dataset_preview(limit: int = 20):
    """Return schema, summary, and sample rows from the active transaction dataset."""
    if data_store.clean_df is None:
        raise HTTPException(status_code=404, detail="No active dataset loaded.")
    
    sample = data_store.clean_df.head(limit).copy()
    if "InvoiceDate" in sample.columns:
        sample["InvoiceDate"] = sample["InvoiceDate"].dt.strftime("%Y-%m-%d %H:%M:%S")

    return {
        "summary": data_store.summary,
        "columns": data_store.clean_df.columns.tolist(),
        "sample_records": sample.to_dict(orient="records"),
    }

@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload new transaction dataset in CSV or Excel format."""
    filename = file.filename or "uploaded_data.csv"
    ext = filename.lower().split(".")[-1]

    if ext not in ["csv", "xlsx", "xls"]:
        raise HTTPException(status_code=400, detail="Unsupported file format. Please upload a .csv, .xlsx, or .xls file.")

    contents = await file.read()
    try:
        if ext == "csv":
            df = pd.read_csv(io.BytesIO(contents))
        else:
            df = pd.read_excel(io.BytesIO(contents))
        
        data_store.set_dataset(df, filename=filename)
        return {
            "message": f"Successfully processed '{filename}'.",
            "summary": data_store.summary,
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to process dataset: {str(exc)}")

@router.post("/reset-default")
def reset_to_default():
    """Reset platform to original Online Retail dataset."""
    try:
        data_store.load_default()
        return {
            "message": "Reset to default Online Retail dataset successfully.",
            "summary": data_store.summary,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not load default dataset: {str(exc)}")
