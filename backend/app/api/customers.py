from fastapi import APIRouter, HTTPException, Query
from typing import Optional, Any

from app.services.data_loader import data_store
from app.services.clustering import fit_clustering_model

router = APIRouter(prefix="/customers", tags=["Customer Explorer"])

@router.get("/search")
def search_customers(
    query: Optional[str] = None,
    persona: Optional[str] = None,
    sort_by: Any = "Monetary",
    sort_desc: bool = True,
    page: Any = 1,
    page_size: Any = 25,
    k: Any = 4,
):
    if hasattr(sort_by, "default"):
        sort_by = sort_by.default
    sort_by = str(sort_by)
    if hasattr(page, "default"):
        page = page.default
    page = int(page)
    if hasattr(page_size, "default"):
        page_size = page_size.default
    page_size = int(page_size)
    if hasattr(k, "default"):
        k = k.default
    k = int(k)

    if data_store.rfm_df is None:
        raise HTTPException(status_code=404, detail="No active dataset.")

    customers_df, _, _ = fit_clustering_model(data_store.rfm_df, algorithm="kmeans", k=k)
    filtered = customers_df.copy()

    if query and query.strip():
        q = query.strip()
        filtered = filtered[filtered["CustomerID"].astype(str).str.contains(q)]

    if persona and persona.strip() and persona.lower() != "all":
        filtered = filtered[filtered["Persona"] == persona.strip()]

    total_count = len(filtered)
    filtered = filtered.sort_values(sort_by, ascending=not sort_desc)

    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    page_data = filtered.iloc[start_idx:end_idx]

    cols = ["CustomerID", "Recency", "Frequency", "Monetary", "Cluster", "Persona", "Marketing_Action", "Risk_Level", "CLV_Estimate"]
    
    # Available personas for dropdown
    all_personas = sorted(customers_df["Persona"].unique().tolist())

    return {
        "total_records": total_count,
        "page": page,
        "page_size": page_size,
        "total_pages": max(1, (total_count + page_size - 1) // page_size),
        "personas": all_personas,
        "records": page_data[cols].to_dict(orient="records"),
    }

@router.get("/{customer_id}")
def get_customer_detail(customer_id: int, k: Any = 4):
    if hasattr(k, "default"):
        k = k.default
    k = int(k)
    if data_store.rfm_df is None:
        raise HTTPException(status_code=404, detail="No active dataset.")

    customers_df, _, _ = fit_clustering_model(data_store.rfm_df, algorithm="kmeans", k=k)
    matched = customers_df[customers_df["CustomerID"] == customer_id]
    if matched.empty:
        raise HTTPException(status_code=404, detail=f"Customer ID {customer_id} not found.")

    row = matched.iloc[0].to_dict()
    
    # Also fetch recent purchase history from clean_df if available
    transactions = []
    if data_store.clean_df is not None:
        tx_df = data_store.clean_df[data_store.clean_df["CustomerID"] == customer_id].copy()
        tx_df = tx_df.sort_values("InvoiceDate", ascending=False).head(15)
        if "InvoiceDate" in tx_df.columns:
            tx_df["InvoiceDate"] = tx_df["InvoiceDate"].dt.strftime("%Y-%m-%d %H:%M:%S")
        transactions = tx_df[["InvoiceNo", "InvoiceDate", "Quantity", "UnitPrice", "TotalAmount"]].to_dict(orient="records")

    return {
        "customer": row,
        "transactions": transactions,
    }
