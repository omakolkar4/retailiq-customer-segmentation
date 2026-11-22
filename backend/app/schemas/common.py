from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

class DataSummaryResponse(BaseModel):
    filename: str
    total_raw_rows: int
    cleaned_rows: int
    total_customers: int
    date_min: Optional[str] = None
    date_max: Optional[str] = None
    reference_date: Optional[str] = None
    total_revenue: float
    total_invoices: int
    avg_order_value: float
    countries_count: int
    missing_customer_ids: int
    cancelled_orders: int
    negative_or_zero_qty_price: int

class DatasetPreviewResponse(BaseModel):
    summary: DataSummaryResponse
    columns: List[str]
    sample_records: List[Dict[str, Any]]

class DashboardKPIs(BaseModel):
    total_customers: int
    total_revenue: float
    avg_customer_spend: float
    avg_frequency: float
    median_spend: float
    total_orders: int
    avg_order_value: float
    customer_segments_count: int
    retention_rate: float
    at_risk_count: int
    at_risk_revenue: float

class SegmentProfile(BaseModel):
    cluster: int
    persona: str
    total_customers: int
    pct_customers: float
    total_revenue: float
    pct_revenue: float
    avg_recency: float
    avg_frequency: float
    avg_monetary: float
    median_monetary: float
    clv_estimate: float
    marketing_action: str
    risk_level: str

class CustomerRecord(BaseModel):
    customer_id: int
    recency: int
    frequency: int
    monetary: float
    cluster: int
    persona: str
    marketing_action: str
    pca1: float
    pca2: float
    clv_estimate: float
    risk_level: str

class ModelMetric(BaseModel):
    model_name: str
    silhouette_score: float
    davies_bouldin_index: float
    calinski_harabasz_score: float
    inertia: Optional[float] = None
    execution_time_ms: float
    n_clusters: int
    description: str

class ElbowSilhouettePoint(BaseModel):
    k: int
    inertia: float
    silhouette: float

class ModelComparisonResponse(BaseModel):
    models: List[ModelMetric]
    elbow_curve: List[ElbowSilhouettePoint]
    recommended_model: str
    recommendation_reason: str

class StabilityIteration(BaseModel):
    iteration: int
    sample_size: int
    ari_score: float
    silhouette_score: float

class StabilityKComparison(BaseModel):
    k: int
    mean_ari: float
    std_ari: float
    mean_silhouette: float
    std_silhouette: float

class ValidationResponse(BaseModel):
    selected_k: int
    n_iterations: int
    subsample_ratio: float
    mean_ari: float
    std_ari: float
    mean_silhouette: float
    std_silhouette: float
    iterations: List[StabilityIteration]
    k_comparison: List[StabilityKComparison]
    methodology_notes: str
    limitations: str

class MarketingStrategy(BaseModel):
    persona: str
    customer_count: int
    revenue_contribution: float
    revenue_pct: float
    avg_rfm: Dict[str, float]
    behavior_summary: str
    marketing_goal: str
    recommended_campaign: str
    channel_mix: List[str]
    expected_roi_impact: str

class MonthlyTrend(BaseModel):
    month: str
    revenue: float
    orders: int
    active_customers: int

class CountryRevenue(BaseModel):
    country: str
    revenue: float
    orders: int
    customers: int
