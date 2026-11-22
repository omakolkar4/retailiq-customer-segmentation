from typing import Tuple, Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch, DBSCAN, OPTICS
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

from app.services.preprocessing import prepare_clustering_features

PERSONA_ACTIONS = {
    "VIP High-Value Customers": "Offer exclusive rewards, concierge support, VIP access, and preview new product launches.",
    "Loyal Customers": "Provide loyalty tier perks, referral incentives, and personalized cross-category recommendations.",
    "Regular Customers": "Send personalized replenishment reminders, bundle discounts, and loyalty signup bonuses.",
    "At-Risk Customers": "Initiate automated re-engagement workflows, win-back discount vouchers, and feedback surveys.",
    "Outliers / Noise": "Flag for manual audit; represents unusual transactional anomalies, wholesale purchasers, or atypical micro-patterns.",
}

RISK_LEVELS = {
    "VIP High-Value Customers": "Low Risk",
    "Loyal Customers": "Low Risk",
    "Regular Customers": "Moderate Risk",
    "At-Risk Customers": "High Churn Risk",
    "Outliers / Noise": "Anomaly / Unassigned",
}

_clustering_cache: Dict[Tuple[int, str, int], Any] = {}

def clear_clustering_cache():
    _clustering_cache.clear()

def assign_personas(profiles: pd.DataFrame) -> Dict[int, str]:
    """
    Map arbitrary cluster indices to business-meaningful personas based on
    actual RFM centroid ranks, NOT arbitrary cluster numbers.
    Noise clusters (-1) are mapped to 'Outliers / Noise'.
    """
    persona_map = {}
    
    # Handle noise cluster (-1) explicitly
    clean_profiles = profiles[profiles.index != -1].copy()
    if -1 in profiles.index:
        persona_map[-1] = "Outliers / Noise"
        
    if clean_profiles.empty:
        return persona_map
        
    if len(clean_profiles) == 1:
        persona_map[clean_profiles.index[0]] = "Regular Customers"
        return persona_map

    clean_profiles["spend_rank"] = clean_profiles["Average_Monetary"].rank(pct=True)
    clean_profiles["freq_rank"] = clean_profiles["Average_Frequency"].rank(pct=True)
    clean_profiles["recent_rank"] = (1 - clean_profiles["Average_Recency"].rank(pct=True))
    clean_profiles["value_score"] = (
        clean_profiles["spend_rank"] * 0.45
        + clean_profiles["freq_rank"] * 0.35
        + clean_profiles["recent_rank"] * 0.20
    )

    value_order = clean_profiles.sort_values("value_score", ascending=False).index.tolist()
    risk_order = clean_profiles.sort_values(
        ["Average_Recency", "Average_Frequency", "Average_Monetary"],
        ascending=[False, True, True]
    ).index.tolist()

    if len(value_order) >= 1:
        persona_map[value_order[0]] = "VIP High-Value Customers"
    if len(value_order) >= 2:
        persona_map[value_order[1]] = "Loyal Customers"
    if len(risk_order) >= 1 and risk_order[0] not in persona_map:
        persona_map[risk_order[0]] = "At-Risk Customers"
        
    for cluster_id in clean_profiles.index:
        if cluster_id not in persona_map:
            persona_map[cluster_id] = "Regular Customers"

    for cluster_id in clean_profiles.index:
        if list(persona_map.values()).count(persona_map[cluster_id]) > 1:
            persona_map[cluster_id] = f"Customer Segment {cluster_id}"

    return persona_map

def calculate_clv_estimate(monetary: float, frequency: int, recency: int) -> float:
    """Documented Customer Lifetime Value (CLV) heuristic estimation."""
    avg_order_value = monetary / max(frequency, 1)
    recency_months = max(recency / 30.0, 0.5)
    purchase_rate = frequency / recency_months
    estimated_lifespan_months = max(3.0, 24.0 / (1.0 + 0.1 * recency_months))
    estimated_clv = avg_order_value * purchase_rate * (estimated_lifespan_months / 12.0)
    return round(float(max(estimated_clv, monetary * 1.05)), 2)

def fit_clustering_model(
    rfm_df: pd.DataFrame,
    algorithm: str = "kmeans",
    k: Any = 4
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Execute any of the 6 clustering models on standardized log-scaled RFM features:
    1. K-Means (Centroid-based, fixed K)
    2. Agglomerative Hierarchical (Ward linkage, fixed K)
    3. Gaussian Mixture Model (Probabilistic soft clustering, fixed K)
    4. BIRCH (CF-Tree incremental clustering, fixed K)
    5. DBSCAN (Density-based, dynamic clusters, noise detection)
    6. OPTICS (Ordering points, variable-density clusters, noise detection)
    """
    if hasattr(k, "default"):
        k = k.default
    k = int(k)
    algo = algorithm.lower().strip()
    cache_key = (len(rfm_df), algo, k)
    if cache_key in _clustering_cache:
        return _clustering_cache[cache_key]

    scaled_data, _, _ = prepare_clustering_features(rfm_df)
    n_samples = len(rfm_df)
    k_bounded = max(2, min(k, n_samples - 1))
    
    meta: Dict[str, Any] = {"algorithm": algo, "family": "Centroid-based"}

    # =========================================================================
    # THE SIX CLUSTERING ALGORITHMS IMPLEMENTATION
    # We evaluate 6 models spanning 5 distinct mathematical
    # paradigms (Centroid, Hierarchical, Probabilistic, CF-Tree, and Density).
    # All models receive identical standardized log1p RFM customer vectors.
    # =========================================================================

    if algo in ["hierarchical", "agglomerative"]:
        # ---------------------------------------------------------------------
        # MODEL 2: AGGLOMERATIVE HIERARCHICAL CLUSTERING
        # Paradigm: Hierarchical Connectivity (Bottom-Up)
        # Parameters: n_clusters=k_bounded, linkage='ward', metric='euclidean'
        # 
        # - Starts with each customer as a single cluster and iteratively merges
        #   the pair that minimizes within-cluster variance increase (Ward's criterion).
        # - Does NOT assume purely spherical clusters like K-Means.
        # - Captures fine nested customer hierarchies without random initial centroid seeds.
        # ---------------------------------------------------------------------
        meta["family"] = "Hierarchical Connectivity"
        meta["parameters"] = {"n_clusters": k_bounded, "linkage": "ward", "metric": "euclidean"}
        model = AgglomerativeClustering(n_clusters=k_bounded, linkage="ward")
        labels = model.fit_predict(scaled_data)

    elif algo in ["gmm", "gaussian_mixture"]:
        # ---------------------------------------------------------------------
        # MODEL 3: GAUSSIAN MIXTURE MODEL (GMM)
        # Paradigm: Distribution-based (Probabilistic Soft Clustering)
        # Parameters: n_components=k_bounded, covariance_type='full', random_state=42, n_init=3
        # 
        # - Assumes data is generated from a mixture of k Gaussian distributions.
        # - Uses Expectation-Maximization (EM) to compute posterior probabilities.
        # - 'full' covariance allows clusters to have elliptical shapes with arbitrary orientation,
        #   which accommodates overlapping retail spending behaviors.
        # ---------------------------------------------------------------------
        meta["family"] = "Distribution-based (Probabilistic)"
        meta["parameters"] = {"n_components": k_bounded, "covariance_type": "full", "random_state": 42, "n_init": 3}
        model = GaussianMixture(n_components=k_bounded, random_state=42, n_init=3, covariance_type="full")
        labels = model.fit_predict(scaled_data)
        meta["converged"] = bool(model.converged_)
        meta["n_iter"] = int(model.n_iter_)
        meta["lower_bound"] = round(float(model.lower_bound_), 4)

    elif algo == "birch":
        # ---------------------------------------------------------------------
        # MODEL 4: BIRCH CLUSTERING (Balanced Iterative Reducing and Clustering using Hierarchies)
        # Paradigm: CF-Tree (Clustering Feature Tree) Hierarchical
        # Parameters: n_clusters=k_bounded, threshold=0.5, branching_factor=50
        # 
        # - Designed specifically for ultra-large datasets and continuous transaction streams.
        # - Builds an in-memory Clustering Feature Tree (CF-Tree) storing summary statistics
        #   (N, linear sum LS, squared sum SS) without storing all raw data in memory.
        # - Runs in linear O(N) time and memory, making it the most scalable hierarchical method.
        # ---------------------------------------------------------------------
        meta["family"] = "CF-Tree Hierarchical"
        meta["parameters"] = {"n_clusters": k_bounded, "threshold": 0.5, "branching_factor": 50}
        model = Birch(n_clusters=k_bounded, threshold=0.5, branching_factor=50)
        labels = model.fit_predict(scaled_data)

    elif algo == "dbscan":
        # ---------------------------------------------------------------------
        # MODEL 5: DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
        # Paradigm: Density-based Spatial Clustering
        # Parameters: eps=0.5, min_samples=5, metric='euclidean'
        # 
        # - Does NOT require pre-specifying K (it discovers natural cluster counts).
        # - Groups dense core neighborhoods and labels outliers with density below min_samples
        #   as noise (cluster label -1).
        # - Vital in retail for isolating transactional fraud, bulk wholesale buyers, and return anomalies.
        # ---------------------------------------------------------------------
        meta["family"] = "Density-based Spatial Clustering"
        meta["parameters"] = {"eps": 0.5, "min_samples": 5, "metric": "euclidean"}
        model = DBSCAN(eps=0.5, min_samples=5)
        labels = model.fit_predict(scaled_data)

    elif algo == "optics":
        # ---------------------------------------------------------------------
        # MODEL 6: OPTICS (Ordering Points To Identify the Clustering Structure)
        # Paradigm: Density-based Reachability Ordering
        # Parameters: min_samples=10, xi=0.05, min_cluster_size=0.04
        # 
        # - Overcomes DBSCAN's primary limitation: inability to detect clusters of varying density.
        # - Constructs a reachability distance graph ordering points so valleys correspond to clusters.
        # - Trade-off: Higher computational complexity O(N^2) for pairwise reachability calculation.
        # ---------------------------------------------------------------------
        meta["family"] = "Density-based Reachability Ordering"
        meta["parameters"] = {"min_samples": 10, "xi": 0.05, "min_cluster_size": 0.04}
        model = OPTICS(min_samples=10, xi=0.05, min_cluster_size=0.04)
        labels = model.fit_predict(scaled_data)

    else:
        # ---------------------------------------------------------------------
        # MODEL 1: K-MEANS CLUSTERING (Default Recommended Baseline)
        # Paradigm: Centroid-based Partitioning
        # Parameters: n_clusters=k_bounded, random_state=42, n_init=10
        # 
        # - Iterative expectation-maximization minimizing within-cluster sum of squares (Inertia).
        # - Produces convex Voronoi cells that map cleanly to commercial business tiers (VIP, Loyal, etc.).
        # - High computational efficiency O(k * N * d) with 100% customer assignment coverage.
        # - Inertia is mathematically native ONLY to K-Means.
        # ---------------------------------------------------------------------
        meta["family"] = "Centroid-based Partitioning"
        meta["parameters"] = {"n_clusters": k_bounded, "random_state": 42, "n_init": 10}
        model = KMeans(n_clusters=k_bounded, random_state=42, n_init=10)
        labels = model.fit_predict(scaled_data)
        meta["inertia"] = round(float(model.inertia_), 2)

    result_df = rfm_df.copy()
    result_df["Cluster"] = labels

    profiles = result_df.groupby("Cluster").agg(
        Total_Customers=("CustomerID", "count"),
        Total_Revenue=("Monetary", "sum"),
        Average_Recency=("Recency", "mean"),
        Average_Frequency=("Frequency", "mean"),
        Average_Monetary=("Monetary", "mean"),
        Median_Monetary=("Monetary", "median"),
    ).round(2)

    total_all_revenue = float(result_df["Monetary"].sum())
    total_all_customers = len(result_df)

    persona_map = assign_personas(profiles)

    profiles["Persona"] = [persona_map.get(c, f"Cluster {c}") for c in profiles.index]
    profiles["Pct_Customers"] = ((profiles["Total_Customers"] / total_all_customers) * 100).round(1)
    profiles["Pct_Revenue"] = ((profiles["Total_Revenue"] / total_all_revenue) * 100).round(1)
    profiles["Marketing_Action"] = [PERSONA_ACTIONS.get(p, "Standard customer retention strategy.") for p in profiles["Persona"]]
    profiles["Risk_Level"] = [RISK_LEVELS.get(p, "Moderate Risk") for p in profiles["Persona"]]

    profiles["CLV_Estimate"] = [
        calculate_clv_estimate(row["Average_Monetary"], int(row["Average_Frequency"]), int(row["Average_Recency"]))
        for _, row in profiles.iterrows()
    ]

    result_df["Persona"] = result_df["Cluster"].map(persona_map).fillna("Unassigned")
    result_df["Marketing_Action"] = result_df["Persona"].map(PERSONA_ACTIONS).fillna("Standard retention")
    result_df["Risk_Level"] = result_df["Persona"].map(RISK_LEVELS).fillna("Moderate Risk")
    result_df["CLV_Estimate"] = [
        calculate_clv_estimate(m, int(f), int(r))
        for m, f, r in zip(result_df["Monetary"], result_df["Frequency"], result_df["Recency"])
    ]

    # Calculate 2D PCA for spatial visualization
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(scaled_data)
    result_df["PCA1"] = np.round(pca_coords[:, 0], 3)
    result_df["PCA2"] = np.round(pca_coords[:, 1], 3)
    exp_var = [round(float(v), 4) for v in pca.explained_variance_ratio_]
    meta["k"] = len(np.unique(labels[labels != -1])) if len(np.unique(labels[labels != -1])) > 0 else k_bounded
    meta["explained_variance_ratio"] = exp_var
    meta["pca_explained_variance_ratio"] = exp_var

    _clustering_cache[cache_key] = (result_df, profiles, meta)
    return result_df, profiles, meta
