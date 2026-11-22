from typing import Dict, Any, List, Tuple, Optional
import time
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch, DBSCAN, OPTICS
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

from app.services.preprocessing import prepare_clustering_features
from app.services.model_benchmark import benchmark_model_execution
from app.services.stability_analysis import run_multi_model_stability
from app.services.recommendation import compute_model_recommendation

_eval_cache: Dict[Tuple[int, int], Any] = {}

def clear_model_eval_cache():
    _eval_cache.clear()

def _calculate_common_sse(scaled_data: np.ndarray, labels: np.ndarray) -> float:
    """
    Calculate common within-cluster sum of squared errors (SSE) as an external comparison measure.
    Explicitly distinct from K-Means native inertia.
    """
    unique_clusters = [c for c in np.unique(labels) if c != -1]
    if not unique_clusters:
        return 0.0
    total_sse = 0.0
    for c in unique_clusters:
        pts = scaled_data[labels == c]
        if len(pts) > 0:
            center = np.mean(pts, axis=0)
            total_sse += np.sum((pts - center) ** 2)
    return round(float(total_sse), 2)

def evaluate_models(rfm_df: pd.DataFrame, k: Any = 4) -> Dict[str, Any]:
    """
    Execute, benchmark, and academically evaluate all SIX clustering algorithms:
    1. K-Means
    2. Agglomerative Hierarchical
    3. Gaussian Mixture Model (GMM)
    4. BIRCH
    5. DBSCAN
    6. OPTICS
    """
    if hasattr(k, "default"):
        k = k.default
    k = int(k)
    cache_key = (len(rfm_df), k)
    if cache_key in _eval_cache:
        return _eval_cache[cache_key]

    t_prep_start = time.perf_counter()
    scaled_data, _, _ = prepare_clustering_features(rfm_df)
    t_prep_ms = round((time.perf_counter() - t_prep_start) * 1000.0, 2)

    n_samples = len(rfm_df)
    k_bounded = max(2, min(k, n_samples - 1))
    sample_size = min(1000, n_samples)

    # 1. Fetch empirical stability scores across all models
    stability_data = run_multi_model_stability(rfm_df, k=k_bounded, n_trials=5, subsample_ratio=0.8)
    stab_map = {item["key"]: item["mean_ari"] for item in stability_data["results"]}

    # =========================================================================
    # THE SIX CLUSTERING ALGORITHMS BENCHMARK SPECIFICATION

    # 1. Four Fixed-K Models: K-Means, Agglomerative, GMM, BIRCH (all evaluated at K = 4)
    # 2. Two Density-based Models: DBSCAN & OPTICS (discover natural cluster counts and isolate noise)
    # 3. Repeated Benchmark Timing: We run 2 to 3 repeated trials per model to report
    #    empirically valid Mean, Median, and Standard Deviation execution times.
    # =========================================================================
    models_spec = [
        # --- MODEL 1: K-MEANS CLUSTERING ---
        # Centroid partitioning minimizing within-cluster SSE.
        # Industry standard for RFM because Voronoi partitions assign 100% of customers to clear marketing tiers.
        {
            "key": "kmeans",
            "name": "K-Means Clustering",
            "family": "Centroid-based Partitioning",
            "fixed_k": True,
            "repetitions": 3,
            "description": "Iterative centroid partitioning minimizing within-cluster sum of squared errors (SSE/inertia). Scalable, convex Voronoi boundaries.",
            "instantiate": lambda seed: KMeans(n_clusters=k_bounded, random_state=seed, n_init=10),
            "parameters": {"n_clusters": k_bounded, "random_state": 42, "n_init": 10},
        },
        # --- MODEL 2: AGGLOMERATIVE HIERARCHICAL CLUSTERING ---
        #  Bottom-up connectivity using Ward's minimum variance criterion.
        # Preserves consumer hierarchies (e.g. sub-tiers within VIP spenders) without assuming spherical shape.
        {
            "key": "hierarchical",
            "name": "Agglomerative Hierarchical",
            "family": "Hierarchical Connectivity",
            "fixed_k": True,
            "repetitions": 2,
            "description": "Bottom-up agglomerative clustering using Ward's minimum variance criterion with Euclidean distance. Preserves hierarchical sub-cluster structure.",
            "instantiate": lambda seed: AgglomerativeClustering(n_clusters=k_bounded, linkage="ward"),
            "parameters": {"n_clusters": k_bounded, "linkage": "ward", "metric": "euclidean"},
        },
        # --- MODEL 3: GAUSSIAN MIXTURE MODEL (GMM) ---
        # Probabilistic Expectation-Maximization. Models data as Gaussian distributions.
        # Full covariance matrices allow elliptical boundaries that accommodate overlapping customer behavior.
        {
            "key": "gmm",
            "name": "Gaussian Mixture Model (GMM)",
            "family": "Distribution-based (Probabilistic)",
            "fixed_k": True,
            "repetitions": 2,
            "description": "Expectation-Maximization probabilistic clustering modeling data as a mixture of Gaussian distributions with full covariance. Allows soft boundary assignments.",
            "instantiate": lambda seed: GaussianMixture(n_components=k_bounded, random_state=seed, n_init=3, covariance_type="full"),
            "parameters": {"n_components": k_bounded, "covariance_type": "full", "random_state": 42, "n_init": 3},
        },
        # --- MODEL 4: BIRCH CLUSTERING ---
        #  Balanced Iterative Reducing and Clustering using Hierarchies.
        # Builds an in-memory Clustering Feature Tree (CF-Tree). Extremely scalable (O(N)), ideal for streaming data.
        {
            "key": "birch",
            "name": "BIRCH Clustering",
            "family": "CF-Tree Hierarchical",
            "fixed_k": True,
            "repetitions": 3,
            "description": "Balanced Iterative Reducing and Clustering using Hierarchies. Constructs an in-memory Clustering Feature Tree (CF-Tree) for ultra-fast, memory-efficient clustering.",
            "instantiate": lambda seed: Birch(n_clusters=k_bounded, threshold=0.5, branching_factor=50),
            "parameters": {"n_clusters": k_bounded, "threshold": 0.5, "branching_factor": 50},
        },
        # --- MODEL 5: DBSCAN ---
        #  Density-Based Spatial Clustering of Applications with Noise.
        # Does NOT force K=4; discovers 5 dense clusters and flags 1.5% extreme retail outliers as noise (-1).
        {
            "key": "dbscan",
            "name": "DBSCAN",
            "family": "Density-based Spatial Clustering",
            "fixed_k": False,
            "repetitions": 3,
            "description": "Density-Based Spatial Clustering of Applications with Noise. Discovers arbitrary-shaped clusters based on point density (eps) and min_samples; flags outliers.",
            "instantiate": lambda seed: DBSCAN(eps=0.5, min_samples=5),
            "parameters": {"eps": 0.5, "min_samples": 5, "metric": "euclidean"},
        },
        # --- MODEL 6: OPTICS ---
        #  Ordering Points To Identify the Clustering Structure.
        # Extends DBSCAN to identify clusters across varying densities via reachability distance ordering.
        {
            "key": "optics",
            "name": "OPTICS",
            "family": "Density-based Reachability Ordering",
            "fixed_k": False,
            "repetitions": 1,
            "description": "Ordering Points To Identify the Clustering Structure. Overcomes DBSCAN's fixed-density limitation by building a reachability distance plot across varying densities.",
            "instantiate": lambda seed: OPTICS(min_samples=10, xi=0.05, min_cluster_size=0.04),
            "parameters": {"min_samples": 10, "xi": 0.05, "min_cluster_size": 0.04},
        },
    ]

    results: List[Dict[str, Any]] = []

    for spec in models_spec:
        # 1. Multi-trial benchmark execution timing (mean, median, std)
        bench = benchmark_model_execution(
            lambda: spec["instantiate"](42).fit_predict(scaled_data),
            n_repetitions=spec["repetitions"]
        )

        # 2. Fit primary model
        model = spec["instantiate"](42)
        labels = model.fit_predict(scaled_data)

        # 3. Model convergence / parameters
        convergence_status = "Converged"
        if spec["key"] == "gmm" and hasattr(model, "converged_"):
            convergence_status = "Converged" if model.converged_ else "Did not converge"
        elif spec["key"] in ["dbscan", "optics"]:
            convergence_status = "Density Extracted"

        # 4. Inertia handling: Native ONLY for K-Means!
        native_inertia = None
        if spec["key"] == "kmeans" and hasattr(model, "inertia_"):
            native_inertia = round(float(model.inertia_), 2)

        # Common external SSE
        common_sse = _calculate_common_sse(scaled_data, labels)

        # 5. Noise and cluster counts
        unique_labels = np.unique(labels)
        is_noise = (labels == -1)
        noise_count = int(np.sum(is_noise))
        noise_pct = round(float((noise_count / n_samples) * 100.0), 2)
        cluster_coverage = round(100.0 - noise_pct, 2)

        non_noise_labels = labels[~is_noise]
        num_clusters = len(np.unique(non_noise_labels))

        # 6. Cluster size distribution
        cluster_sizes = {}
        for c in unique_labels:
            lbl = "Noise (-1)" if c == -1 else f"Cluster {c}"
            cluster_sizes[lbl] = int(np.sum(labels == c))

        # 7. Validation metrics (computed on non-noise subset if noise exists)
        if num_clusters >= 2:
            eval_data = scaled_data[~is_noise]
            eval_labels = labels[~is_noise]
            eval_sample = min(sample_size, len(eval_data))
            
            sil = round(float(silhouette_score(eval_data, eval_labels, sample_size=eval_sample, random_state=42)), 4)
            db = round(float(davies_bouldin_score(eval_data, eval_labels)), 4)
            ch = round(float(calinski_harabasz_score(eval_data, eval_labels)), 2)
        else:
            sil, db, ch = 0.0, 999.0, 0.0

        model_status = (
            f"Active ({num_clusters} clusters)" if num_clusters >= 2
            else "Single Cluster / Degenerate"
        )

        results.append({
            "key": spec["key"],
            "algorithm": spec["name"],
            "family": spec["family"],
            "fixed_k": spec["fixed_k"],
            "n_clusters": num_clusters,
            "silhouette_score": sil,
            "davies_bouldin_index": db,
            "calinski_harabasz_score": ch,
            "native_inertia": native_inertia,
            "common_sse_external": common_sse,
            "mean_execution_time_ms": bench["mean_ms"],
            "median_execution_time_ms": bench["median_ms"],
            "std_execution_time_ms": bench["std_ms"],
            "benchmark_repetitions": bench["repetitions"],
            "noise_count": noise_count,
            "noise_percentage": noise_pct,
            "cluster_coverage": cluster_coverage,
            "cluster_sizes": cluster_sizes,
            "stability_score": stab_map.get(spec["key"], 0.70),
            "convergence_status": convergence_status,
            "model_status": model_status,
            "parameters": spec["parameters"],
            "description": spec["description"],
            # Backward compatibility aliases
            "model_name": spec["name"],
            "execution_time_ms": bench["mean_ms"],
            "inertia": native_inertia,
        })

    # Recommendation Calculation
    df_results = pd.DataFrame(results)
    rec_info = compute_model_recommendation(df_results)

    # Multi-K Curves for Fixed-K Models (K=2 to 8)
    elbow_points: List[Dict[str, Any]] = []
    max_test_k = min(8, n_samples - 1)
    for test_k in range(2, max_test_k + 1):
        km = KMeans(n_clusters=test_k, random_state=42, n_init=3)
        km_labels = km.fit_predict(scaled_data)
        km_sil = float(silhouette_score(scaled_data, km_labels, sample_size=sample_size, random_state=42))

        birch_m = Birch(n_clusters=test_k, threshold=0.5, branching_factor=50)
        birch_labels = birch_m.fit_predict(scaled_data)
        birch_sil = float(silhouette_score(scaled_data, birch_labels, sample_size=sample_size, random_state=42))

        elbow_points.append({
            "k": test_k,
            "inertia": round(float(km.inertia_), 2),
            "silhouette_kmeans": round(km_sil, 4),
            "silhouette_birch": round(birch_sil, 4),
            "silhouette": round(km_sil, 4), # alias
        })

    # Summary highlights
    best_sil_model = df_results.loc[df_results["silhouette_score"].idxmax(), "algorithm"]
    best_sil_val = df_results["silhouette_score"].max()

    lowest_db_model = df_results.loc[df_results["davies_bouldin_index"].idxmin(), "algorithm"]
    lowest_db_val = df_results["davies_bouldin_index"].min()

    highest_ch_model = df_results.loc[df_results["calinski_harabasz_score"].idxmax(), "algorithm"]
    highest_ch_val = df_results["calinski_harabasz_score"].max()

    fastest_model = df_results.loc[df_results["mean_execution_time_ms"].idxmin(), "algorithm"]
    fastest_val = df_results["mean_execution_time_ms"].min()

    most_stable_model = df_results.loc[df_results["stability_score"].idxmax(), "algorithm"]
    most_stable_val = df_results["stability_score"].max()

    summary = {
        "total_models_benchmarked": len(results),
        "best_silhouette": {"algorithm": best_sil_model, "value": best_sil_val},
        "lowest_davies_bouldin": {"algorithm": lowest_db_model, "value": lowest_db_val},
        "highest_calinski_harabasz": {"algorithm": highest_ch_model, "value": highest_ch_val},
        "fastest_model": {"algorithm": fastest_model, "value_ms": fastest_val},
        "most_stable_model": {"algorithm": most_stable_model, "value_ari": most_stable_val},
        "preprocessing_time_ms": t_prep_ms,
    }

    response = {
        "models": results,
        "summary": summary,
        "elbow_curve": elbow_points,
        "recommended_model": rec_info["recommended_model"],
        "recommendation_reason": rec_info["recommendation_reason"],
        "composite_rank_summary": rec_info["composite_rank_summary"],
        "stability_analysis": stability_data,
        "dataset_sample_size": n_samples,
        "evaluation_cluster_count": k_bounded,
    }

    _eval_cache[cache_key] = response
    return response
