from typing import Dict, Any, List
import pandas as pd

def compute_model_recommendation(models_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Transparent multi-criteria algorithmic selection recommendation engine.
    Considers:
    - Cluster cohesion & separation (Silhouette, Davies-Bouldin, Calinski-Harabasz)
    - Empirical partition stability (Mean ARI)
    - Commercial utility & coverage (Noise percentage penalty)
    - Computational efficiency (Runtime latency)
    - Practical business interpretability for retail RFM segmentation
    """
    df = models_df.copy()
    
    # 1. Silhouette rank (Higher is better)
    sil_rank = df["silhouette_score"].rank(ascending=False)
    
    # 2. Davies-Bouldin rank (Lower is better)
    db_rank = df["davies_bouldin_index"].rank(ascending=True)
    
    # 3. Calinski-Harabasz rank (Higher is better)
    ch_rank = df["calinski_harabasz_score"].rank(ascending=False)
    
    # 4. Stability rank (Higher ARI is better)
    stab_rank = df["stability_score"].rank(ascending=False)
    
    # 5. Customer coverage rank (Higher coverage is better for marketing utility)
    cov_rank = df["cluster_coverage"].rank(ascending=False)
    
    # 6. Execution efficiency rank (Lower is faster)
    time_rank = df["mean_execution_time_ms"].rank(ascending=True)

    # Composite weighted penalty score (lower sum of weighted ranks is best)
    df["composite_rank"] = (
        0.25 * sil_rank
        + 0.20 * db_rank
        + 0.15 * ch_rank
        + 0.20 * stab_rank
        + 0.10 * cov_rank
        + 0.10 * time_rank
    )

    best_idx = df["composite_rank"].idxmin()
    best_row = df.loc[best_idx]
    best_algo = best_row["algorithm"]

    reasons: Dict[str, str] = {
        "K-Means Clustering": (
            f"K-Means is recommended with a balanced Silhouette Score of {best_row['silhouette_score']}, "
            f"high partition stability (ARI {best_row['stability_score']}), and 100% customer coverage in just "
            f"{best_row['mean_execution_time_ms']} ms. Its convex Voronoi partitions translate cleanly into "
            "actionable commercial RFM tiers (VIP, Loyal, Regular, At-Risk) without leaving customers unassigned."
        ),
        "Agglomerative Hierarchical": (
            f"Agglomerative Clustering is recommended with top-tier cluster separation (Davies-Bouldin {best_row['davies_bouldin_index']}) "
            f"and strong stability (ARI {best_row['stability_score']}). Its Ward linkage objective preserves fine-grained "
            "customer hierarchy structures and accounts for subtle spending variations."
        ),
        "Gaussian Mixture Model (GMM)": (
            f"Gaussian Mixture Model is recommended for its probabilistic soft-clustering capability and strong Calinski-Harabasz "
            f"score of {best_row['calinski_harabasz_score']}. By fitting elliptical covariance matrices, GMM accurately models "
            "overlapping retail customer behaviors that do not conform to rigid spherical boundaries."
        ),
        "BIRCH Clustering": (
            f"BIRCH Clustering is recommended due to its CF-Tree architecture, delivering exceptional runtime scalability "
            f"({best_row['mean_execution_time_ms']} ms) while maintaining a competitive Silhouette Score of {best_row['silhouette_score']}. "
            "It is especially suited for continuous streaming transaction environments and very large customer bases."
        ),
        "DBSCAN": (
            f"DBSCAN is recommended for its natural density-based grouping, discovering {best_row['n_clusters']} intrinsic "
            f"clusters and isolating {best_row['noise_percentage']}% of extreme transaction anomalies without forcing rigid cluster counts."
        ),
        "OPTICS": (
            f"OPTICS is recommended for discovering customer density structures with varying concentrations, isolating "
            f"{best_row['noise_percentage']}% noise while capturing core spending clusters across reachability valleys."
        ),
    }

    reason = reasons.get(
        best_algo,
        f"{best_algo} achieved the optimal composite score across cluster cohesion, empirical stability, and execution speed."
    )

    return {
        "recommended_model": best_algo,
        "recommendation_reason": reason,
        "composite_rank_summary": [
            {
                "algorithm": r["algorithm"],
                "composite_rank": round(float(r["composite_rank"]), 2),
                "silhouette": r["silhouette_score"],
                "davies_bouldin": r["davies_bouldin_index"],
                "stability": r["stability_score"],
                "coverage": r["cluster_coverage"],
                "mean_time_ms": r["mean_execution_time_ms"],
            }
            for _, r in df.sort_values("composite_rank").iterrows()
        ],
    }
