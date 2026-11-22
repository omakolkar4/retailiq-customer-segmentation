from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

from app.services.rfm import prepare_rfm_features

_validation_cache: Dict[Tuple[int, int, int], Any] = {}

def clear_validation_cache():
    _validation_cache.clear()

def run_clustering_validation(
    rfm_df: pd.DataFrame,
    k: Any = 4,
    n_iterations: Any = 10,
    subsample_ratio: Any = 0.8
) -> Dict[str, Any]:
    """
    Perform rigorous unsupervised stability testing via repeated bootstrap subsampling.
    Evaluates partition consistency using Adjusted Rand Index (ARI) and Silhouette stability.
    """
    if hasattr(k, "default"):
        k = k.default
    k = int(k)
    if hasattr(n_iterations, "default"):
        n_iterations = n_iterations.default
    n_iterations = int(n_iterations)
    if hasattr(subsample_ratio, "default"):
        subsample_ratio = subsample_ratio.default
    subsample_ratio = float(subsample_ratio)

    cache_key = (len(rfm_df), k, n_iterations)
    if cache_key in _validation_cache:
        return _validation_cache[cache_key]

    scaled_data, _ = prepare_rfm_features(rfm_df)
    n_samples = len(scaled_data)
    sample_size = int(n_samples * subsample_ratio)

    # 1. Fit baseline model on the entire dataset
    baseline_model = KMeans(n_clusters=k, random_state=42, n_init=10)
    baseline_labels = baseline_model.fit_predict(scaled_data)

    iterations_data: List[Dict[str, Any]] = []
    ari_scores: List[float] = []
    sil_scores: List[float] = []

    # 2. Resample and measure partition consistency
    rng = np.random.RandomState(42)
    for i in range(1, n_iterations + 1):
        indices = rng.choice(n_samples, size=sample_size, replace=False)
        sub_scaled = scaled_data[indices]

        sub_model = KMeans(n_clusters=k, random_state=i * 10, n_init=5)
        sub_labels = sub_model.fit_predict(sub_scaled)

        # Compare labels of the subsample against full dataset model labels on identical points
        ari = float(adjusted_rand_score(baseline_labels[indices], sub_labels))
        sil = float(silhouette_score(sub_scaled, sub_labels, sample_size=min(1500, sample_size), random_state=42))

        ari_scores.append(ari)
        sil_scores.append(sil)

        iterations_data.append({
            "iteration": i,
            "sample_size": sample_size,
            "ari_score": round(ari, 4),
            "silhouette_score": round(sil, 4),
        })

    # 3. Stability comparison across cluster counts (K = 2 to 6)
    k_comparison: List[Dict[str, Any]] = []
    for test_k in range(2, min(7, n_samples - 1)):
        test_base = KMeans(n_clusters=test_k, random_state=42, n_init=3)
        test_base_labels = test_base.fit_predict(scaled_data)

        k_aris = []
        k_sils = []
        for j in range(2): # 2 fast iterations per K for responsiveness
            idx = rng.choice(n_samples, size=sample_size, replace=False)
            km_sub = KMeans(n_clusters=test_k, random_state=j * 17, n_init=3)
            km_sub_labels = km_sub.fit_predict(scaled_data[idx])
            k_aris.append(adjusted_rand_score(test_base_labels[idx], km_sub_labels))
            k_sils.append(silhouette_score(scaled_data[idx], km_sub_labels, sample_size=min(800, sample_size), random_state=42))

        k_comparison.append({
            "k": test_k,
            "mean_ari": round(float(np.mean(k_aris)), 4),
            "std_ari": round(float(np.std(k_aris)), 4),
            "mean_silhouette": round(float(np.mean(k_sils)), 4),
            "std_silhouette": round(float(np.std(k_sils)), 4),
        })

    methodology = (
        "Unsupervised validation cannot use supervised classification metrics (e.g., accuracy, ROC-AUC) "
        "due to the absence of ground-truth class labels. Instead, stability validation assesses how robust "
        "the discovered customer clusters are against data perturbations via repeated 80% subsampling. "
        "An Adjusted Rand Index (ARI) approaching 1.0 indicates high cluster stability and resilience to sampling variance."
    )

    limitations = (
        "1. High stability does not guarantee business optimal partitions; clusters may be statistically stable yet lack operational distinction.\n"
        "2. ARI measures partition agreement between samples, not semantic truth.\n"
        "3. Outlier sensitivity: extreme spending anomalies can skew centroid locations across resamples."
    )

    result = {
        "selected_k": k,
        "n_iterations": n_iterations,
        "subsample_ratio": subsample_ratio,
        "mean_ari": round(float(np.mean(ari_scores)), 4),
        "std_ari": round(float(np.std(ari_scores)), 4),
        "mean_silhouette": round(float(np.mean(sil_scores)), 4),
        "std_silhouette": round(float(np.std(sil_scores)), 4),
        "iterations": iterations_data,
        "k_comparison": k_comparison,
        "methodology_notes": methodology,
        "limitations": limitations,
    }
    _validation_cache[cache_key] = result
    return result
