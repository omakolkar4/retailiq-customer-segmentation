from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch, DBSCAN, OPTICS
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score

from app.services.preprocessing import prepare_clustering_features

_stability_cache: Dict[Tuple[int, int, int], Any] = {}

def clear_stability_cache():
    _stability_cache.clear()

def run_multi_model_stability(
    rfm_df: pd.DataFrame,
    k: int = 4,
    n_trials: int = 5,
    subsample_ratio: float = 0.8
) -> Dict[str, Any]:
    """
    Perform rigorous unsupervised stability testing via repeated bootstrap subsampling (80%).
    Evaluates partition consistency across multiple clustering algorithms using Adjusted Rand Index (ARI).
    
    ARI is permutation-invariant: it accurately scores clustering agreement regardless of arbitrary cluster ID indexing.
    For density models (DBSCAN/OPTICS), noise points (-1) are treated as an unassigned outlier class.
    """
    cache_key = (len(rfm_df), k, n_trials)
    if cache_key in _stability_cache:
        return _stability_cache[cache_key]

    scaled_data, _, _ = prepare_clustering_features(rfm_df)
    n_samples = len(scaled_data)
    k_bounded = max(2, min(k, n_samples - 1))

    # Representative stratified subsample ceiling (1000) for real-time responsiveness and non-parametric rigor
    max_eval_samples = min(1000, n_samples)
    if n_samples > max_eval_samples:
        rng_pre = np.random.RandomState(42)
        eval_idx = rng_pre.choice(n_samples, size=max_eval_samples, replace=False)
        eval_data = scaled_data[eval_idx]
        eval_n = max_eval_samples
    else:
        eval_data = scaled_data
        eval_n = n_samples
        
    eval_sub_size = int(eval_n * subsample_ratio)

    # =========================================================================
    # THE SIX MODELS FOR BOOTSTRAP STABILITY TESTING
    #
    # Traditional supervised cross-validation (accuracy, ROC-AUC) is mathematically
    # invalid here because unsupervised retail clustering has NO ground-truth labels.
    # Instead, we validate stability by drawing repeated 80% subsamples without replacement.
    # The Adjusted Rand Index (ARI) evaluates whether common customers are grouped
    # together consistently across perturbations. ARI is immune to cluster label permutation!
    # =========================================================================
    models_to_test = [
        {"name": "K-Means Clustering", "key": "kmeans"},
        {"name": "Agglomerative Hierarchical", "key": "hierarchical"},
        {"name": "Gaussian Mixture Model (GMM)", "key": "gmm"},
        {"name": "BIRCH Clustering", "key": "birch"},
        {"name": "DBSCAN", "key": "dbscan"},
        {"name": "OPTICS", "key": "optics"},
    ]

    def _instantiate_and_fit(key: str, data: np.ndarray, trial_seed: int):
        # Fit model on subsample with controlled seeds
        if key == "kmeans":
            # K-Means: Tested with varying initialization seeds per trial
            m = KMeans(n_clusters=k_bounded, random_state=trial_seed, n_init=3)
            return m.fit_predict(data)
        elif key == "hierarchical":
            # Agglomerative: Deterministic linkage (Ward)
            m = AgglomerativeClustering(n_clusters=k_bounded, linkage="ward")
            return m.fit_predict(data)
        elif key == "gmm":
            # GMM: Expectation-Maximization soft clustering with full covariance
            m = GaussianMixture(n_components=k_bounded, random_state=trial_seed, n_init=1, covariance_type="full")
            return m.fit_predict(data)
        elif key == "birch":
            # BIRCH: In-memory CF-Tree clustering
            m = Birch(n_clusters=k_bounded, threshold=0.5, branching_factor=50)
            return m.fit_predict(data)
        elif key == "dbscan":
            # DBSCAN: Density spatial clustering (noise points labeled -1 are treated as their own outlier partition)
            m = DBSCAN(eps=0.5, min_samples=5)
            return m.fit_predict(data)
        elif key == "optics":
            # OPTICS: Density reachability ordering
            m = OPTICS(min_samples=10, xi=0.05, min_cluster_size=0.04)
            return m.fit_predict(data)
        return np.zeros(len(data))

    stability_results: List[Dict[str, Any]] = []

    for m_info in models_to_test:
        key = m_info["key"]
        name = m_info["name"]

        # Baseline fit on evaluation set
        baseline_labels = _instantiate_and_fit(key, eval_data, trial_seed=42)
        
        ari_scores: List[float] = []
        rng = np.random.RandomState(42)

        for t in range(n_trials):
            sub_indices = rng.choice(eval_n, size=eval_sub_size, replace=False)
            sub_data = eval_data[sub_indices]
            sub_labels = _instantiate_and_fit(key, sub_data, trial_seed=(t + 1) * 11)

            # Measure agreement on common subset
            ari = float(adjusted_rand_score(baseline_labels[sub_indices], sub_labels))
            ari_scores.append(ari)

        arr = np.array(ari_scores)
        mean_ari = round(float(np.mean(arr)), 4)
        std_ari = round(float(np.std(arr)), 4)

        stability_results.append({
            "algorithm": name,
            "key": key,
            "mean_ari": max(-1.0, min(1.0, mean_ari)),
            "std_ari": std_ari,
            "trials": n_trials,
            "subsample_ratio": subsample_ratio,
            "stability_status": "High Stability" if mean_ari >= 0.75 else ("Moderate Stability" if mean_ari >= 0.50 else "Variable Stability"),
        })

    result = {
        "trials": n_trials,
        "subsample_ratio": subsample_ratio,
        "results": stability_results,
        "methodology": (
            "Because ground-truth labels do not exist in unsupervised retail clustering, traditional supervised "
            "cross-validation is mathematically invalid. We perform non-parametric stability validation: repeated "
            "80% subsampling across 5 independent trials. The Adjusted Rand Index (ARI) evaluates consistency "
            "between subsample partitions and baseline partitions on identical customer vectors, immune to label switching."
        ),
    }

    _stability_cache[cache_key] = result
    return result
