import time
from typing import Dict, Any, List, Callable
import numpy as np

def benchmark_model_execution(
    fit_fn: Callable[[], Any],
    n_repetitions: int = 3
) -> Dict[str, float]:
    """
    Execute repeated benchmark trials to calculate robust execution timing:
    - Mean execution time (ms)
    - Median execution time (ms)
    - Standard deviation (ms)
    - Repetitions count
    """
    times_ms: List[float] = []
    
    for _ in range(n_repetitions):
        t0 = time.perf_counter()
        fit_fn()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)
        
    arr = np.array(times_ms)
    return {
        "mean_ms": round(float(np.mean(arr)), 2),
        "median_ms": round(float(np.median(arr)), 2),
        "std_ms": round(float(np.std(arr)), 2),
        "repetitions": n_repetitions,
    }
