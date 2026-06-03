"""MLPerf metrics calculator."""

import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class LatencyMetrics:
    mean: float
    p50: float
    p90: float
    p95: float
    p99: float
    min_val: float
    max_val: float


class MetricsCollector:
    """Collect and compute MLPerf metrics."""

    def __init__(self):
        self.latencies: List[float] = []
        self.throughput_samples: List[float] = []

    def add_latency(self, latency_ms: float):
        self.latencies.append(latency_ms)

    def add_throughput(self, throughput: float):
        self.throughput_samples.append(throughput)

    def get_latency_metrics(self) -> Optional[LatencyMetrics]:
        if not self.latencies:
            return None
        arr = np.array(self.latencies)
        return LatencyMetrics(
            mean=float(np.mean(arr)),
            p50=float(np.percentile(arr, 50)),
            p90=float(np.percentile(arr, 90)),
            p95=float(np.percentile(arr, 95)),
            p99=float(np.percentile(arr, 99)),
            min_val=float(np.min(arr)),
            max_val=float(np.max(arr)),
        )

    def get_throughput(self) -> float:
        if not self.throughput_samples:
            return 0.0
        return float(np.mean(self.throughput_samples))

    def summary(self) -> Dict:
        latency = self.get_latency_metrics()
        return {
            "throughput_samples_per_sec": self.get_throughput(),
            "latency_p50_ms": latency.p50 if latency else None,
            "latency_p90_ms": latency.p90 if latency else None,
            "latency_p99_ms": latency.p99 if latency else None,
            "total_queries": len(self.latencies),
        }
