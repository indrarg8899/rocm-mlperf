"""Configuration loader for MLPerf benchmarks."""

import yaml
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BenchmarkConfig:
    model: str = "resnet50"
    scenario: str = "offline"
    precision: str = "fp32"
    batch_size: int = 64
    max_samples: int = 50000
    data_dir: str = "./datasets"
    log_dir: str = "./logs"
    output_dir: str = "./results"
    deterministic: bool = False
    num_threads: int = 1
    use_graphs: bool = False
    num_gpus: int = 1
    target_latency_ms: Optional[float] = None
    target_accuracy: Optional[float] = None


def load_config(path: str) -> BenchmarkConfig:
    """Load YAML config file into BenchmarkConfig."""
    with open(path) as f:
        data = yaml.safe_load(f) or {}

    config = BenchmarkConfig()
    for key, value in data.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return config
