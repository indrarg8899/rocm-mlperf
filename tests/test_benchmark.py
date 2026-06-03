"""Unit tests for ROCm MLPerf benchmark runner."""

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_config_loading():
    """Test YAML config loading."""
    from src.utils.config import load_config, BenchmarkConfig

    # Test with dummy config
    config = BenchmarkConfig()
    assert config.model == "resnet50"
    assert config.scenario == "offline"
    assert config.precision == "fp32"


def test_metrics_collector():
    """Test metrics calculation."""
    from src.utils.metrics import MetricsCollector

    collector = MetricsCollector()
    collector.add_latency(10.0)
    collector.add_latency(15.0)
    collector.add_latency(20.0)
    collector.add_throughput(100.0)

    metrics = collector.get_latency_metrics()
    assert metrics.p50 == 15.0
    assert metrics.p90 == 20.0


def test_gpu_monitor_init():
    """Test GPU monitor initialization."""
    from src.utils.gpu_monitor import GPUMonitor

    monitor = GPUMonitor(device_id=0)
    assert monitor.device_id == 0
    assert not monitor._running
    assert len(monitor._stats) == 0


def test_loadgen_wrapper_import():
    """Test that LoadGen wrapper imports correctly."""
    from src.loadgen_wrapper import MLPerfSUT, MLPerfQSL
    assert MLPerfSUT is not None
    assert MLPerfQSL is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
