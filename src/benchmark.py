"""
ROCm MLPerf - Main benchmark runner
Integrates MLPerf LoadGen with ROCm/PyTorch inference
"""

import argparse
import json
import os
import time
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

import numpy as np

try:
    import mlperf_loadgen as lg
    HAS_LOADGEN = True
except ImportError:
    HAS_LOADGEN = False
    print("⚠️  mlperf_loadgen not installed. Running in dry-run mode.")

import torch
import torch.backends.cudnn as cudnn

from src.loadgen_wrapper import MLPerfSUT, MLPerfQSL
from src.utils.config import load_config, BenchmarkConfig
from src.utils.gpu_monitor import GPUMonitor
from src.utils.metrics import MetricsCollector


@dataclass
class BenchmarkResult:
    model: str
    scenario: str
    throughput: float
    latency_p50: Optional[float] = None
    latency_p90: Optional[float] = None
    latency_p99: Optional[float] = None
    accuracy: Optional[float] = None
    total_samples: int = 0
    duration_s: float = 0.0
    gpu_util_avg: float = 0.0
    power_avg_w: float = 0.0


class BenchmarkRunner:
    """Main benchmark orchestrator."""

    def __init__(self, config_path: str, output_path: Optional[str] = None, profile: bool = False):
        self.config = load_config(config_path)
        self.output_path = output_path
        self.profile = profile
        self.device = self._setup_device()
        self.results: List[BenchmarkResult] = []

    def _setup_device(self) -> torch.device:
        if not torch.cuda.is_available():
            raise RuntimeError("No GPU available. ROCm requires AMD GPU.")
        device = torch.device("cuda:0")
        cudnn.benchmark = True
        cudnn.deterministic = self.config.deterministic
        print(f"🖥  Device: {torch.cuda.get_device_name(0)}")
        print(f"   ROCm: {torch.version.hip or 'unknown'}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
        return device

    def run(self) -> BenchmarkResult:
        """Run full MLPerf benchmark."""
        print(f"\n{'='*60}")
        print(f"  Running: {self.config.model} ({self.config.scenario})")
        print(f"{'='*60}\n")

        # Start GPU monitoring
        monitor = GPUMonitor(device_id=0)
        monitor.start(interval=1.0)

        # Load model
        model = self._load_model()
        print(f"✅ Model loaded: {self.config.model}")

        # Load dataset
        dataset = self._load_dataset()
        print(f"✅ Dataset loaded: {len(dataset)} samples")

        # Create LoadGen SUT and QSL
        sut = MLPerfSUT(model=model, config=self.config, device=self.device)
        qsl = MLPerfQSL(dataset=dataset, total_count=self.config.max_samples)

        # Configure LoadGen
        if HAS_LOADGEN:
            result = self._run_loadgen(sut, qsl)
        else:
            result = self._run_dryrun(sut, qsl)

        # Stop monitoring
        gpu_stats = monitor.stop()

        # Collect results
        result.gpu_util_avg = gpu_stats.get("util_avg", 0.0)
        result.power_avg_w = gpu_stats.get("power_avg", 0.0)
        self.results.append(result)

        # Export
        if self.output_path:
            self._export_results()

        self._print_result(result)
        return result

    def _load_model(self):
        """Load model based on config."""
        model_name = self.config.model.lower()
        if model_name == "resnet50":
            from src.models.resnet50 import load_resnet50
            return load_resnet50(self.device, self.config.precision)
        elif model_name == "bert":
            from src.models.bert import load_bert
            return load_bert(self.device, self.config.precision)
        elif model_name == "dlrm":
            from src.models.dlrm import load_dlrm
            return load_dlrm(self.device, self.config.precision)
        elif model_name == "unet3d":
            from src.models.unet3d import load_unet3d
            return load_unet3d(self.device, self.config.precision)
        else:
            raise ValueError(f"Unknown model: {self.config.model}")

    def _load_dataset(self):
        """Load dataset based on config."""
        model_name = self.config.model.lower()
        if model_name == "resnet50":
            from src.datasets.imagenet import load_imagenet
            return load_imagenet(self.config.data_dir, split="val")
        elif model_name == "bert":
            from src.datasets.squad import load_squad
            return load_squad(self.config.data_dir)
        elif model_name == "dlrm":
            from src.datasets.kaggle import load_criteo
            return load_criteo(self.config.data_dir)
        else:
            return list(range(self.config.max_samples))

    def _run_loadgen(self, sut: MLPerfSUT, qsl: MLPerfQSL) -> BenchmarkResult:
        """Run with MLPerf LoadGen."""
        settings = lg.TestSettings()
        settings.FromConfig(lg.GENERIC_SETTINGS, self.config.model, self.config.scenario)
        settings.scenario = getattr(lg.TestScenario, self.config.scenario.upper())
        settings.mode = lg.TestMode.PerformanceOnly

        if self.config.max_samples:
            settings.min_query_count = self.config.max_samples
            settings.max_query_count = self.config.max_samples

        log_settings = lg.LogSettings()
        log_settings.log_output.outdir = self.config.log_dir
        log_settings.log_output.copy_detail_to_stdout = False
        log_settings.enable_trace = self.profile

        print(f"🚀 Starting LoadGen benchmark...")
        start = time.time()
        lg.StartTestWithLogSettings(sut.sut, qsl.qsl, settings, log_settings)
        duration = time.time() - start

        # Parse results
        result_file = os.path.join(self.config.log_dir, "mlperf_log_summary.txt")
        throughput, latencies = self._parse_loadgen_results(result_file)

        return BenchmarkResult(
            model=self.config.model,
            scenario=self.config.scenario,
            throughput=throughput,
            latency_p50=latencies.get("p50"),
            latency_p90=latencies.get("p90"),
            latency_p99=latencies.get("p99"),
            total_samples=self.config.max_samples,
            duration_s=duration,
        )

    def _run_dryrun(self, sut: MLPerfSUT, qsl: MLPerfQSL) -> BenchmarkResult:
        """Dry-run without LoadGen."""
        print("⚠️  Running in dry-run mode (LoadGen not installed)")
        samples = qsl.get_samples(list(range(min(100, len(qsl.dataset)))))
        start = time.time()
        sut.issue_queries(samples)
        duration = time.time() - start
        throughput = len(samples) / duration

        return BenchmarkResult(
            model=self.config.model,
            scenario=self.config.scenario,
            throughput=throughput,
            total_samples=len(samples),
            duration_s=duration,
        )

    def _parse_loadgen_results(self, path: str) -> tuple:
        """Parse MLPerf log_summary.txt."""
        throughput = 0.0
        latencies = {}
        if not os.path.exists(path):
            return throughput, latencies
        with open(path) as f:
            for line in f:
                if "Samples per second" in line:
                    throughput = float(line.split(":")[-1].strip())
                elif "50.00 percentile" in line:
                    latencies["p50"] = float(line.split(":")[-1].strip().replace("ms", ""))
                elif "90.00 percentile" in line:
                    latencies["p90"] = float(line.split(":")[-1].strip().replace("ms", ""))
                elif "99.00 percentile" in line:
                    latencies["p99"] = float(line.split(":")[-1].strip().replace("ms", ""))
        return throughput, latencies

    def _export_results(self):
        """Export results to JSON."""
        data = []
        for r in self.results:
            data.append({
                "model": r.model,
                "scenario": r.scenario,
                "throughput": r.throughput,
                "latency_p50_ms": r.latency_p50,
                "latency_p90_ms": r.latency_p90,
                "latency_p99_ms": r.latency_p99,
                "accuracy": r.accuracy,
                "total_samples": r.total_samples,
                "duration_s": r.duration_s,
                "gpu_util_avg": r.gpu_util_avg,
                "power_avg_w": r.power_avg_w,
            })
        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)
        with open(self.output_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"📄 Results saved to {self.output_path}")

    def _print_result(self, r: BenchmarkResult):
        print(f"\n{'='*60}")
        print(f"  📊 RESULTS: {r.model} ({r.scenario})")
        print(f"{'='*60}")
        print(f"  Throughput:    {r.throughput:,.1f} samples/s")
        if r.latency_p50:
            print(f"  Latency P50:   {r.latency_p50:.1f} ms")
        if r.latency_p99:
            print(f"  Latency P99:   {r.latency_p99:.1f} ms")
        if r.accuracy:
            print(f"  Accuracy:      {r.accuracy:.2%}")
        print(f"  GPU Util:      {r.gpu_util_avg:.1f}%")
        print(f"  Power:         {r.power_avg_w:.0f} W")
        print(f"  Duration:      {r.duration_s:.1f} s")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="ROCm MLPerf Benchmark Runner")
    parser.add_argument("--config", required=True, help="YAML config file")
    parser.add_argument("--output", help="JSON output file")
    parser.add_argument("--profile", action="store_true", help="Enable rocprof profiling")
    parser.add_argument("--deterministic", action="store_true", help="Deterministic mode")
    parser.add_argument("--dry-run", action="store_true", help="Dry-run mode")
    args = parser.parse_args()

    runner = BenchmarkRunner(
        config_path=args.config,
        output_path=args.output,
        profile=args.profile,
    )
    runner.run()


if __name__ == "__main__":
    main()
