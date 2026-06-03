"""GPU monitoring for AMD ROCm devices using rocm-smi."""

import subprocess
import time
import threading
import csv
import os
from typing import Dict, List, Optional


class GPUMonitor:
    """Real-time GPU metrics collection via rocm-smi."""

    def __init__(self, device_id: int = 0):
        self.device_id = device_id
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._stats: List[Dict] = []
        self._interval = 1.0

    def _query_rocm_smi(self) -> Dict:
        """Query rocm-smi for GPU metrics."""
        try:
            result = subprocess.run(
                ["rocm-smi", "--showuse", "--showmemuse", "--showtemp", "--showpower", "--csv"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode != 0:
                return self._fallback_query()

            # Parse CSV output
            lines = result.stdout.strip().split("\n")
            if len(lines) < 2:
                return {}

            # Find our device in CSV
            for line in lines[1:]:
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 5:
                    return {
                        "utilization": float(parts[2].replace("%", "")) if "%" in parts[2] else 0.0,
                        "memory_used_mb": float(parts[3].replace("MB", "")) if "MB" in parts[3] else 0.0,
                        "temperature": float(parts[4].replace("C", "").strip()) if "C" in parts[4] else 0.0,
                        "power_w": float(parts[5].replace("W", "")) if "W" in parts[5] else 0.0,
                    }
            return {}
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return self._fallback_query()

    def _fallback_query(self) -> Dict:
        """Fallback: parse rocm-smi non-CSV output."""
        try:
            result = subprocess.run(["rocm-smi"], capture_output=True, text=True, timeout=5)
            stats = {}
            for line in result.stdout.split("\n"):
                line = line.strip()
                if "GPU use" in line:
                    stats["utilization"] = float(line.split(":")[-1].replace("%", "").strip())
                elif "GPU Memory" in line or "VRAM" in line:
                    parts = line.split(":")[-1].strip().split("/")
                    if len(parts) == 2:
                        stats["memory_used_mb"] = float(parts[0].replace("MB", "").strip())
                elif "Temperature" in line:
                    stats["temperature"] = float(line.split(":")[-1].replace("C", "").strip())
                elif "Average" in line or "Power" in line:
                    val = line.split(":")[-1].strip()
                    if "W" in val:
                        stats["power_w"] = float(val.replace("W", "").strip())
            return stats
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return {}

    def _monitor_loop(self):
        """Background monitoring thread."""
        while self._running:
            stats = self._query_rocm_smi()
            stats["timestamp"] = time.time()
            self._stats.append(stats)
            time.sleep(self._interval)

    def start(self, interval: float = 1.0):
        """Start background GPU monitoring."""
        self._interval = interval
        self._running = True
        self._stats.clear()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self) -> Dict:
        """Stop monitoring and return average stats."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)

        if not self._stats:
            return {"util_avg": 0.0, "power_avg": 0.0, "temp_max": 0.0}

        return {
            "util_avg": sum(s.get("utilization", 0) for s in self._stats) / len(self._stats),
            "power_avg": sum(s.get("power_w", 0) for s in self._stats) / len(self._stats),
            "temp_max": max((s.get("temperature", 0) for s in self._stats), default=0),
            "mem_avg_mb": sum(s.get("memory_used_mb", 0) for s in self._stats) / len(self._stats),
            "samples": len(self._stats),
        }

    def export_csv(self, path: str):
        """Export collected stats to CSV."""
        if not self._stats:
            return
        keys = self._stats[0].keys()
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self._stats)
        print(f"📊 GPU metrics exported to {path}")


def main():
    """CLI interface for GPU monitoring."""
    import argparse
    parser = argparse.ArgumentParser(description="AMD GPU Monitor")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--output", help="CSV output path")
    args = parser.parse_args()

    monitor = GPUMonitor(args.device)
    print(f"🖥  Monitoring GPU {args.device} (interval: {args.interval}s, duration: {args.duration}s)")
    monitor.start(args.interval)
    time.sleep(args.duration)
    stats = monitor.stop()

    print(f"\n📊 Summary:")
    print(f"  Avg Utilization: {stats['util_avg']:.1f}%")
    print(f"  Avg Power:       {stats['power_avg']:.0f} W")
    print(f"  Max Temperature: {stats['temp_max']:.0f}°C")
    print(f"  Avg Memory:      {stats.get('mem_avg_mb', 0):.0f} MB")

    if args.output:
        monitor.export_csv(args.output)


if __name__ == "__main__":
    main()
