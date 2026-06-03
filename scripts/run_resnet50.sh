#!/bin/bash
set -euo pipefail

echo "🚀 Running ResNet-50 MLPerf Benchmark (Offline)"
echo "================================================"

python -m src.benchmark \
    --config configs/resnet50_offline.yml \
    --output results/resnet50_offline.json \
    "$@"
