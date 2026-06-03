#!/bin/bash
set -euo pipefail

echo "🚀 ROCm MLPerf - ROCm Environment Setup"
echo "========================================"

# Check ROCm installation
if ! command -v rocm-smi &> /dev/null; then
    echo "❌ ROCm not found. Install ROCm 6.3+ first."
    exit 1
fi

echo "✅ ROCm found:"
rocm-smi --showproductname

# Check HIP
if ! command -v hipcc &> /dev/null; then
    echo "❌ HIP compiler not found."
    exit 1
fi

echo "✅ HIP compiler: $(hipcc --version | head -1)"

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Install MLPerf LoadGen
echo "📦 Installing MLPerf LoadGen..."
pip install mlperf-loadgen

# Verify PyTorch ROCm
echo "🔍 Verifying PyTorch ROCm..."
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'HIP version: {torch.version.hip}')
if torch.cuda.is_available():
    print(f'Device: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"

echo ""
echo "✅ Setup complete! Run benchmarks with:"
echo "   python -m src.benchmark --config configs/resnet50_offline.yml"
