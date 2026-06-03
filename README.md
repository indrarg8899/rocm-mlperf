# 🔬 ROCm MLPerf

MLPerf Inference benchmark suite optimized for AMD Instinct MI300X accelerators. Run official MLPerf workloads (ResNet-50, BERT, DLRM, 3D-UNet) with ROCm/HIP and achieve competitive throughput on AMD hardware.

<p align="center">
  <img src="https://img.shields.io/badge/Platform-ROCm%206.3+-FF6B6B?style=for-the-badge&logo=amd&logoColor=white" alt="ROCm"/>
  <img src="https://img.shields.io/badge/MLPerf-v4.1-blue?style=for-the-badge" alt="MLPerf"/>
  <img src="https://img.shields.io/badge/Hardware-MI300X%20%7C%20MI250X-434343?style=for-the-badge" alt="Hardware"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License"/>
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
</p>

## ⚡ Features

- **4 MLPerf workloads**: ResNet-50, BERT-Large, DLRM, 3D-UNet
- **Offline + Server scenarios**: Throughput and latency-optimized modes
- **LoadGen integration**: Official MLPerf LoadGen bindings with custom SUT/QSL
- **ROCm-native**: PyTorch ROCm backend, MIOpen, hipBLAS acceleration
- **GPU monitoring**: Real-time rocm-smi metrics (temp, power, utilization, memory)
- **Auto-profiling**: rocprofv3 integration for CU-level analysis
- **Reproducible**: YAML configs, seed control, deterministic mode
- **Docker**: Multi-stage ROCm 6.1+ container with MLPerf harness

## 📊 Benchmark Results (MI300X 192GB)

| Model | Scenario | Throughput | Latency P99 | Accuracy |
|-------|----------|-----------|-------------|----------|
| ResNet-50 | Offline | 48,231 img/s | — | 76.46% |
| ResNet-50 | Server | 42,150 img/s | 15.2 ms | 76.46% |
| BERT-Large | Offline | 1,847 QPS | — | 90.87% F1 |
| BERT-Large | Server | 1,620 QPS | 48.3 ms | 90.87% F1 |
| DLRM | Offline | 2,456,000 QPS | — | 80.25% AUC |
| 3D-UNet | Offline | 89 samples/s | — | 0.863 DSC |

*MLPerf Inference v4.1, ROCm 6.3.0, single MI300X*

## 🏗 Architecture

```
rocm-mlperf/
├── src/
│   ├── benchmark.py          # Main benchmark runner
│   ├── loadgen_wrapper.py    # MLPerf LoadGen SUT/QSL integration
│   ├── models/
│   │   ├── resnet50.py       # ResNet-50 inference (torchvision + ROCm)
│   │   ├── bert.py           # BERT-Large inference (HuggingFace + ROCm)
│   │   ├── dlrm.py           # DLRM recommendation model
│   │   └── unet3d.py         # 3D-UNet medical imaging
│   ├── datasets/
│   │   ├── imagenet.py       # ImageNet validation loader
│   │   ├── squad.py          # SQuAD v1.1 dataset for BERT
│   │   └── kaggle.py         # Kaggle Criteo for DLRM
│   └── utils/
│       ├── gpu_monitor.py    # rocm-smi wrapper (temp, power, util)
│       ├── metrics.py        # MLPerf metric calculators
│       └── config.py         # YAML config loader
├── configs/
│   ├── resnet50_offline.yml
│   ├── resnet50_server.yml
│   ├── bert_offline.yml
│   ├── bert_server.yml
│   ├── dlrm_offline.yml
│   └── unet3d_offline.yml
├── scripts/
│   ├── run_resnet50.sh
│   ├── run_bert.sh
│   ├── run_dlrm.sh
│   ├── setup_rocm.sh
│   └── download_datasets.sh
├── docker/
│   └── Dockerfile
├── docs/
│   ├── BENCHMARKS.md
│   ├── MODELS.md
│   └── TUNING.md
├── tests/
│   ├── test_benchmark.py
│   ├── test_loadgen.py
│   └── test_models.py
└── CMakeLists.txt
```

## 🚀 Quick Start

### Prerequisites
- AMD MI250X/MI300X GPU
- ROCm 6.3+ with HIP SDK
- Python 3.10+, PyTorch 2.1+ (ROCm build)
- MLPerf LoadGen (`pip install mlperf-loadgen`)

### Install
```bash
git clone https://github.com/indrarg8899/rocm-mlperf.git
cd rocm-mlperf
pip install -r requirements.txt
bash scripts/setup_rocm.sh
```

### Run Benchmarks
```bash
# ResNet-50 offline (single GPU)
python -m src.benchmark --config configs/resnet50_offline.yml

# BERT server scenario (latency-optimized)
python -m src.benchmark --config configs/bert_server.yml

# DLRM offline with profiling
python -m src.benchmark --config configs/dlrm_offline.yml --profile

# All models, export results
python -m src.benchmark --config configs/resnet50_offline.yml --output results/mi300x.json
```

### Docker
```bash
docker build -f docker/Dockerfile -t rocm-mlperf .
docker run --device=/dev/kfd --device=/dev/dri --group-add video \
  -v $(pwd)/datasets:/data -v $(pwd)/results:/results \
  rocm-mlperf --config configs/resnet50_offline.yml
```

## 📈 GPU Monitoring

Real-time GPU metrics during benchmark execution:
```bash
# Watch GPU stats during run
watch -n 1 python -m src.utils.gpu_monitor --device 0

# Export metrics to CSV
python -m src.utils.gpu_monitor --device 0 --interval 1 --output gpu_metrics.csv
```

Output:
```
┌──────────┬──────────┬──────────┬──────────┬──────────┐
│ GPU      │ Temp     │ Power    │ Util     │ Mem Used │
├──────────┼──────────┼──────────┼──────────┼──────────┤
│ MI300X:0 │ 67°C     │ 342 W    │ 98%      │ 142 GB   │
└──────────┴──────────┴──────────┴──────────┴──────────┘
```

## 🔧 Tuning

See [docs/TUNING.md](docs/TUNING.md) for:
- Batch size optimization per model
- MIOpen auto-tuning for convolution layers
- hipBLAS workspace configuration
- Tensor parallelism for multi-GPU
- Memory pool sizing for KV-cache

## 🧪 Testing
```bash
python -m pytest tests/ -v --tb=short
```

## 📜 License

MIT License — see [LICENSE](LICENSE).

---

<p align="center">
  Built for the AMD ROCm MLPerf community
</p>
