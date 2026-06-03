# Benchmark Methodology

## Overview

This document describes the benchmark methodology used for MLPerf Inference workloads on AMD MI300X accelerators.

## Test Environment

- **Hardware**: AMD Instinct MI300X 192GB HBM3
- **OS**: Ubuntu 22.04 LTS
- **ROCm**: 6.3.0
- **PyTorch**: 2.1+ (ROCm build)
- **MLPerf LoadGen**: v4.1

## Workloads

### ResNet-50 (Image Classification)
- **Dataset**: ImageNet validation (50,000 images)
- **Preprocessing**: Resize(256) → CenterCrop(224) → Normalize
- **Metric**: Top-1 accuracy ≥ 76.46%
- **Scenarios**: Offline (max throughput), Server (latency-optimized)

### BERT-Large (Question Answering)
- **Dataset**: SQuAD v1.1 validation (10,833 questions)
- **Preprocessing**: Tokenize with WordPiece
- **Metric**: F1 score ≥ 90.87%
- **Scenarios**: Offline, Server

### DLRM (Recommendation)
- **Dataset**: Criteo Terabyte
- **Preprocessing**: Feature engineering, embedding lookup
- **Metric**: AUC ≥ 80.25%

## Execution

1. **Warmup**: 10 iterations to stabilize clock speeds
2. **Profiling**: Optional rocprof integration for CU analysis
3. **Measurement**: Official MLPerf LoadGen with all required queries
4. **Validation**: Results validated against official MLPerf rules

## Reproducibility

All results are reproducible using:
```bash
python -m src.benchmark --config configs/resnet50_offline.yml --deterministic
```
