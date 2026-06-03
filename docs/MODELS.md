# Supported Models

## ResNet-50
- **Task**: Image classification
- **Parameters**: 25.6M
- **Input**: 224×224×3
- **Output**: 1000 class probabilities
- **FP32 TFLOPS**: ~4.1
- **FP16 TFLOPS**: ~8.2

## BERT-Large
- **Task**: Question answering
- **Parameters**: 340M
- **Input**: Token sequences (max 384)
- **Output**: Start/end logits
- **FP32 TFLOPS**: ~150
- **FP16 TFLOPS**: ~300

## DLRM
- **Task**: Click-through rate prediction
- **Parameters**: ~500M (varies with embedding size)
- **Input**: Dense features (13) + Sparse features (26)
- **Output**: Click probability [0,1]
- **FP32 TFLOPS**: ~0.5

## 3D-UNet
- **Task**: Medical image segmentation
- **Parameters**: ~30M
- **Input**: 128×128×128
- **Output**: 128×128×128 segmentation mask
- **FP32 TFLOPS**: ~20

## Adding New Models

1. Create `src/models/{model_name}.py`
2. Implement `load_{model_name}(device, precision)` function
3. Add preprocessing/postprocessing methods
4. Update `src/benchmark.py` model loading logic
5. Add YAML config in `configs/`
