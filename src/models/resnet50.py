"""ResNet-50 model for MLPerf inference on ROCm."""

import torch
import torch.nn as nn
from torchvision import models


def load_resnet50(device: torch.device, precision: str = "fp32") -> nn.Module:
    """Load ResNet-50 with optional FP16/INT8 quantization."""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model = model.to(device)

    if precision == "fp16":
        model = model.half()
    elif precision == "int8":
        # PT2E dynamic quantization for ROCm
        try:
            from torch.ao.quantization.quantizer.xpu_inductor_quantizer import XPUInductorQuantizer
            model = torch.ao.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)
        except ImportError:
            model = model.half()  # Fallback to FP16

    model.eval()
    return model


class ResNet50Model:
    """ResNet-50 wrapper with pre/post processing."""

    def __init__(self, model: nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.input_shape = (1, 3, 224, 224)

    def preprocess(self, image):
        """Preprocess image to model input tensor."""
        import numpy as np
        from torchvision import transforms

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        if isinstance(image, np.ndarray):
            from PIL import Image
            image = Image.fromarray(image)

        return transform(image).unsqueeze(0).to(self.device)

    def postprocess(self, output):
        """Convert model output to predicted class."""
        probs = torch.softmax(output, dim=1)
        top5 = torch.topk(probs, 5)
        return {
            "class_id": top5.indices[0][0].item(),
            "confidence": top5.values[0][0].item(),
            "top5_classes": top5.indices[0].tolist(),
            "top5_confidences": top5.values[0].tolist(),
        }

    @torch.inference_mode()
    def infer(self, batch):
        """Run inference on batch."""
        return self.model(batch)
