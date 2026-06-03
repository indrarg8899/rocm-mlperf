"""
MLPerf LoadGen integration wrapper
SUT (System Under Test) and QSL (Query Sample Library) implementations
"""

import time
import threading
from typing import List, Any, Callable

import numpy as np
import torch

try:
    import mlperf_loadgen as lg
    HAS_LOADGEN = True
except ImportError:
    HAS_LOADGEN = False


class QuerySample:
    """Represents a single query sample."""
    def __init__(self, index: int, data: Any):
        self.index = index
        self.data = data


class MLPerfQSL:
    """Query Sample Library - manages dataset for LoadGen."""

    def __init__(self, dataset: Any, total_count: int = 50000):
        self.dataset = dataset
        self.total_count = min(total_count, len(dataset) if hasattr(dataset, '__len__') else total_count)

        if HAS_LOADGEN:
            self.qsl = lg.ConstructQSL(
                self.total_count,
                min(1024, self.total_count),
                self.load_samples_to_ram,
                self.unload_samples_from_ram,
            )
        else:
            self.qsl = None
            self._loaded = {}

    def load_samples_to_ram(self, sample_indices: List[int]):
        """Load samples into memory."""
        for idx in sample_indices:
            if idx < len(self.dataset):
                self._loaded[idx] = self.dataset[idx]

    def unload_samples_from_ram(self, sample_indices: List[int]):
        """Unload samples from memory."""
        for idx in sample_indices:
            self._loaded.pop(idx, None)

    def get_samples(self, indices: List[int]) -> List[QuerySample]:
        """Get samples by index."""
        samples = []
        for idx in indices:
            if idx < len(self.dataset):
                data = self._loaded.get(idx, self.dataset[idx])
                samples.append(QuerySample(idx, data))
        return samples

    def __len__(self):
        return self.total_count


class MLPerfSUT:
    """System Under Test - runs inference for LoadGen."""

    def __init__(self, model: torch.nn.Module, config: Any, device: torch.device):
        self.model = model
        self.config = config
        self.device = device
        self.model.eval()
        self._process_fn = self._get_process_fn()

        if HAS_LOADGEN:
            self.sut = lg.ConstructSUT(
                self.issue_queries,
                self.flush_queries,
                self.process_latencies,
            )
        else:
            self.sut = None
        self._latencies = []
        self._lock = threading.Lock()

    def _get_process_fn(self) -> Callable:
        """Get model-specific processing function."""
        model_name = self.config.model.lower()
        if model_name == "resnet50":
            return self._process_resnet
        elif model_name == "bert":
            return self._process_bert
        elif model_name == "dlrm":
            return self._process_dlrm
        else:
            return self._process_generic

    def issue_queries(self, query_samples: List[Any]):
        """Process batch of queries."""
        if not query_samples:
            return

        start = time.time()

        # Batch inference
        batch_size = self.config.batch_size
        for i in range(0, len(query_samples), batch_size):
            batch = query_samples[i:i + batch_size]
            results = self._process_batch(batch)

            # Send responses to LoadGen
            if HAS_LOADGEN:
                responses = []
                for sample, result in zip(batch, results):
                    response = lg.QuerySampleResponse(
                        sample.id,
                        result.ctypes.data,
                        result.nbytes,
                    )
                    responses.append(response)
                lg.QuerySamplesComplete(responses)

        elapsed = time.time() - start
        with self._lock:
            self._latencies.append(elapsed * 1000)  # ms

    def _process_batch(self, batch: List[Any]) -> List[np.ndarray]:
        """Run inference on a batch."""
        with torch.no_grad():
            return self._process_fn(batch)

    def _process_resnet(self, batch: List[Any]) -> List[np.ndarray]:
        """ResNet-50 inference."""
        images = torch.stack([self._preprocess_image(s.data) for s in batch])
        images = images.to(self.device)
        outputs = self.model(images)
        probs = torch.softmax(outputs, dim=1)
        return [probs[i].cpu().numpy() for i in range(len(batch))]

    def _process_bert(self, batch: List[Any]) -> List[np.ndarray]:
        """BERT inference."""
        input_ids = torch.stack([s.data["input_ids"] for s in batch]).to(self.device)
        attention_mask = torch.stack([s.data["attention_mask"] for s in batch]).to(self.device)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return [outputs.logits[i].cpu().numpy() for i in range(len(batch))]

    def _process_dlrm(self, batch: List[Any]) -> List[np.ndarray]:
        """DLRM inference."""
        dense = torch.stack([s.data["dense"] for s in batch]).to(self.device)
        sparse = torch.stack([s.data["sparse"] for s in batch]).to(self.device)
        outputs = self.model(dense, sparse)
        return [outputs[i].cpu().numpy() for i in range(len(batch))]

    def _process_generic(self, batch: List[Any]) -> List[np.ndarray]:
        """Generic inference."""
        inputs = torch.stack([torch.tensor(s.data) for s in batch]).to(self.device)
        outputs = self.model(inputs)
        return [outputs[i].cpu().numpy() for i in range(len(batch))]

    def _preprocess_image(self, image: Any) -> torch.Tensor:
        """Preprocess image for ResNet-50."""
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
        return transform(image)

    def flush_queries(self):
        """Flush pending queries (no-op for synchronous processing)."""
        pass

    def process_latencies(self, latencies_ns: List[int]):
        """Process latency measurements from LoadGen."""
        with self._lock:
            self._latencies = [ns / 1e6 for ns in latencies_ns]  # Convert to ms

    def get_latencies(self) -> List[float]:
        """Get collected latencies in ms."""
        with self._lock:
            return self._latencies.copy()
