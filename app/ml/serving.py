"""
Model Serving Optimization
ONNX conversion, quantization, and serving optimizations

Features:
- ONNX model conversion for faster inference
- INT8 quantization
- Model warming on startup
- Lazy loading
- GPU batch inference
- Model caching
- Input preprocessing optimization
"""

import logging
import time
from typing import Dict, List, Any, Optional
from pathlib import Path
import pickle

import numpy as np
import torch
import onnx
import onnxruntime as ort
from torch.quantization import quantize_dynamic

logger = logging.getLogger(__name__)


class ModelServingOptimizer:
    """
    Model serving optimization system

    Features:
    - ONNX conversion
    - Quantization
    - Model warming
    - Lazy loading
    """

    def __init__(
        self,
        cache_dir: str = "/tmp/model_cache",
        use_gpu: bool = False,
        quantize: bool = True
    ):
        """
        Initialize serving optimizer

        Args:
            cache_dir: Directory for cached models
            use_gpu: Use GPU for inference
            quantize: Enable quantization
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.quantize = quantize

        # Model cache
        self.loaded_models: Dict[str, Any] = {}
        self.onnx_sessions: Dict[str, ort.InferenceSession] = {}

        # Performance stats
        self.load_times: Dict[str, float] = {}
        self.inference_times: Dict[str, List[float]] = {}

        logger.info(
            f"Model Serving Optimizer initialized "
            f"(GPU: {self.use_gpu}, Quantize: {self.quantize})"
        )

    def convert_to_onnx(
        self,
        model: torch.nn.Module,
        model_name: str,
        input_shape: tuple,
        opset_version: int = 13
    ) -> str:
        """
        Convert PyTorch model to ONNX format

        Args:
            model: PyTorch model
            model_name: Model name
            input_shape: Input tensor shape
            opset_version: ONNX opset version

        Returns:
            Path to ONNX model
        """
        logger.info(f"Converting {model_name} to ONNX")

        start_time = time.time()

        # Set model to eval mode
        model.eval()

        # Create dummy input
        dummy_input = torch.randn(*input_shape)

        # Export to ONNX
        onnx_path = self.cache_dir / f"{model_name}.onnx"

        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )

        # Verify ONNX model
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)

        conversion_time = time.time() - start_time

        logger.info(f"ONNX conversion completed in {conversion_time:.2f}s")
        logger.info(f"Saved to: {onnx_path}")

        return str(onnx_path)

    def quantize_model(
        self,
        model: torch.nn.Module,
        model_name: str,
        dtype: torch.dtype = torch.qint8
    ) -> torch.nn.Module:
        """
        Quantize model to INT8

        Args:
            model: PyTorch model
            model_name: Model name
            dtype: Quantization dtype

        Returns:
            Quantized model
        """
        logger.info(f"Quantizing {model_name} to INT8")

        start_time = time.time()

        # Dynamic quantization (post-training)
        quantized_model = quantize_dynamic(
            model,
            {torch.nn.Linear, torch.nn.LSTM, torch.nn.GRU},
            dtype=dtype
        )

        quantization_time = time.time() - start_time

        # Calculate size reduction
        original_size = self._get_model_size(model)
        quantized_size = self._get_model_size(quantized_model)
        size_reduction = (1 - quantized_size / original_size) * 100

        logger.info(
            f"Quantization completed in {quantization_time:.2f}s "
            f"(size reduced by {size_reduction:.1f}%)"
        )

        # Save quantized model
        quantized_path = self.cache_dir / f"{model_name}_quantized.pt"
        torch.save(quantized_model.state_dict(), quantized_path)

        return quantized_model

    def load_onnx_model(
        self,
        onnx_path: str,
        model_name: str
    ) -> ort.InferenceSession:
        """
        Load ONNX model for inference

        Args:
            onnx_path: Path to ONNX model
            model_name: Model name

        Returns:
            ONNX Runtime inference session
        """
        if model_name in self.onnx_sessions:
            logger.info(f"Using cached ONNX session: {model_name}")
            return self.onnx_sessions[model_name]

        logger.info(f"Loading ONNX model: {onnx_path}")

        start_time = time.time()

        # Configure session options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Set execution providers
        if self.use_gpu:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        # Create inference session
        session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_options,
            providers=providers
        )

        load_time = time.time() - start_time
        self.load_times[model_name] = load_time

        # Cache session
        self.onnx_sessions[model_name] = session

        logger.info(f"ONNX model loaded in {load_time:.2f}s")

        return session

    def predict_onnx(
        self,
        session: ort.InferenceSession,
        input_data: np.ndarray
    ) -> np.ndarray:
        """
        Run inference with ONNX model

        Args:
            session: ONNX Runtime session
            input_data: Input data

        Returns:
            Predictions
        """
        start_time = time.time()

        # Get input name
        input_name = session.get_inputs()[0].name

        # Run inference
        outputs = session.run(None, {input_name: input_data})

        inference_time = time.time() - start_time

        # Track inference time
        model_name = "onnx_model"
        if model_name not in self.inference_times:
            self.inference_times[model_name] = []
        self.inference_times[model_name].append(inference_time)

        return outputs[0]

    def warm_up_model(
        self,
        session: ort.InferenceSession,
        input_shape: tuple,
        num_iterations: int = 10
    ):
        """
        Warm up model with dummy inputs

        Args:
            session: ONNX Runtime session
            input_shape: Input shape
            num_iterations: Number of warmup iterations
        """
        logger.info(f"Warming up model ({num_iterations} iterations)")

        dummy_input = np.random.randn(*input_shape).astype(np.float32)

        for i in range(num_iterations):
            self.predict_onnx(session, dummy_input)

        logger.info("Model warmup completed")

    def batch_predict(
        self,
        session: ort.InferenceSession,
        input_data_list: List[np.ndarray],
        batch_size: int = 32
    ) -> List[np.ndarray]:
        """
        Batch prediction for efficiency

        Args:
            session: ONNX Runtime session
            input_data_list: List of input arrays
            batch_size: Batch size

        Returns:
            List of predictions
        """
        logger.info(f"Batch prediction: {len(input_data_list)} samples")

        results = []

        for i in range(0, len(input_data_list), batch_size):
            batch = input_data_list[i:i + batch_size]
            batch_array = np.vstack(batch)

            batch_predictions = self.predict_onnx(session, batch_array)

            for pred in batch_predictions:
                results.append(pred)

        return results

    def optimize_preprocessing(
        self,
        raw_features: Dict[str, float],
        feature_names: List[str],
        normalization_params: Optional[Dict[str, tuple]] = None
    ) -> np.ndarray:
        """
        Optimized feature preprocessing

        Args:
            raw_features: Raw feature dictionary
            feature_names: Expected feature names
            normalization_params: Mean and std for each feature

        Returns:
            Preprocessed feature array
        """
        # Extract features in correct order
        features = np.array([raw_features.get(name, 0.0) for name in feature_names])

        # Normalize if params provided
        if normalization_params:
            for i, name in enumerate(feature_names):
                if name in normalization_params:
                    mean, std = normalization_params[name]
                    features[i] = (features[i] - mean) / (std + 1e-8)

        return features.reshape(1, -1).astype(np.float32)

    def get_performance_stats(self, model_name: str) -> Dict[str, Any]:
        """
        Get performance statistics

        Args:
            model_name: Model name

        Returns:
            Performance stats
        """
        stats = {}

        if model_name in self.load_times:
            stats["load_time_seconds"] = self.load_times[model_name]

        if model_name in self.inference_times:
            times = self.inference_times[model_name]
            stats["inference_stats"] = {
                "count": len(times),
                "mean_ms": np.mean(times) * 1000,
                "p50_ms": np.percentile(times, 50) * 1000,
                "p95_ms": np.percentile(times, 95) * 1000,
                "p99_ms": np.percentile(times, 99) * 1000,
                "min_ms": np.min(times) * 1000,
                "max_ms": np.max(times) * 1000
            }

        return stats

    def clear_cache(self, model_name: Optional[str] = None):
        """
        Clear model cache

        Args:
            model_name: Specific model to clear (None for all)
        """
        if model_name:
            if model_name in self.loaded_models:
                del self.loaded_models[model_name]
            if model_name in self.onnx_sessions:
                del self.onnx_sessions[model_name]
            logger.info(f"Cleared cache for {model_name}")
        else:
            self.loaded_models.clear()
            self.onnx_sessions.clear()
            logger.info("Cleared all cached models")

    def _get_model_size(self, model: torch.nn.Module) -> int:
        """Get model size in bytes"""
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()

        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        return param_size + buffer_size

    def benchmark_models(
        self,
        original_model: torch.nn.Module,
        onnx_session: ort.InferenceSession,
        input_shape: tuple,
        num_iterations: int = 100
    ) -> Dict[str, Any]:
        """
        Benchmark original vs optimized model

        Args:
            original_model: Original PyTorch model
            onnx_session: ONNX session
            input_shape: Input shape
            num_iterations: Number of benchmark iterations

        Returns:
            Benchmark results
        """
        logger.info(f"Benchmarking models ({num_iterations} iterations)")

        # Prepare input
        input_data = np.random.randn(*input_shape).astype(np.float32)
        input_tensor = torch.from_numpy(input_data)

        # Benchmark original model
        original_model.eval()
        pytorch_times = []

        with torch.no_grad():
            for _ in range(num_iterations):
                start = time.time()
                _ = original_model(input_tensor)
                pytorch_times.append(time.time() - start)

        # Benchmark ONNX model
        onnx_times = []

        for _ in range(num_iterations):
            start = time.time()
            self.predict_onnx(onnx_session, input_data)
            onnx_times.append(time.time() - start)

        # Calculate statistics
        results = {
            "pytorch": {
                "mean_ms": np.mean(pytorch_times) * 1000,
                "std_ms": np.std(pytorch_times) * 1000,
                "p50_ms": np.percentile(pytorch_times, 50) * 1000,
                "p95_ms": np.percentile(pytorch_times, 95) * 1000,
            },
            "onnx": {
                "mean_ms": np.mean(onnx_times) * 1000,
                "std_ms": np.std(onnx_times) * 1000,
                "p50_ms": np.percentile(onnx_times, 50) * 1000,
                "p95_ms": np.percentile(onnx_times, 95) * 1000,
            },
            "speedup": np.mean(pytorch_times) / np.mean(onnx_times)
        }

        logger.info(
            f"Benchmark complete: ONNX is {results['speedup']:.2f}x faster "
            f"({results['onnx']['mean_ms']:.2f}ms vs {results['pytorch']['mean_ms']:.2f}ms)"
        )

        return results
