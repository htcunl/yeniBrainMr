"""
2. Grup metrikleri: Parameters (M), Model size (MB), GFLOPs, GPU memory.
Training time ve Inference time script seviyesinde ölçülür.
"""
from __future__ import annotations

import io
from contextlib import redirect_stdout, redirect_stderr
from typing import Optional

import tensorflow as tf


def parameters_millions(model: tf.keras.Model) -> float:
    """Model parametre sayısını milyon (M) cinsinden döndürür."""
    return model.count_params() / 1e6


def model_size_mb(model: tf.keras.Model, bytes_per_param: int = 4) -> float:
    """Model boyutunu MB cinsinden döndürür (float32 varsayımı: 4 byte/param)."""
    return (model.count_params() * bytes_per_param) / (1024 * 1024)


def get_gflops(model: tf.keras.Model, input_shape: Optional[tuple] = None) -> Optional[float]:
    """
    Model için GFLOPs (milyar floating point operation) tahmini.
    input_shape: (batch_size, H, W, C) - verilmezse model.input_shape kullanılır.
    """
    if input_shape is None:
        input_shape = (1,) + tuple(model.input_shape[1:])
    try:
        # TF2 profiler ile FLOPs sayımı (rapor konsola yazılmasın diye stdout gizlenir)
        forward = tf.function(
            model.call,
            input_signature=[tf.TensorSpec(shape=input_shape, dtype=model.input.dtype)],
        )
        concrete = forward.get_concrete_function()
        graph = concrete.graph
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            flops = tf.compat.v1.profiler.profile(
                graph,
                options=tf.compat.v1.profiler.ProfileOptionBuilder.float_operation(),
            )
        total_float_ops = flops.total_float_ops
        return total_float_ops / 1e9  # GFLOPs
    except Exception:
        return None


def get_gpu_memory_mb() -> Optional[float]:
    """Mevcut GPU bellek kullanımını MB cinsinden döndürür (varsa)."""
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        return None
    try:
        # TF 2.x memory info (TF 2.1+)
        if hasattr(tf.config.experimental, "get_memory_info"):
            info = tf.config.experimental.get_memory_info("GPU:0")
            # 'current' veya 'peak' (TF sürümüne göre)
            used = info.get("current", info.get("peak", 0))
            return used / (1024 * 1024)
    except Exception:
        pass
    return None


def get_gpu_memory_usage_mb() -> Optional[float]:
    """
    Eğitim/inference sırasında kullanılan GPU bellek (MB).
    Çağrıldığı anda mevcut kullanımı döndürür.
    """
    return get_gpu_memory_mb()


def collect_model_metrics(
    model: tf.keras.Model,
    input_shape: Optional[tuple] = None,
    training_time_sec: Optional[float] = None,
    inference_time_sec: Optional[float] = None,
    num_inference_samples: Optional[int] = None,
    gpu_memory_mb: Optional[float] = None,
) -> dict:
    """
    2. Grup metriklerini toplar.
    training_time_sec: Toplam eğitim süresi (saniye).
    inference_time_sec: Toplam inference süresi (saniye).
    num_inference_samples: Inference yapılan örnek sayısı (inference time/sample için).
    """
    if input_shape is None:
        input_shape = (1,) + tuple(model.input_shape[1:])

    out = {
        "parameters_M": round(parameters_millions(model), 4),
        "model_size_MB": round(model_size_mb(model), 4),
        "GFLOPs": None,
        "training_time_sec": training_time_sec,
        "training_time_min": round(training_time_sec / 60, 2) if training_time_sec is not None else None,
        "inference_time_sec": inference_time_sec,
        "inference_time_per_sample_ms": None,
        "GPU_memory_MB": gpu_memory_mb,
    }
    gflops = get_gflops(model, input_shape)
    if gflops is not None:
        out["GFLOPs"] = round(gflops, 4)
    if inference_time_sec is not None and num_inference_samples is not None and num_inference_samples > 0:
        out["inference_time_per_sample_ms"] = round(1000.0 * inference_time_sec / num_inference_samples, 2)
    return out


