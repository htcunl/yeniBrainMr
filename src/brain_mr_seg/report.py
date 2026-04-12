"""
Tüm metrikleri tek bir rapor dosyasında toplar.
1. Grup: Dice, IoU, Sensitivity, Specificity, Precision, F1, HD95, ASD, MCC
2. Grup: Parameters (M), Model size (MB), Training time, Inference time, GFLOPs, GPU memory
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


# Eksik değer gösterimi
NA = "—"


def _fmt(value: Any, decimals: int = 4, unit: str = "") -> str:
    """Sayıyı stringe çevirir; None/NaN ise NA."""
    if value is None:
        return NA
    try:
        if isinstance(value, float) and (value != value or abs(value) == float("inf")):
            return NA
    except (TypeError, ValueError):
        return NA
    if isinstance(value, float):
        s = f"{value:.{decimals}f}"
    else:
        s = str(value)
    return f"{s}{unit}" if unit else s


def _fmt_mean_std(mean: Any, std: Any, decimals: int = 4) -> str:
    if mean is None and std is None:
        return NA
    if std is None or (isinstance(std, float) and (std != std or std == 0)):
        return _fmt(mean, decimals)
    return f"{_fmt(mean, decimals)} ± {_fmt(std, decimals)}"


def write_full_metrics_report(
    path: Path,
    *,
    # 1. Grup (hepsi isteğe bağlı)
    dice_mean: Optional[float] = None,
    dice_std: Optional[float] = None,
    iou_mean: Optional[float] = None,
    iou_std: Optional[float] = None,
    sensitivity_mean: Optional[float] = None,
    sensitivity_std: Optional[float] = None,
    specificity_mean: Optional[float] = None,
    specificity_std: Optional[float] = None,
    precision_mean: Optional[float] = None,
    precision_std: Optional[float] = None,
    f1_mean: Optional[float] = None,
    f1_std: Optional[float] = None,
    hd95_mean: Optional[float] = None,
    hd95_std: Optional[float] = None,
    asd_mean: Optional[float] = None,
    asd_std: Optional[float] = None,
    mcc_mean: Optional[float] = None,
    mcc_std: Optional[float] = None,
    # 2. Grup
    parameters_M: Optional[float] = None,
    model_size_MB: Optional[float] = None,
    training_time_sec: Optional[float] = None,
    inference_time_sec: Optional[float] = None,
    inference_time_per_sample_ms: Optional[float] = None,
    gflops: Optional[float] = None,
    gpu_memory_MB: Optional[float] = None,
    # Ek bilgi
    num_test_samples: Optional[int] = None,
    checkpoint_path: Optional[str] = None,
) -> None:
    """
    Tüm metrikleri tek bir metin dosyasına yazar. Eksik alanlar "—" olur.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Grup metinleri
    g1_dice = _fmt_mean_std(dice_mean, dice_std)
    g1_iou = _fmt_mean_std(iou_mean, iou_std)
    g1_sensitivity = _fmt_mean_std(sensitivity_mean, sensitivity_std)
    g1_specificity = _fmt_mean_std(specificity_mean, specificity_std)
    g1_precision = _fmt_mean_std(precision_mean, precision_std)
    g1_f1 = _fmt_mean_std(f1_mean, f1_std)
    _hd95 = _fmt_mean_std(hd95_mean, hd95_std)
    g1_hd95 = f"{_hd95} (piksel)" if _hd95 != NA else NA
    _asd = _fmt_mean_std(asd_mean, asd_std)
    g1_asd = f"{_asd} (piksel)" if _asd != NA else NA
    g1_mcc = _fmt_mean_std(mcc_mean, mcc_std)

    # 2. Grup metinleri
    g2_params = _fmt(parameters_M, 4) if parameters_M is not None else NA
    g2_size = _fmt(model_size_MB, 4) if model_size_MB is not None else NA
    if training_time_sec is not None:
        g2_train = f"{training_time_sec:.2f} s ({training_time_sec/60:.2f} min)"
    else:
        g2_train = NA
    if inference_time_sec is not None:
        g2_inf = f"{inference_time_sec:.2f} s"
        if inference_time_per_sample_ms is not None:
            g2_inf += f" ({inference_time_per_sample_ms:.2f} ms/örnek)"
    else:
        g2_inf = NA
    g2_gflops = _fmt(gflops, 4) if gflops is not None else NA
    g2_gpu = _fmt(gpu_memory_MB, 2) if gpu_memory_MB is not None else NA

    header_extra = []
    if checkpoint_path:
        header_extra.append(f"  Checkpoint         : {checkpoint_path}")
    if num_test_samples is not None:
        header_extra.append(f"  Test örnek sayısı  : {num_test_samples}")
    if header_extra:
        header_extra.append("")

    lines = [
        "",
        "=" * 70,
        "           BEYİN MR SEGMENTASYONU — TÜM METRİKLER (TEK RAPOR)",
        "=" * 70,
        "",
        *header_extra,
        "----------------------------------------",
        "1. GRUP (Segmentasyon Metrikleri)",
        "----------------------------------------",
        f"  Dice              : {g1_dice}",
        f"  IoU               : {g1_iou}",
        f"  Sensitivity       : {g1_sensitivity}",
        f"  Specificity       : {g1_specificity}",
        f"  Precision         : {g1_precision}",
        f"  F1                : {g1_f1}",
        f"  HD95              : {g1_hd95}",
        f"  ASD               : {g1_asd}",
        f"  MCC               : {g1_mcc}",
        "",
        "----------------------------------------",
        "2. GRUP (Model / Performans)",
        "----------------------------------------",
        f"  Parameters (M)    : {g2_params}",
        f"  Model size (MB)   : {g2_size}",
        f"  Training time     : {g2_train}",
        f"  Inference time    : {g2_inf}",
        f"  GFLOPs            : {g2_gflops}",
        f"  GPU memory (MB)   : {g2_gpu}",
        "",
        "=" * 70,
    ]

    text = "\n".join(lines)
    path.write_text(text, encoding="utf-8")


def write_full_metrics_report_from_dict(path: Path, group1: dict, group2: dict, meta: Optional[dict] = None) -> None:
    """
    Sözlüklerden tam rapor yazar.
    group1: dice_mean, dice_std, iou_mean, ... (opsiyonel anahtarlar)
    group2: parameters_M, model_size_MB, training_time_sec, inference_time_sec,
            inference_time_per_sample_ms, gflops, gpu_memory_MB
    meta: num_test_samples, checkpoint_path
    """
    def g1(k_mean: str, k_std: str):
        return group1.get(k_mean), group1.get(k_std)

    write_full_metrics_report(
        path,
        dice_mean=g1("dice_mean", "dice_std")[0],
        dice_std=g1("dice_mean", "dice_std")[1],
        iou_mean=g1("iou_mean", "iou_std")[0],
        iou_std=g1("iou_mean", "iou_std")[1],
        sensitivity_mean=g1("sensitivity_mean", "sensitivity_std")[0],
        sensitivity_std=g1("sensitivity_mean", "sensitivity_std")[1],
        specificity_mean=g1("specificity_mean", "specificity_std")[0],
        specificity_std=g1("specificity_mean", "specificity_std")[1],
        precision_mean=g1("precision_mean", "precision_std")[0],
        precision_std=g1("precision_mean", "precision_std")[1],
        f1_mean=g1("f1_mean", "f1_std")[0],
        f1_std=g1("f1_mean", "f1_std")[1],
        hd95_mean=g1("hd95_mean", "hd95_std")[0],
        hd95_std=g1("hd95_mean", "hd95_std")[1],
        asd_mean=g1("asd_mean", "asd_std")[0],
        asd_std=g1("asd_mean", "asd_std")[1],
        mcc_mean=g1("mcc_mean", "mcc_std")[0],
        mcc_std=g1("mcc_mean", "mcc_std")[1],
        parameters_M=group2.get("parameters_M"),
        model_size_MB=group2.get("model_size_MB"),
        training_time_sec=group2.get("training_time_sec"),
        inference_time_sec=group2.get("inference_time_sec"),
        inference_time_per_sample_ms=group2.get("inference_time_per_sample_ms"),
        gflops=group2.get("gflops"),
        gpu_memory_MB=group2.get("gpu_memory_MB"),
        num_test_samples=(meta or {}).get("num_test_samples"),
        checkpoint_path=(meta or {}).get("checkpoint_path"),
    )


def save_training_metrics_json(path: Path, training_time_sec: float, parameters_M: float, model_size_MB: float,
                               gflops: Optional[float], gpu_memory_MB: Optional[float]) -> None:
    """Eğitim sonrası 2. grup değerlerini JSON olarak kaydeder; test script'i bunları rapora ekleyebilir."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "training_time_sec": training_time_sec,
        "parameters_M": parameters_M,
        "model_size_MB": model_size_MB,
        "gflops": gflops,
        "gpu_memory_MB": gpu_memory_MB,
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_training_metrics_json(path: Path) -> Optional[dict]:
    """Kaydedilmiş eğitim metriklerini okur."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


