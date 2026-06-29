#!/usr/bin/env python
"""
TensorFlow ile Eğitilen Modeli Test Et
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

# Matplotlib
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from brain_mr_seg.model import build_unet
from brain_mr_seg.dataset import create_dataset, load_image_mask_pair
from brain_mr_seg.losses import BCEDiceLoss
import time
from brain_mr_seg.metrics import (
    DiceCoefficient,
    IoUCoefficient,
    dice_coefficient,
    iou_coefficient,
    compute_dice_np,
    compute_iou_np,
    compute_sensitivity_np,
    compute_specificity_np,
    compute_precision_np,
    compute_f1_np,
    compute_hd95_np,
    compute_asd_np,
    compute_mcc_np,
)
from brain_mr_seg.splits import load_split
from brain_mr_seg.model_metrics import (
    parameters_millions,
    model_size_mb,
    get_gflops,
    get_gpu_memory_mb,
)
from brain_mr_seg.report import write_full_metrics_report, load_training_metrics_json

# viz_test_indices_20.json yoksa yedek (proje kökündeki JSON ile senkron tutun)
_FALLBACK_VIZ_INDICES_20 = [
    82, 101, 131, 132, 163, 181, 182, 199, 227, 231,
    238, 300, 370, 393, 436, 505, 542, 576, 601, 609,
]


def load_fixed_viz_indices(num_samples: int, project_root: Path, n_items: int) -> list[int]:
    """Modelden bağımsız sabit test indeksleri (70/10/20 ile 80/10/10 aynı görseller)."""
    path = project_root / "viz_test_indices_20.json"
    indices: list[int]
    if path.is_file():
        data = json.loads(path.read_text(encoding="utf-8"))
        indices = [int(i) for i in data["indices"]]
    else:
        indices = list(_FALLBACK_VIZ_INDICES_20)
    if num_samples <= len(indices):
        out = indices[:num_samples]
    else:
        rng = np.random.default_rng(42)
        extra = num_samples - len(indices)
        pool = [i for i in range(n_items) if i not in indices]
        pick = rng.choice(pool, size=min(extra, len(pool)), replace=False).tolist()
        out = indices + sorted(pick)
    return [i for i in out if 0 <= i < n_items]


def evaluate_model(model, dataset, num_samples: int) -> dict:
    """Model değerlendirmesi - 1. Grup: Dice, IoU, Sensitivity, Specificity, Precision, F1, HD95, ASD, MCC + inference time."""
    all_dice = []
    all_iou = []
    all_sensitivity = []
    all_specificity = []
    all_precision = []
    all_f1 = []
    all_hd95 = []
    all_asd = []
    all_mcc = []

    print("\nTest ediliyor (1. Grup + inference time)...")
    t0 = time.perf_counter()
    for images, masks in tqdm(dataset, desc="Testing"):
        preds = model(images, training=False)
        preds_binary = tf.cast(tf.nn.sigmoid(preds) > 0.5, tf.float32)

        batch_size = tf.shape(images)[0]
        for i in range(batch_size):
            pred_i = preds_binary[i]
            mask_i = masks[i]
            m = mask_i.numpy()
            p = pred_i.numpy()

            all_dice.append(compute_dice_np(m, p))
            all_iou.append(compute_iou_np(m, p))
            all_sensitivity.append(compute_sensitivity_np(m, p))
            all_specificity.append(compute_specificity_np(m, p))
            all_precision.append(compute_precision_np(m, p))
            all_f1.append(compute_f1_np(m, p))
            hd = compute_hd95_np(m, p)
            all_hd95.append(hd if np.isfinite(hd) else np.nan)
            asd_val = compute_asd_np(m, p)
            all_asd.append(asd_val if np.isfinite(asd_val) else np.nan)
            all_mcc.append(compute_mcc_np(m, p))

            if len(all_dice) >= num_samples:
                break

        if len(all_dice) >= num_samples:
            break
    inference_elapsed = time.perf_counter() - t0

    n = num_samples
    dice_arr = np.array(all_dice[:n])
    iou_arr = np.array(all_iou[:n])
    sens_arr = np.array(all_sensitivity[:n])
    spec_arr = np.array(all_specificity[:n])
    prec_arr = np.array(all_precision[:n])
    f1_arr = np.array(all_f1[:n])
    hd95_arr = np.array(all_hd95[:n])
    hd95_finite = hd95_arr[np.isfinite(hd95_arr)]
    asd_arr = np.array(all_asd[:n])
    asd_finite = asd_arr[np.isfinite(asd_arr)]
    mcc_arr = np.array(all_mcc[:n])

    return {
        "dice_mean": float(dice_arr.mean()),
        "dice_std": float(dice_arr.std()),
        "iou_mean": float(iou_arr.mean()),
        "iou_std": float(iou_arr.std()),
        "sensitivity_mean": float(sens_arr.mean()),
        "sensitivity_std": float(sens_arr.std()),
        "specificity_mean": float(spec_arr.mean()),
        "specificity_std": float(spec_arr.std()),
        "precision_mean": float(prec_arr.mean()),
        "precision_std": float(prec_arr.std()),
        "f1_mean": float(f1_arr.mean()),
        "f1_std": float(f1_arr.std()),
        "hd95_mean": float(hd95_finite.mean()) if len(hd95_finite) > 0 else float("nan"),
        "hd95_std": float(hd95_finite.std()) if len(hd95_finite) > 1 else 0.0,
        "asd_mean": float(asd_finite.mean()) if len(asd_finite) > 0 else float("nan"),
        "asd_std": float(asd_finite.std()) if len(asd_finite) > 1 else 0.0,
        "mcc_mean": float(mcc_arr.mean()),
        "mcc_std": float(mcc_arr.std()),
        "inference_time_sec": inference_elapsed,
        "inference_time_per_sample_ms": 1000.0 * inference_elapsed / n if n > 0 else 0.0,
        "num_test_samples": n,
        "all_dice": all_dice[:n],
        "all_iou": all_iou[:n],
        "all_sensitivity": all_sensitivity[:n],
        "all_specificity": all_specificity[:n],
        "all_precision": all_precision[:n],
        "all_f1": all_f1[:n],
        "all_hd95": all_hd95[:n],
        "all_asd": all_asd[:n],
        "all_mcc": all_mcc[:n],
    }


def plot_metrics_histogram(results: dict, output_dir: Path):
    """Dice ve IoU skorlarının dağılım histogramını çiz"""
    if not HAS_MATPLOTLIB:
        print("UYARI: matplotlib yüklü değil, grafik atlanıyor.")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Dice Histogram
    axes[0].hist(results['all_dice'], bins=20, color='#3498db', edgecolor='white', alpha=0.8)
    axes[0].axvline(results['dice_mean'], color='red', linestyle='--', linewidth=2, 
                    label=f"Ortalama: {results['dice_mean']:.4f}")
    axes[0].set_xlabel('Dice Score', fontsize=12)
    axes[0].set_ylabel('Frekans', fontsize=12)
    axes[0].set_title('Dice Score Dağılımı', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # IoU Histogram
    axes[1].hist(results['all_iou'], bins=20, color='#2ecc71', edgecolor='white', alpha=0.8)
    axes[1].axvline(results['iou_mean'], color='red', linestyle='--', linewidth=2,
                    label=f"Ortalama: {results['iou_mean']:.4f}")
    axes[1].set_xlabel('IoU Score', fontsize=12)
    axes[1].set_ylabel('Frekans', fontsize=12)
    axes[1].set_title('IoU Score Dağılımı', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "metrics_histogram.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Histogram grafiği kaydedildi: {output_dir / 'metrics_histogram.png'}")


def plot_metrics_boxplot(results: dict, output_dir: Path):
    """Dice ve IoU skorlarının box plot grafiğini çiz"""
    if not HAS_MATPLOTLIB:
        print("UYARI: matplotlib yüklü değil, grafik atlanıyor.")
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    data = [results['all_dice'], results['all_iou']]
    bp = ax.boxplot(data, labels=['Dice Score', 'IoU Score'], patch_artist=True)
    
    colors = ['#3498db', '#2ecc71']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Skor', fontsize=12)
    ax.set_title('Test Metrikleri Box Plot', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Ortalama değerleri ekle
    means = [results['dice_mean'], results['iou_mean']]
    ax.scatter([1, 2], means, color='red', marker='D', s=50, zorder=5, label='Ortalama')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / "metrics_boxplot.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Box plot grafiği kaydedildi: {output_dir / 'metrics_boxplot.png'}")


def plot_metrics_bar(results: dict, output_dir: Path):
    """1. Grup metriklerinin bar chart grafiğini çiz (Dice, IoU, Sensitivity, Specificity, Precision, F1, MCC; HD95 ayrı)"""
    if not HAS_MATPLOTLIB:
        print("UYARI: matplotlib yüklü değil, grafik atlanıyor.")
        return

    fig, ax = plt.subplots(figsize=(14, 6))

    metrics = ['Dice', 'IoU', 'Sensitivity', 'Specificity', 'Precision', 'F1', 'MCC']
    means = [
        results['dice_mean'],
        results['iou_mean'],
        results['sensitivity_mean'],
        results['specificity_mean'],
        results['precision_mean'],
        results['f1_mean'],
        results['mcc_mean'],
    ]
    stds = [
        results['dice_std'],
        results['iou_std'],
        results['sensitivity_std'],
        results['specificity_std'],
        results['precision_std'],
        results['f1_std'],
        results['mcc_std'],
    ]
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#f39c12', '#34495e', '#1abc9c']

    bars = ax.bar(metrics, means, yerr=stds, capsize=6, color=colors,
                  edgecolor='white', alpha=0.8, linewidth=2)

    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + std + 0.02,
                f'{mean:.4f} ± {std:.4f}', ha='center', va='bottom',
                fontsize=9, fontweight='bold')

    ax.set_ylabel('Skor', fontsize=12)
    ax.set_title('1. Grup Metrikleri — Test Sonuçları', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / "metrics_bar.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Bar chart grafiği kaydedildi: {output_dir / 'metrics_bar.png'}")

    # HD95 ayrı göster (farklı ölçek - piksel)
    if np.isfinite(results.get('hd95_mean', float('nan'))):
        fig2, ax2 = plt.subplots(figsize=(4, 5))
        ax2.bar(['HD95'], [results['hd95_mean']], yerr=[results['hd95_std']], capsize=8,
                color='#1abc9c', edgecolor='white', alpha=0.8)
        ax2.set_ylabel('Mesafe (piksel)', fontsize=12)
        ax2.set_title('HD95 (95. Yüzdelik Hausdorff Mesafesi)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(output_dir / "metrics_hd95.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"HD95 grafiği kaydedildi: {output_dir / 'metrics_hd95.png'}")


def visualize_predictions(
    model,
    data_dir: str,
    items: list,
    target_size: tuple,
    output_dir: Path,
    num_samples: int = 10,
    all_dice_scores: list = None,
    all_iou_scores: list = None,
    vis_mode: str = "fixed",
    project_root: Path | None = None,
):
    """Tahminleri görselleştir.

    vis_mode:
      - fixed: viz_test_indices_20.json (modelden bağımsız); 70/10/20 ve 80/10/10 aynı örnekler.
      - balanced: mevcut Dice sıralamasına göre quintile örnekleme (model bazlı, karşılaştırmada farklı örnekler).
    """
    if not HAS_MATPLOTLIB:
        print("UYARI: matplotlib yüklü değil, görselleştirme atlanıyor.")
        return
    
    vis_dir = output_dir / "predictions"
    vis_dir.mkdir(exist_ok=True)
    # Eski tahmin görsellerini sil (yeni 20 görsel ile değiştirilecek)
    for old in vis_dir.glob("*.png"):
        old.unlink()

    if vis_mode == "fixed":
        root = project_root or Path(__file__).resolve().parent.parent
        indices_list = load_fixed_viz_indices(num_samples, root, len(items))
        indices = np.array(indices_list, dtype=np.int64)
        print(f"   Sabit örnekleme (modelden bağımsız): {len(indices)} örnek (viz_test_indices_20.json)")
    # Dengeli örnekleme: farklı performans seviyelerinden örnekler seç (model tahminine bağlı)
    elif all_dice_scores is not None and len(all_dice_scores) == len(items):
        np.random.seed(42)
        dice_arr = np.array(all_dice_scores)
        
        # Skorlara göre sırala
        sorted_indices = np.argsort(dice_arr)
        n = len(sorted_indices)
        
        # 5 grup: çok kötü, kötü, orta, iyi, çok iyi
        samples_per_group = num_samples // 5
        extra = num_samples % 5
        
        selected_indices = []
        
        # Çok kötü (en düşük %20)
        low_range = sorted_indices[:n//5]
        selected_indices.extend(np.random.choice(low_range, min(samples_per_group, len(low_range)), replace=False))
        
        # Kötü (%20-40)
        low_mid_range = sorted_indices[n//5:2*n//5]
        selected_indices.extend(np.random.choice(low_mid_range, min(samples_per_group, len(low_mid_range)), replace=False))
        
        # Orta (%40-60)
        mid_range = sorted_indices[2*n//5:3*n//5]
        selected_indices.extend(np.random.choice(mid_range, min(samples_per_group, len(mid_range)), replace=False))
        
        # İyi (%60-80)
        high_mid_range = sorted_indices[3*n//5:4*n//5]
        selected_indices.extend(np.random.choice(high_mid_range, min(samples_per_group, len(high_mid_range)), replace=False))
        
        # Çok iyi (en yüksek %20)
        high_range = sorted_indices[4*n//5:]
        selected_indices.extend(np.random.choice(high_range, min(samples_per_group + extra, len(high_range)), replace=False))
        
        indices = np.array(selected_indices)
        print(f"   Dengeli örnekleme: {len(indices)} örnek (farklı performans seviyelerinden)")
    else:
        np.random.seed(42)
        indices = np.random.choice(len(items), min(num_samples, len(items)), replace=False)

    indices = np.clip(indices, 0, len(items) - 1)
    
    for idx in indices:
        item = items[idx]
        img, mask = load_image_mask_pair(data_dir, item, target_size)
        
        # Predict
        img_batch = np.expand_dims(img, 0)
        pred = model(img_batch, training=False)
        pred = tf.nn.sigmoid(pred).numpy()[0]
        pred_binary = (pred > 0.5).astype(np.float32)
        
        # Tüm 9 metriği hesapla (Dice, IoU, Sensitivity, Specificity, Precision, F1, HD95, ASD, MCC)
        m = mask
        p = pred_binary
        dice = compute_dice_np(m, p)
        iou = compute_iou_np(m, p)
        sensitivity = compute_sensitivity_np(m, p)
        specificity = compute_specificity_np(m, p)
        precision = compute_precision_np(m, p)
        f1 = compute_f1_np(m, p)
        hd95 = compute_hd95_np(m, p)
        asd_val = compute_asd_np(m, p)
        mcc = compute_mcc_np(m, p)

        def _fmt_scalar(x: float, nd: int = 4) -> str:
            """Dice/IoU vb.: her zaman sayı veya nan/inf etiketi (görüntüde boş kalmasın)."""
            if np.isnan(x):
                return "nan"
            if np.isinf(x):
                return "inf"
            return f"{x:.{nd}f}"

        def _fmt_surface_dist(x: float) -> str:
            """HD95/ASD: inf = bir maske boş (sınır yok); nan = tanımsız."""
            if np.isnan(x):
                return "nan"
            if np.isposinf(x):
                return "∞ (boş)"
            if np.isneginf(x):
                return "-∞"
            return f"{x:.2f}"

        # Görselleştir: 4 panel + altta metrik kutusu (çok satır; tight_layout alt satırı kesebiliyordu)
        fig = plt.figure(figsize=(16, 6.8))
        gs = fig.add_gridspec(2, 4, height_ratios=[1.15, 0.72], hspace=0.36)
        axes = [fig.add_subplot(gs[0, i]) for i in range(4)]

        axes[0].imshow(img[:, :, 0], cmap="gray")
        axes[0].set_title("Giriş Görüntüsü", fontsize=11)
        axes[0].axis("off")

        axes[1].imshow(mask[:, :, 0], cmap="gray")
        axes[1].set_title("Ground Truth", fontsize=11)
        axes[1].axis("off")

        axes[2].imshow(pred[:, :, 0], cmap="gray")
        axes[2].set_title("Tahmin (Olasılık)", fontsize=11)
        axes[2].axis("off")

        overlay = np.zeros((*img.shape[:2], 3))
        overlay[..., 0] = pred_binary[:, :, 0]
        overlay[..., 1] = mask[:, :, 0]
        axes[3].imshow(img[:, :, 0], cmap="gray", alpha=0.7)
        axes[3].imshow(overlay, alpha=0.5)
        axes[3].set_title("Karşılaştırma (Yeşil: GT, Kırmızı: Tahmin)", fontsize=10)
        axes[3].axis("off")

        # Metrik kutusu (3 satır; monospace taşmayı azaltır)
        ax_txt = fig.add_subplot(gs[1, :])
        ax_txt.set_axis_off()
        hd95_s = _fmt_surface_dist(float(hd95))
        asd_s = _fmt_surface_dist(float(asd_val))
        metrics_text = (
            f"Dice: {_fmt_scalar(dice)}  |  IoU: {_fmt_scalar(iou)}  |  Sens: {_fmt_scalar(sensitivity)}  |  Spec: {_fmt_scalar(specificity)}\n"
            f"Prec: {_fmt_scalar(precision)}  |  F1: {_fmt_scalar(f1)}  |  MCC: {_fmt_scalar(mcc)}\n"
            f"HD95: {hd95_s} px  |  ASD: {asd_s} px"
        )
        ax_txt.text(
            0.5,
            0.5,
            metrics_text,
            transform=ax_txt.transAxes,
            fontsize=9,
            verticalalignment="center",
            horizontalalignment="center",
            family="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )
        # tight_layout alt metin kutusunu kırpar; manuel boşluk + pad_inches güvenli
        fig.subplots_adjust(left=0.03, right=0.97, top=0.96, bottom=0.08)
        plt.savefig(
            vis_dir / f"sample_{idx:04d}.png",
            dpi=150,
            bbox_inches="tight",
            pad_inches=0.35,
        )
        plt.close()
    
    print(f"{len(indices)} tahmin görselleştirmesi kaydedildi: {vis_dir}")


def main():
    parser = argparse.ArgumentParser(description="TensorFlow ile Beyin MR Segmentasyonu Test")
    parser.add_argument("--data-dir", type=str, required=True, help="Data klasörü yolu")
    parser.add_argument("--split-file", type=str, required=True, help="Split JSON dosyası")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint yolu")
    parser.add_argument("--output-dir", type=str, default="test_results", help="Çıktı klasörü")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch boyutu")
    parser.add_argument("--image-size", type=int, default=256, help="Görüntü boyutu")
    parser.add_argument("--visualize", action="store_true", help="Tahminleri görselleştir")
    parser.add_argument("--num-vis", type=int, default=20, help="Görselleştirilecek örnek sayısı (predictions klasörüne kaydedilir)")
    parser.add_argument(
        "--vis-mode",
        type=str,
        choices=("fixed", "balanced"),
        default="fixed",
        help="fixed: viz_test_indices_20.json (tüm modellerde aynı 20 örnek); balanced: Dice quintile (modele göre değişir).",
    )
    args = parser.parse_args()

    # GPU info
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"GPU kullanılıyor: {len(gpus)} GPU")
    else:
        print("UYARI: GPU bulunamadı, CPU kullanılıyor!")

    # Çıktı klasörü
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Split dosyasını yükle
    print(f"\nSplit dosyası yükleniyor: {args.split_file}")
    splits = load_split(args.split_file)
    test_items = splits["test"]
    print(f"   Test seti: {len(test_items)} örnek")

    # Model yükle
    print(f"\nModel yükleniyor: {args.checkpoint}")
    model = keras.models.load_model(
        args.checkpoint,
        custom_objects={
            "BCEDiceLoss": BCEDiceLoss,
            "DiceCoefficient": DiceCoefficient,
            "IoUCoefficient": IoUCoefficient,
        }
    )
    print("   Model yüklendi!")

    # 2. Grup metrikleri (params, size, GFLOPs, GPU memory)
    target_size = (args.image_size, args.image_size)
    params_M = parameters_millions(model)
    size_mb = model_size_mb(model)
    gflops = get_gflops(model, (1, target_size[0], target_size[1], 1))
    gpu_mem = get_gpu_memory_mb()
    print("\n--- 2. GRUP (Model / Performans) ---")
    print(f"   Parameters (M): {params_M:.4f}")
    print(f"   Model size (MB): {size_mb:.4f}")
    if gflops is not None:
        print(f"   GFLOPs: {gflops:.4f}")
    if gpu_mem is not None:
        print(f"   GPU memory (MB): {gpu_mem:.2f}")

    # Test dataset
    target_size = (args.image_size, args.image_size)
    test_dataset = create_dataset(
        data_dir=args.data_dir,
        items=test_items,
        batch_size=args.batch_size,
        target_size=target_size,
        shuffle=False,
        augment=False,
    )

    # Değerlendirme
    results = evaluate_model(model, test_dataset, len(test_items))

    # Sonuçları yazdır - 1. Grup metrikleri
    print("\n" + "=" * 60)
    print("TEST SONUCLARI — 1. GRUP METRİKLERİ")
    print("=" * 60)
    print(f"   Dice:        {results['dice_mean']:.4f} ± {results['dice_std']:.4f}")
    print(f"   IoU:         {results['iou_mean']:.4f} ± {results['iou_std']:.4f}")
    print(f"   Sensitivity: {results['sensitivity_mean']:.4f} ± {results['sensitivity_std']:.4f}")
    print(f"   Specificity: {results['specificity_mean']:.4f} ± {results['specificity_std']:.4f}")
    print(f"   Precision:   {results['precision_mean']:.4f} ± {results['precision_std']:.4f}")
    print(f"   F1:          {results['f1_mean']:.4f} ± {results['f1_std']:.4f}")
    hd95_str = f"{results['hd95_mean']:.2f} ± {results['hd95_std']:.2f}" if np.isfinite(results['hd95_mean']) else "N/A"
    print(f"   HD95:        {hd95_str} (piksel)")
    asd_str = f"{results['asd_mean']:.2f} ± {results['asd_std']:.2f}" if np.isfinite(results['asd_mean']) else "N/A"
    print(f"   ASD:         {asd_str} (piksel)")
    print(f"   MCC:         {results['mcc_mean']:.4f} ± {results['mcc_std']:.4f}")
    print("=" * 60)
    print("2. GRUP (Test sırasında)")
    print(f"   Inference time (toplam): {results['inference_time_sec']:.2f} s")
    print(f"   Inference time (örnek başına): {results['inference_time_per_sample_ms']:.2f} ms")
    print("=" * 60)

    # Eğitim metrikleri (training_metrics.json varsa - örn. outputs/ - eğitim süresi eklenir)
    checkpoint_path = Path(args.checkpoint)
    train_metrics_path = checkpoint_path.parent.parent / "training_metrics.json"
    train_metrics = load_training_metrics_json(train_metrics_path)
    training_time_sec = train_metrics.get("training_time_sec") if train_metrics else None

    # Tek rapor dosyası: tüm metrikler başlıklar altında
    report_path = output_dir / "full_metrics_report.txt"
    write_full_metrics_report(
        report_path,
        dice_mean=results["dice_mean"],
        dice_std=results["dice_std"],
        iou_mean=results["iou_mean"],
        iou_std=results["iou_std"],
        sensitivity_mean=results["sensitivity_mean"],
        sensitivity_std=results["sensitivity_std"],
        specificity_mean=results["specificity_mean"],
        specificity_std=results["specificity_std"],
        precision_mean=results["precision_mean"],
        precision_std=results["precision_std"],
        f1_mean=results["f1_mean"],
        f1_std=results["f1_std"],
        hd95_mean=results["hd95_mean"] if np.isfinite(results["hd95_mean"]) else None,
        hd95_std=results["hd95_std"],
        asd_mean=results["asd_mean"] if np.isfinite(results["asd_mean"]) else None,
        asd_std=results["asd_std"],
        mcc_mean=results["mcc_mean"],
        mcc_std=results["mcc_std"],
        parameters_M=params_M,
        model_size_MB=size_mb,
        training_time_sec=training_time_sec,
        inference_time_sec=results["inference_time_sec"],
        inference_time_per_sample_ms=results["inference_time_per_sample_ms"],
        gflops=gflops,
        gpu_memory_MB=gpu_mem,
        num_test_samples=len(test_items),
        checkpoint_path=str(args.checkpoint),
    )
    print(f"\nTüm metrikler raporu: {report_path}")

    # Eski formatta test_results.txt (isteğe bağlı tutuldu)
    results_file = output_dir / "test_results.txt"
    with open(results_file, "w", encoding="utf-8") as f:
        f.write("Beyin MR Segmentasyonu Test Sonuçları\n")
        f.write("=" * 50 + "\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Test örnekleri: {len(test_items)}\n\n")
        f.write("1. GRUP METRİKLERİ\n")
        f.write("-" * 50 + "\n")
        f.write(f"Dice:        {results['dice_mean']:.4f} ± {results['dice_std']:.4f}\n")
        f.write(f"IoU:         {results['iou_mean']:.4f} ± {results['iou_std']:.4f}\n")
        f.write(f"Sensitivity: {results['sensitivity_mean']:.4f} ± {results['sensitivity_std']:.4f}\n")
        f.write(f"Specificity: {results['specificity_mean']:.4f} ± {results['specificity_std']:.4f}\n")
        f.write(f"Precision:   {results['precision_mean']:.4f} ± {results['precision_std']:.4f}\n")
        f.write(f"F1:          {results['f1_mean']:.4f} ± {results['f1_std']:.4f}\n")
        f.write(f"HD95:        {hd95_str} (piksel)\n")
        f.write(f"ASD:         {asd_str} (piksel)\n")
        f.write(f"MCC:         {results['mcc_mean']:.4f} ± {results['mcc_std']:.4f}\n")
        f.write("\n2. GRUP METRİKLERİ\n")
        f.write("-" * 50 + "\n")
        f.write(f"Parameters (M):     {params_M:.4f}\n")
        f.write(f"Model size (MB):   {size_mb:.4f}\n")
        if gflops is not None:
            f.write(f"GFLOPs:            {gflops:.4f}\n")
        if gpu_mem is not None:
            f.write(f"GPU memory (MB):   {gpu_mem:.2f}\n")
        f.write(f"Inference time (s): {results['inference_time_sec']:.2f}\n")
        f.write(f"Inference time (ms/örnek): {results['inference_time_per_sample_ms']:.2f}\n")
    print(f"Sonuçlar (özet): {results_file}")

    # Grafikleri oluştur
    print("\nGrafikler oluşturuluyor...")
    plot_metrics_histogram(results, output_dir)
    plot_metrics_boxplot(results, output_dir)
    plot_metrics_bar(results, output_dir)

    # Tahmin görselleştirmesi
    print("\nTahminler görselleştiriliyor...")
    visualize_predictions(
        model,
        args.data_dir,
        test_items,
        target_size,
        output_dir,
        args.num_vis,
        all_dice_scores=results["all_dice"],
        all_iou_scores=results["all_iou"],
        vis_mode=args.vis_mode,
        project_root=Path(__file__).resolve().parent.parent,
    )

    print("\nTest tamamlandı!")


if __name__ == "__main__":
    main()
