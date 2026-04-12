#!/usr/bin/env python
"""
Eğitim sonuçlarını görselleştir ve test et
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# GPU bellek büyümesi
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from brain_mr_seg.model import build_unet
from brain_mr_seg.dataset import load_image_mask_pair
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


def plot_training_history(csv_path: str, output_dir: Path):
    """Eğitim geçmişini grafiklerle göster"""
    df = pd.read_csv(csv_path)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Beyin MR Segmentasyonu - Eğitim Sonuçları', fontsize=16, fontweight='bold')
    
    # Loss
    ax1 = axes[0, 0]
    ax1.plot(df['epoch'] + 1, df['loss'], 'b-', label='Train Loss', linewidth=2)
    ax1.plot(df['epoch'] + 1, df['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Grafiği')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Dice
    ax2 = axes[0, 1]
    ax2.plot(df['epoch'] + 1, df['dice'], 'b-', label='Train Dice', linewidth=2)
    ax2.plot(df['epoch'] + 1, df['val_dice'], 'r-', label='Validation Dice', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Dice Score')
    ax2.set_title('Dice Score Grafiği')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # IoU
    ax3 = axes[1, 0]
    ax3.plot(df['epoch'] + 1, df['iou'], 'b-', label='Train IoU', linewidth=2)
    ax3.plot(df['epoch'] + 1, df['val_iou'], 'r-', label='Validation IoU', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('IoU Score')
    ax3.set_title('IoU Score Grafiği')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Learning Rate
    ax4 = axes[1, 1]
    ax4.plot(df['epoch'] + 1, df['lr'], 'g-', linewidth=2)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Learning Rate')
    ax4.set_title('Learning Rate Değişimi')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Egitim grafikleri kaydedildi: {output_dir / 'training_curves.png'}")
    
    # En iyi değerleri yazdır
    best_idx = df['val_dice'].idxmax()
    print(f"\nEN IYI EPOCH: {df.loc[best_idx, 'epoch'] + 1}")
    print(f"   Train Dice: {df.loc[best_idx, 'dice']:.4f}")
    print(f"   Val Dice: {df.loc[best_idx, 'val_dice']:.4f}")
    print(f"   Train IoU: {df.loc[best_idx, 'iou']:.4f}")
    print(f"   Val IoU: {df.loc[best_idx, 'val_iou']:.4f}")


def test_model(model, data_dir: str, test_items: list, target_size=(256, 256)):
    """Test seti üzerinde model değerlendirmesi — 1. Grup: Dice, IoU, Sensitivity, Specificity, Precision, F1, HD95, ASD, MCC + inference time."""
    all_dice = []
    all_iou = []
    all_sensitivity = []
    all_specificity = []
    all_precision = []
    all_f1 = []
    all_hd95 = []
    all_asd = []
    all_mcc = []

    print(f"\nTest ediliyor (1. Grup + inference time)... ({len(test_items)} ornek)")
    inference_start = time.perf_counter()

    for i, item in enumerate(test_items):
        try:
            img, mask = load_image_mask_pair(data_dir, item, target_size)
            img_batch = np.expand_dims(img, 0)

            pred = model(img_batch, training=False)
            pred_prob = tf.nn.sigmoid(pred).numpy()
            pred_binary = (pred_prob > 0.5).astype(np.float32)
            m = mask
            p = pred_binary[0] if pred_binary.ndim == 3 else pred_binary

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

            if (i + 1) % 100 == 0:
                print(f"   İlerleme: {i + 1}/{len(test_items)}")
        except Exception as e:
            key = item.get("image", item) if isinstance(item, dict) else getattr(item, "image", item)
            print(f"   Hata: {key} - {e}")

    inference_elapsed = time.perf_counter() - inference_start

    dice_arr = np.array(all_dice)
    iou_arr = np.array(all_iou)
    sens_arr = np.array(all_sensitivity)
    spec_arr = np.array(all_specificity)
    prec_arr = np.array(all_precision)
    f1_arr = np.array(all_f1)
    hd95_arr = np.array(all_hd95)
    hd95_finite = hd95_arr[np.isfinite(hd95_arr)]
    asd_arr = np.array(all_asd)
    asd_finite = asd_arr[np.isfinite(asd_arr)]
    mcc_arr = np.array(all_mcc)

    return {
        'dice_mean': float(dice_arr.mean()),
        'dice_std': float(dice_arr.std()),
        'iou_mean': float(iou_arr.mean()),
        'iou_std': float(iou_arr.std()),
        'sensitivity_mean': float(sens_arr.mean()),
        'sensitivity_std': float(sens_arr.std()),
        'specificity_mean': float(spec_arr.mean()),
        'specificity_std': float(spec_arr.std()),
        'precision_mean': float(prec_arr.mean()),
        'precision_std': float(prec_arr.std()),
        'f1_mean': float(f1_arr.mean()),
        'f1_std': float(f1_arr.std()),
        'hd95_mean': float(hd95_finite.mean()) if len(hd95_finite) > 0 else float('nan'),
        'hd95_std': float(hd95_finite.std()) if len(hd95_finite) > 1 else 0.0,
        'asd_mean': float(asd_finite.mean()) if len(asd_finite) > 0 else float('nan'),
        'asd_std': float(asd_finite.std()) if len(asd_finite) > 1 else 0.0,
        'mcc_mean': float(mcc_arr.mean()),
        'mcc_std': float(mcc_arr.std()),
        'inference_time_sec': inference_elapsed,
        'inference_time_per_sample_ms': 1000.0 * inference_elapsed / len(test_items) if test_items else 0.0,
        'num_test_samples': len(test_items),
        'all_dice': all_dice,
        'all_iou': all_iou,
        'all_sensitivity': all_sensitivity,
        'all_specificity': all_specificity,
        'all_precision': all_precision,
        'all_f1': all_f1,
        'all_hd95': all_hd95,
        'all_asd': all_asd,
        'all_mcc': all_mcc,
    }


def visualize_predictions(model, data_dir: str, test_items: list, output_dir: Path, num_samples=12, target_size=(256, 256)):
    """Test tahminlerini görselleştir"""
    indices = np.random.choice(len(test_items), min(num_samples, len(test_items)), replace=False)
    
    rows = (len(indices) + 3) // 4
    fig, axes = plt.subplots(rows, 4, figsize=(16, 4 * rows))
    fig.suptitle('Test Görüntüleri - Tahminler', fontsize=16, fontweight='bold')
    
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx_plot, idx_data in enumerate(indices):
        row = idx_plot // 4
        col = idx_plot % 4
        
        item = test_items[idx_data]
        img, mask = load_image_mask_pair(data_dir, item, target_size)
        
        pred = model(np.expand_dims(img, 0), training=False)
        pred_prob = tf.nn.sigmoid(pred).numpy()[0]
        pred_binary = (pred_prob > 0.5).astype(np.float32)
        
        # Dice hesapla
        intersection = np.sum(mask * pred_binary)
        union = np.sum(mask) + np.sum(pred_binary)
        dice = (2 * intersection + 1e-6) / (union + 1e-6)
        
        # Overlay görüntüsü oluştur
        overlay = np.zeros((*target_size, 3))
        overlay[:, :, 0] = img[:, :, 0]  # Orijinal görüntü (R)
        overlay[:, :, 1] = img[:, :, 0]  # Orijinal görüntü (G)
        overlay[:, :, 2] = img[:, :, 0]  # Orijinal görüntü (B)
        
        # Mask ve prediction overlay
        overlay[:, :, 1] = np.clip(overlay[:, :, 1] + mask[:, :, 0] * 0.3, 0, 1)  # GT yeşil
        overlay[:, :, 0] = np.clip(overlay[:, :, 0] + pred_binary[:, :, 0] * 0.3, 0, 1)  # Pred kırmızı
        
        axes[row, col].imshow(overlay)
        axes[row, col].set_title(f'Dice: {dice:.3f}', fontsize=10)
        axes[row, col].axis('off')
    
    # Boş eksenleri gizle
    for idx_plot in range(len(indices), rows * 4):
        row = idx_plot // 4
        col = idx_plot % 4
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'test_predictions.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Test tahminleri kaydedildi: {output_dir / 'test_predictions.png'}")


def plot_test_distribution(results: dict, output_dir: Path):
    """Test sonuçlarının dağılımını göster"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('Test Seti - Metrik Dağılımları', fontsize=14, fontweight='bold')
    
    # Dice dağılımı
    axes[0].hist(results['all_dice'], bins=30, color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].axvline(results['dice_mean'], color='red', linestyle='--', linewidth=2, label=f"Ortalama: {results['dice_mean']:.4f}")
    axes[0].set_xlabel('Dice Score')
    axes[0].set_ylabel('Frekans')
    axes[0].set_title('Dice Score Dağılımı')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # IoU dağılımı
    axes[1].hist(results['all_iou'], bins=30, color='seagreen', edgecolor='black', alpha=0.7)
    axes[1].axvline(results['iou_mean'], color='red', linestyle='--', linewidth=2, label=f"Ortalama: {results['iou_mean']:.4f}")
    axes[1].set_xlabel('IoU Score')
    axes[1].set_ylabel('Frekans')
    axes[1].set_title('IoU Score Dağılımı')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'test_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Dagilim grafikleri kaydedildi: {output_dir / 'test_distribution.png'}")


def main():
    # Ayarlar
    data_dir = r"c:\Users\Lenovo\Desktop\archive (1)"
    split_file = r"c:\Users\Lenovo\Desktop\yeniBrainMr\splits.json"
    checkpoint_path = r"c:\Users\Lenovo\Desktop\yeniBrainMr\outputs\checkpoints\best_model.h5"
    csv_path = r"c:\Users\Lenovo\Desktop\yeniBrainMr\outputs\training_log.csv"
    output_dir = Path(r"c:\Users\Lenovo\Desktop\yeniBrainMr\outputs")
    
    print("=" * 60)
    print("BEYIN MR SEGMENTASYONU - TEST VE GORSELLESTIRME")
    print("=" * 60)
    
    # 1. Egitim grafiklerini ciz
    print("\nEgitim grafikleri olusturuluyor...")
    plot_training_history(csv_path, output_dir)
    
    # 2. Model yukle
    print("\nModel yukleniyor...")
    model = tf.keras.models.load_model(
        checkpoint_path,
        custom_objects={
            'BCEDiceLoss': BCEDiceLoss,
            'DiceCoefficient': DiceCoefficient,
            'IoUCoefficient': IoUCoefficient,
        }
    )
    print("   Model yuklendi!")
    
    # 3. Test setini yükle
    splits = load_split(split_file)
    test_items = splits['test']
    print(f"\nTest seti: {len(test_items)} goruntu")
    
    # 4. 2. Grup model metrikleri (params, size, GFLOPs, GPU memory)
    target_size = (256, 256)
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

    # 5. Test et
    results = test_model(model, data_dir, test_items, target_size)

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

    # Tek rapor dosyası (tüm metrikler başlıklar altında)
    train_metrics = load_training_metrics_json(output_dir / "training_metrics.json")
    training_time_sec = train_metrics.get("training_time_sec") if train_metrics else None
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
        checkpoint_path=checkpoint_path,
    )
    print(f"\nTüm metrikler raporu: {report_path}")
    
    # 6. Test dagilimlarini ciz
    print("\nTest dagilimlari olusturuluyor...")
    plot_test_distribution(results, output_dir)
    
    # 7. Tahminleri gorsellestir
    print("\nOrnek tahminler gorsellestiriliyor...")
    visualize_predictions(model, data_dir, test_items, output_dir, num_samples=12, target_size=target_size)
    
    # 8. Sonuçları dosyaya kaydet
    results_file = output_dir / 'test_results.txt'
    hd95_str = f"{results['hd95_mean']:.2f} ± {results['hd95_std']:.2f}" if np.isfinite(results['hd95_mean']) else "N/A"
    asd_str = f"{results['asd_mean']:.2f} ± {results['asd_std']:.2f}" if np.isfinite(results['asd_mean']) else "N/A"
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("=" * 50 + "\n")
        f.write("BEYİN MR SEGMENTASYONU - TEST SONUÇLARI\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Test Görüntü Sayısı: {len(test_items)}\n\n")
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
        f.write(f"Model size (MB):    {size_mb:.4f}\n")
        if gflops is not None:
            f.write(f"GFLOPs:             {gflops:.4f}\n")
        if gpu_mem is not None:
            f.write(f"GPU memory (MB):    {gpu_mem:.2f}\n")
        f.write(f"Inference time (s): {results['inference_time_sec']:.2f}\n")
        f.write(f"Inference time (ms/örnek): {results['inference_time_per_sample_ms']:.2f}\n")
        f.write("\n" + "=" * 50 + "\n")
    print(f"\nSonuclar kaydedildi: {results_file}")
    
    print("\nTamamlandi!")


if __name__ == "__main__":
    main()
