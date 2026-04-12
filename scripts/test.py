#!/usr/bin/env python
"""
TensorFlow ile Eğitilen Modeli Test Et
"""
from __future__ import annotations

import argparse
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
from brain_mr_seg.metrics import DiceCoefficient, IoUCoefficient, dice_coefficient, iou_coefficient
from brain_mr_seg.splits import load_split


def evaluate_model(model, dataset, num_samples: int) -> dict:
    """Model değerlendirmesi yap - batch bazında verimli"""
    all_dice = []
    all_iou = []
    
    print("\nTest ediliyor...")
    for images, masks in tqdm(dataset, desc="Testing"):
        preds = model(images, training=False)
        preds_binary = tf.cast(tf.nn.sigmoid(preds) > 0.5, tf.float32)
        
        # Batch bazında metrik hesapla (daha verimli)
        batch_size = tf.shape(images)[0]
        for i in range(batch_size):
            pred_i = preds_binary[i]
            mask_i = masks[i]
            
            # Dice hesapla
            intersection = tf.reduce_sum(pred_i * mask_i)
            union = tf.reduce_sum(pred_i) + tf.reduce_sum(mask_i)
            dice = ((2.0 * intersection + 1e-6) / (union + 1e-6)).numpy()
            
            # IoU hesapla
            iou = ((intersection + 1e-6) / (union - intersection + 1e-6)).numpy()
            
            all_dice.append(dice)
            all_iou.append(iou)
            
            if len(all_dice) >= num_samples:
                break
        
        if len(all_dice) >= num_samples:
            break
    
    dice_arr = np.array(all_dice[:num_samples])
    iou_arr = np.array(all_iou[:num_samples])
    
    return {
        "dice_mean": dice_arr.mean(),
        "dice_std": dice_arr.std(),
        "iou_mean": iou_arr.mean(),
        "iou_std": iou_arr.std(),
        "all_dice": all_dice[:num_samples],
        "all_iou": all_iou[:num_samples],
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
    """Ortalama metriklerin bar chart grafiğini çiz"""
    if not HAS_MATPLOTLIB:
        print("UYARI: matplotlib yüklü değil, grafik atlanıyor.")
        return
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    metrics = ['Dice Score', 'IoU Score']
    means = [results['dice_mean'], results['iou_mean']]
    stds = [results['dice_std'], results['iou_std']]
    colors = ['#3498db', '#2ecc71']
    
    bars = ax.bar(metrics, means, yerr=stds, capsize=8, color=colors, 
                  edgecolor='white', alpha=0.8, linewidth=2)
    
    # Değerleri bar üzerine yaz
    for bar, mean, std in zip(bars, means, stds):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.02,
                f'{mean:.4f} ± {std:.4f}', ha='center', va='bottom', 
                fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Skor', fontsize=12)
    ax.set_title('Test Sonuçları - Ortalama Metrikler', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / "metrics_bar.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Bar chart grafiği kaydedildi: {output_dir / 'metrics_bar.png'}")


def visualize_predictions(
    model,
    data_dir: str,
    items: list,
    target_size: tuple,
    output_dir: Path,
    num_samples: int = 10,
    all_dice_scores: list = None,
    all_iou_scores: list = None,
):
    """Tahminleri görselleştir - dengeli örnekleme ile"""
    if not HAS_MATPLOTLIB:
        print("UYARI: matplotlib yüklü değil, görselleştirme atlanıyor.")
        return
    
    vis_dir = output_dir / "predictions"
    vis_dir.mkdir(exist_ok=True)
    
    # Dengeli örnekleme: farklı performans seviyelerinden örnekler seç
    if all_dice_scores is not None and len(all_dice_scores) == len(items):
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
        indices = np.random.choice(len(items), min(num_samples, len(items)), replace=False)
    
    for idx in indices:
        item = items[idx]
        img, mask = load_image_mask_pair(data_dir, item, target_size)
        
        # Predict
        img_batch = np.expand_dims(img, 0)
        pred = model(img_batch, training=False)
        pred = tf.nn.sigmoid(pred).numpy()[0]
        pred_binary = (pred > 0.5).astype(np.float32)
        
        # Dice ve IoU hesapla
        intersection = np.sum(mask * pred_binary)
        union = np.sum(mask) + np.sum(pred_binary)
        dice = (2 * intersection + 1e-6) / (union + 1e-6)
        iou = (intersection + 1e-6) / (union - intersection + 1e-6)
        
        # Görselleştir
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        axes[0].imshow(img[:, :, 0], cmap="gray")
        axes[0].set_title("Giriş Görüntüsü", fontsize=12)
        axes[0].axis("off")
        
        axes[1].imshow(mask[:, :, 0], cmap="gray")
        axes[1].set_title("Ground Truth", fontsize=12)
        axes[1].axis("off")
        
        axes[2].imshow(pred[:, :, 0], cmap="gray")
        axes[2].set_title("Tahmin (Olasılık)", fontsize=12)
        axes[2].axis("off")
        
        # Overlay - Ground truth (yeşil) vs Prediction (kırmızı)
        overlay = np.zeros((*img.shape[:2], 3))
        overlay[..., 0] = pred_binary[:, :, 0]  # Kırmızı - tahmin
        overlay[..., 1] = mask[:, :, 0]  # Yeşil - ground truth
        axes[3].imshow(img[:, :, 0], cmap="gray", alpha=0.7)
        axes[3].imshow(overlay, alpha=0.5)
        axes[3].set_title(f"Karşılaştırma\nDice: {dice:.4f} | IoU: {iou:.4f}\n(Yeşil: GT, Kırmızı: Tahmin)", fontsize=11)
        axes[3].axis("off")
        
        plt.tight_layout()
        plt.savefig(vis_dir / f"sample_{idx:04d}.png", dpi=150, bbox_inches='tight')
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
    parser.add_argument("--num-vis", type=int, default=50, help="Görselleştirilecek örnek sayısı")
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

    # Sonuçları yazdır
    print("\n" + "=" * 60)
    print("TEST SONUCLARI")
    print("=" * 60)
    print(f"   Dice Score: {results['dice_mean']:.4f} ± {results['dice_std']:.4f}")
    print(f"   IoU Score:  {results['iou_mean']:.4f} ± {results['iou_std']:.4f}")
    print("=" * 60)

    # Sonuçları kaydet
    results_file = output_dir / "test_results.txt"
    with open(results_file, "w", encoding="utf-8") as f:
        f.write("Beyin MR Segmentasyonu Test Sonuçları\n")
        f.write("=" * 40 + "\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Test örnekleri: {len(test_items)}\n\n")
        f.write(f"Dice Score: {results['dice_mean']:.4f} ± {results['dice_std']:.4f}\n")
        f.write(f"IoU Score:  {results['iou_mean']:.4f} ± {results['iou_std']:.4f}\n")
    print(f"\nSonuçlar kaydedildi: {results_file}")

    # Grafikleri oluştur
    print("\nGrafikler oluşturuluyor...")
    plot_metrics_histogram(results, output_dir)
    plot_metrics_boxplot(results, output_dir)
    plot_metrics_bar(results, output_dir)

    # Tahmin görselleştirmesi
    print("\nTahminler görselleştiriliyor...")
    visualize_predictions(
        model, args.data_dir, test_items, target_size, output_dir, args.num_vis,
        all_dice_scores=results['all_dice'],
        all_iou_scores=results['all_iou'],
    )

    print("\nTest tamamlandı!")


if __name__ == "__main__":
    main()
