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

from brain_mr_seg.model_tf import build_unet
from brain_mr_seg.dataset_tf import load_image_mask_pair
from brain_mr_seg.losses_tf import BCEDiceLoss
from brain_mr_seg.metrics_tf import DiceCoefficient, IoUCoefficient, dice_coefficient, iou_coefficient
from brain_mr_seg.splits import load_split


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
    """Test seti üzerinde model değerlendirmesi"""
    all_dice = []
    all_iou = []
    
    print(f"\nTest ediliyor... ({len(test_items)} ornek)")
    
    for i, item in enumerate(test_items):
        try:
            img, mask = load_image_mask_pair(data_dir, item, target_size)
            img_batch = np.expand_dims(img, 0)
            
            pred = model(img_batch, training=False)
            pred_prob = tf.nn.sigmoid(pred).numpy()
            pred_binary = (pred_prob > 0.5).astype(np.float32)
            
            # Dice ve IoU hesapla
            intersection = np.sum(mask * pred_binary[0])
            union_dice = np.sum(mask) + np.sum(pred_binary[0])
            union_iou = np.sum(mask) + np.sum(pred_binary[0]) - intersection
            
            dice = (2 * intersection + 1e-6) / (union_dice + 1e-6)
            iou = (intersection + 1e-6) / (union_iou + 1e-6)
            
            all_dice.append(dice)
            all_iou.append(iou)
            
            if (i + 1) % 100 == 0:
                print(f"   İlerleme: {i + 1}/{len(test_items)}")
        except Exception as e:
            print(f"   Hata: {item.image} - {e}")
    
    dice_arr = np.array(all_dice)
    iou_arr = np.array(all_iou)
    
    return {
        'dice_mean': dice_arr.mean(),
        'dice_std': dice_arr.std(),
        'iou_mean': iou_arr.mean(),
        'iou_std': iou_arr.std(),
        'all_dice': all_dice,
        'all_iou': all_iou,
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
    
    # 4. Test et
    results = test_model(model, data_dir, test_items)
    
    print("\n" + "=" * 60)
    print("TEST SONUCLARI")
    print("=" * 60)
    print(f"   Dice Score: {results['dice_mean']:.4f} ± {results['dice_std']:.4f}")
    print(f"   IoU Score:  {results['iou_mean']:.4f} ± {results['iou_std']:.4f}")
    print("=" * 60)
    
    # 5. Test dagilimlarini ciz
    print("\nTest dagilimlari olusturuluyor...")
    plot_test_distribution(results, output_dir)
    
    # 6. Tahminleri gorsellestir
    print("\nOrnek tahminler gorsellestiriliyor...")
    visualize_predictions(model, data_dir, test_items, output_dir, num_samples=12)
    
    # 7. Sonuçları dosyaya kaydet
    results_file = output_dir / 'test_results.txt'
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("=" * 50 + "\n")
        f.write("BEYİN MR SEGMENTASYONU - TEST SONUÇLARI\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Test Görüntü Sayısı: {len(test_items)}\n\n")
        f.write(f"Dice Score: {results['dice_mean']:.4f} ± {results['dice_std']:.4f}\n")
        f.write(f"IoU Score:  {results['iou_mean']:.4f} ± {results['iou_std']:.4f}\n")
        f.write("\n" + "=" * 50 + "\n")
    print(f"\nSonuclar kaydedildi: {results_file}")
    
    print("\nTamamlandi!")


if __name__ == "__main__":
    main()
