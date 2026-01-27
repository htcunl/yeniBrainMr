#!/usr/bin/env python
"""
TensorFlow ile Beyin MR Segmentasyonu Eğitimi
GPU kullanarak UNet modelini eğitir.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# TensorFlow GPU ayarları
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'  # INFO mesajlarını gizle

import tensorflow as tf

# Bellek büyümesini etkinleştir
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from brain_mr_seg.model import build_unet
from brain_mr_seg.dataset import create_dataset
from brain_mr_seg.losses import BCEDiceLoss
from brain_mr_seg.metrics import DiceCoefficient, IoUCoefficient
from brain_mr_seg.splits import load_split


def print_gpu_info():
    """GPU bilgilerini yazdır"""
    print("\n" + "=" * 60)
    print("🖥️  DONANIM BİLGİSİ")
    print("=" * 60)
    print(f"TensorFlow version: {tf.__version__}")
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ GPU sayısı: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            print(f"   GPU {i}: {gpu.name}")
            # GPU bellek bilgisi
            try:
                details = tf.config.experimental.get_device_details(gpu)
                if 'device_name' in details:
                    print(f"   İsim: {details['device_name']}")
            except:
                pass
    else:
        print("❌ GPU bulunamadı! CPU kullanılacak.")
    print("=" * 60 + "\n")


def get_callbacks(output_dir: Path, patience: int = 10):
    """Eğitim callback'lerini oluştur"""
    callbacks = [
        # En iyi modeli kaydet
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "checkpoints" / "best_model.h5"),
            monitor="val_dice",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        # Son modeli kaydet
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "checkpoints" / "last_model.h5"),
            save_best_only=False,
            verbose=0,
        ),
        # Early stopping
        tf.keras.callbacks.EarlyStopping(
            monitor="val_dice",
            mode="max",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        # Learning rate scheduler
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1,
        ),
        # TensorBoard
        tf.keras.callbacks.TensorBoard(
            log_dir=str(output_dir / "logs" / datetime.now().strftime("%Y%m%d-%H%M%S")),
            histogram_freq=0,
            write_graph=True,
        ),
        # CSV Logger
        tf.keras.callbacks.CSVLogger(
            str(output_dir / "training_log.csv"),
            append=True,
        ),
    ]
    return callbacks
    return callbacks


def main():
    parser = argparse.ArgumentParser(description="TensorFlow ile Beyin MR Segmentasyonu Eğitimi")
    parser.add_argument("--data-dir", type=str, required=True, help="Data klasörü yolu")
    parser.add_argument("--split-file", type=str, required=True, help="Split JSON dosyası")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Çıktı klasörü")
    parser.add_argument("--epochs", type=int, default=50, help="Epoch sayısı")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch boyutu")
    parser.add_argument("--lr", type=float, default=1e-3, help="Öğrenme oranı")
    parser.add_argument("--image-size", type=int, default=256, help="Görüntü boyutu")
    parser.add_argument("--base-filters", type=int, default=32, help="UNet base filtre sayısı")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint'tan devam et")
    args = parser.parse_args()

    # GPU bilgisi
    print_gpu_info()

    # Çıktı klasörleri
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)

    # Split dosyasını yükle
    print(f"📂 Split dosyası yükleniyor: {args.split_file}")
    splits = load_split(args.split_file)
    train_items = splits["train"]
    val_items = splits["val"]
    
    print(f"   Eğitim seti: {len(train_items)} örnek")
    print(f"   Validasyon seti: {len(val_items)} örnek")
    print(f"   Test seti: {len(splits['test'])} örnek (eğitimde kullanılmayacak!)")

    # Dataset oluştur
    print("\n📊 Dataset oluşturuluyor...")
    target_size = (args.image_size, args.image_size)
    
    train_dataset = create_dataset(
        data_dir=args.data_dir,
        items=train_items,
        batch_size=args.batch_size,
        target_size=target_size,
        shuffle=True,
        augment=True,
    )
    
    val_dataset = create_dataset(
        data_dir=args.data_dir,
        items=val_items,
        batch_size=args.batch_size,
        target_size=target_size,
        shuffle=False,
        augment=False,
    )

    # Model oluştur
    print("\n🧠 Model oluşturuluyor...")
    model = build_unet(
        input_shape=(*target_size, 1),
        base_filters=args.base_filters,
        num_classes=1,
    )
    
    # Model özeti
    total_params = model.count_params()
    print(f"   Model: UNet")
    print(f"   Toplam parametre: {total_params:,}")

    # Resume from checkpoint
    initial_epoch = 0
    if args.resume and os.path.exists(args.resume):
        print(f"\n📥 Model yükleniyor: {args.resume}")
        model = tf.keras.models.load_model(
            args.resume,
            custom_objects={
                "BCEDiceLoss": BCEDiceLoss,
                "DiceCoefficient": DiceCoefficient,
                "IoUCoefficient": IoUCoefficient,
            }
        )
        # Epoch bilgisini almaya çalış
        try:
            import re
            match = re.search(r'epoch_(\d+)', args.resume)
            if match:
                initial_epoch = int(match.group(1))
        except:
            pass

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss=BCEDiceLoss(bce_weight=0.5, dice_weight=0.5),
        metrics=[DiceCoefficient(), IoUCoefficient()],
    )

    # Callbacks
    callbacks = get_callbacks(output_dir, patience=args.patience)

    # Eğitim
    print("\n" + "=" * 60)
    print("🚀 EĞİTİM BAŞLIYOR")
    print("=" * 60)
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.lr}")
    print(f"   Image size: {args.image_size}x{args.image_size}")
    print("=" * 60 + "\n")

    # Steps per epoch hesapla
    steps_per_epoch = len(train_items) // args.batch_size
    validation_steps = len(val_items) // args.batch_size

    history = model.fit(
        train_dataset,
        epochs=args.epochs,
        initial_epoch=initial_epoch,
        validation_data=val_dataset,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1,
    )

    # Sonuçları yazdır
    print("\n" + "=" * 60)
    print("🎉 EĞİTİM TAMAMLANDI!")
    print("=" * 60)
    
    best_dice = max(history.history.get('val_dice', [0]))
    best_iou = max(history.history.get('val_iou', [0]))
    
    print(f"   En iyi Validation Dice: {best_dice:.4f}")
    print(f"   En iyi Validation IoU: {best_iou:.4f}")
    print(f"   Modeller: {output_dir / 'checkpoints'}")
    print(f"   Loglar: {output_dir / 'logs'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
