#!/usr/bin/env python
"""
TensorFlow ile Beyin MR Segmentasyonu Eğitimi
GPU kullanarak UNet modelini eğitir.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# TensorFlow GPU ayarları (importtan önce)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")
# VRAM parçalanmasında OOM azaltmak için (CUDA uyumlu sürücülerde; TensorFlow uyarısında önerilen)
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")

import tensorflow as tf

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import time

from brain_mr_seg.model import build_unet
from brain_mr_seg.dataset import create_dataset
from brain_mr_seg.losses import BCEDiceLoss
from brain_mr_seg.metrics import DiceCoefficient, IoUCoefficient
from brain_mr_seg.splits import load_split
from brain_mr_seg.model_metrics import (
    parameters_millions,
    model_size_mb,
    get_gflops,
    get_gpu_memory_mb,
)
from brain_mr_seg.report import (
    write_full_metrics_report,
    save_training_metrics_json,
    save_training_run_config,
    update_training_run_config,
)


def configure_gpu_or_exit() -> None:
    """
    Eğitim yalnızca GPU ile çalışır. GPU yoksa veya görünmüyorsa süreç sonlanır.
    """
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        print("\n" + "=" * 60)
        print("HATA: GPU bulunamadi — egitim GPU zorunlu, CPU ile devam edilmez.")
        print("=" * 60)
        print(f"TensorFlow: {tf.__version__}")
        if cuda_visible.strip() in ("", "-1"):
            if cuda_visible.strip() == "-1":
                print("CUDA_VISIBLE_DEVICES=-1 ayarli; GPU gizlenmis olabilir. Kaldirin veya bos birakin.")
        print("Kontrol listesi:")
        print("  - NVIDIA surucusu guncel mi (nvidia-smi)?")
        print("  - TensorFlow surumunuz GPU + CUDA ile uyumlu mu?")
        print("  - https://www.tensorflow.org/install/pip")
        print("=" * 60 + "\n")
        sys.exit(1)
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU bellek ayari: {e}")


def print_gpu_info():
    """GPU bilgilerini yazdır (configure_gpu_or_exit sonrasi cagrilmali)."""
    print("\n" + "=" * 60)
    print("DONANIM BILGISI")
    print("=" * 60)
    print(f"TensorFlow version: {tf.__version__}")
    gpus = tf.config.list_physical_devices("GPU")
    print(f"GPU sayisi: {len(gpus)} (egitim GPU uzerinde)")
    for i, gpu in enumerate(gpus):
        print(f"   GPU {i}: {gpu.name}")
        try:
            details = tf.config.experimental.get_device_details(gpu)
            if "device_name" in details:
                print(f"   İsim: {details['device_name']}")
        except Exception:
            pass
    print("=" * 60 + "\n")


def get_callbacks(output_dir: Path, patience: int = 10, csv_append: bool = False):
    """Eğitim callback'lerini oluştur.
    Not: Metrik adlari compile'daki DiceCoefficient / IoUCoefficient ile ayni olmali
    (Keras log: dice_coeff -> val_dice_coeff, iou_coeff -> val_iou_coeff).
    """
    callbacks = [
        # En iyi modeli kaydet
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "checkpoints" / "best_model.h5"),
            monitor="val_dice_coeff",
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
            monitor="val_dice_coeff",
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
        # CSV Logger (yeni egitimde dosyayi ez; --resume ile devamda append)
        tf.keras.callbacks.CSVLogger(
            str(output_dir / "training_log.csv"),
            append=csv_append,
        ),
    ]
    return callbacks


def main():
    parser = argparse.ArgumentParser(description="TensorFlow ile Beyin MR Segmentasyonu Eğitimi")
    parser.add_argument("--data-dir", type=str, required=True, help="Data klasörü yolu")
    parser.add_argument("--split-file", type=str, required=True, help="Split JSON dosyası")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Çıktı klasörü")
    parser.add_argument("--epochs", type=int, default=50, help="Epoch sayısı")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch boyutu (6 GB civarı VRAM'de OOM olursa 8 veya 4 deneyin)",
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Başlangıç öğrenme oranı (Adam)")
    parser.add_argument("--image-size", type=int, default=256, help="Görüntü boyutu")
    parser.add_argument("--base-filters", type=int, default=32, help="UNet base filtre sayısı")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--resume", type=str, default=None, help=".h5 checkpoint'tan devam")
    parser.add_argument(
        "--initial-epoch",
        type=int,
        default=None,
        help="Resume sonrası Keras initial_epoch (0 tabanlı). Örn. 14 tam epoch bittiyse 14. "
        "Verilmezse dosya adında epoch_ bulunursa oradan alınır.",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Yalnizca zorunlu durumda: GPU yoksa CPU ile dene (yavas, onerilmez).",
    )
    args = parser.parse_args()

    if args.allow_cpu:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(f"GPU bellek ayari: {e}")
            print("\nUYARI: --allow-cpu gecildi; GPU var, GPU kullanilacak.\n")
        else:
            print("\nUYARI: --allow-cpu ile GPU yok, CPU kullanilacak (cok yavas).\n")
    else:
        configure_gpu_or_exit()

    print_gpu_info()

    # Çıktı klasörleri
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)

    # Split dosyasini yukle
    print(f"Split dosyasi yukleniyor: {args.split_file}")
    splits = load_split(args.split_file)
    train_items = splits["train"]
    val_items = splits["val"]
    
    print(f"   Eğitim seti: {len(train_items)} örnek")
    print(f"   Validasyon seti: {len(val_items)} örnek")
    print(f"   Test seti: {len(splits['test'])} örnek (eğitimde kullanılmayacak!)")

    split_path = Path(args.split_file)
    split_meta: dict = {}
    if split_path.exists():
        try:
            split_meta = json.loads(split_path.read_text(encoding="utf-8")).get("meta") or {}
        except Exception:
            split_meta = {}

    steps_per_epoch = len(train_items) // args.batch_size
    validation_steps = len(val_items) // args.batch_size

    run_config_path = output_dir / "training_run_config.json"
    save_training_run_config(
        run_config_path,
        {
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "tensorflow_version": tf.__version__,
            "data_dir": str(Path(args.data_dir).resolve()),
            "split_file": str(split_path.resolve()),
            "split_meta_from_json": split_meta,
            "output_dir": str(output_dir.resolve()),
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.lr,
            "image_size": args.image_size,
            "base_filters": args.base_filters,
            "early_stopping_patience": args.patience,
            "resume_checkpoint": args.resume,
            "train_samples": len(train_items),
            "val_samples": len(val_items),
            "test_samples": len(splits["test"]),
            "steps_per_epoch": steps_per_epoch,
            "validation_steps": validation_steps,
            "optimizer": "Adam",
            "loss": "BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)",
            "early_stopping": {
                "monitor": "val_dice_coeff",
                "mode": "max",
                "patience_epochs": args.patience,
                "restore_best_weights": True,
            },
            "reduce_lr_on_plateau": {
                "monitor": "val_loss",
                "factor": 0.5,
                "patience_epochs": 5,
                "min_lr": 1e-7,
            },
            "note": "Bu dosya train.py çalıştığında kaydedilir; geçmiş eğitimlerde yoksa komut satırı / konsol çıktısı kanıt olur.",
        },
    )
    print(f"   Eğitim ayarları kaydedildi: {run_config_path}")

    # Dataset olustur
    print("\nDataset olusturuluyor...")
    target_size = (args.image_size, args.image_size)
    
    train_dataset = create_dataset(
        data_dir=args.data_dir,
        items=train_items,
        batch_size=args.batch_size,
        target_size=target_size,
        shuffle=True,
        augment=True,
        repeat=True,
    )
    
    val_dataset = create_dataset(
        data_dir=args.data_dir,
        items=val_items,
        batch_size=args.batch_size,
        target_size=target_size,
        shuffle=False,
        augment=False,
    )

    # Model olustur
    print("\nModel olusturuluyor...")
    model = build_unet(
        input_shape=(*target_size, 1),
        base_filters=args.base_filters,
        num_classes=1,
    )
    
    # Model özeti ve 2. Grup metrikleri (params, size, GFLOPs)
    total_params = model.count_params()
    params_M = parameters_millions(model)
    size_mb = model_size_mb(model)
    gflops = get_gflops(model, (1, *target_size, 1))
    print(f"   Model: UNet")
    print(f"   Toplam parametre: {total_params:,}")
    print(f"   Parameters (M): {params_M:.4f}")
    print(f"   Model size (MB): {size_mb:.4f}")
    if gflops is not None:
        print(f"   GFLOPs: {gflops:.4f}")

    # Resume from checkpoint
    initial_epoch = 0
    if args.resume and os.path.exists(args.resume):
        print(f"\nModel yukleniyor: {args.resume}")
        model = tf.keras.models.load_model(
            args.resume,
            custom_objects={
                "BCEDiceLoss": BCEDiceLoss,
                "DiceCoefficient": DiceCoefficient,
                "IoUCoefficient": IoUCoefficient,
            }
        )
        if args.initial_epoch is not None:
            initial_epoch = max(0, args.initial_epoch)
        else:
            try:
                import re
                match = re.search(r"epoch_(\d+)", args.resume)
                if match:
                    initial_epoch = int(match.group(1))
                else:
                    print(
                        "   UYARI: --initial-epoch verilmedi; dosya adinda epoch_ yok. "
                        "0'dan devam edecek (epoch numaralari cakilabilir). "
                        "Kaldigin epoch icin ornek: --initial-epoch 14"
                    )
            except Exception:
                pass

    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss=BCEDiceLoss(bce_weight=0.5, dice_weight=0.5),
        metrics=[DiceCoefficient(), IoUCoefficient()],
    )

    # Callbacks
    callbacks = get_callbacks(output_dir, patience=args.patience, csv_append=bool(args.resume))

    # Egitim
    print("\n" + "=" * 60)
    print("EGITIM BASLIYOR")
    print("=" * 60)
    print(f"   Epochs: {args.epochs}")
    print(f"   Batch size: {args.batch_size}")
    print(f"   Learning rate: {args.lr}")
    print(f"   Image size: {args.image_size}x{args.image_size}")
    print("=" * 60 + "\n")

    training_start = time.perf_counter()
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

    training_elapsed = time.perf_counter() - training_start
    gpu_mem = get_gpu_memory_mb()

    # Sonuclari yazdir
    print("\n" + "=" * 60)
    print("EGITIM TAMAMLANDI!")
    print("=" * 60)
    
    best_dice = max(history.history.get("val_dice_coeff", [0]))
    best_iou = max(history.history.get("val_iou_coeff", [0]))
    
    print("   --- 1. GRUP (Segmentasyon) ---")
    print(f"   En iyi Validation Dice: {best_dice:.4f}")
    print(f"   En iyi Validation IoU: {best_iou:.4f}")
    print("   --- 2. GRUP (Model/Performans) ---")
    print(f"   Parameters (M): {params_M:.4f}")
    print(f"   Model size (MB): {size_mb:.4f}")
    print(f"   Training time: {training_elapsed:.2f} s ({training_elapsed/60:.2f} min)")
    if gflops is not None:
        print(f"   GFLOPs: {gflops:.4f}")
    if gpu_mem is not None:
        print(f"   GPU memory (MB): {gpu_mem:.2f}")
    print(f"   Modeller: {output_dir / 'checkpoints'}")
    print(f"   Loglar: {output_dir / 'logs'}")
    print("=" * 60)

    # Tek rapor dosyası (tüm başlıklar altında; 1. Grup sadece Dice/IoU, 2. Grup tam)
    report_path = output_dir / "full_metrics_report.txt"
    write_full_metrics_report(
        report_path,
        dice_mean=best_dice,
        iou_mean=best_iou,
        parameters_M=params_M,
        model_size_MB=size_mb,
        training_time_sec=training_elapsed,
        inference_time_sec=None,
        inference_time_per_sample_ms=None,
        gflops=gflops,
        gpu_memory_MB=gpu_mem,
    )
    save_training_metrics_json(
        output_dir / "training_metrics.json",
        training_time_sec=training_elapsed,
        parameters_M=params_M,
        model_size_MB=size_mb,
        gflops=gflops,
        gpu_memory_MB=gpu_mem,
    )
    update_training_run_config(
        run_config_path,
        {
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "training_time_sec": training_elapsed,
            "best_val_dice": float(best_dice),
            "best_val_iou": float(best_iou),
            "parameters_M": params_M,
            "model_size_MB": size_mb,
            "gflops": gflops,
            "gpu_memory_MB": gpu_mem,
        },
    )
    print(f"   Tüm metrikler raporu: {report_path}")


if __name__ == "__main__":
    main()
