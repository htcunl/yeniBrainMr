"""
TensorFlow/Keras Dataset
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import tensorflow as tf


def load_image_mask_pair(
    data_dir: str,
    item: dict,
    target_size: Tuple[int, int] = (256, 256),
) -> Tuple[np.ndarray, np.ndarray]:
    """Tek bir görüntü-maske çiftini yükle"""
    img_path = os.path.join(data_dir, item["image"])
    mask_path = os.path.join(data_dir, item["mask"])
    # tf.image.resize: size her zaman (H, W) iki skaler olmalı (py_function içinde güvenli)
    h, w = int(target_size[0]), int(target_size[1])
    size_hw = (h, w)

    # Görüntü yükle
    img = tf.io.read_file(img_path)
    img = tf.image.decode_png(img, channels=1)
    img = tf.image.resize(img, size_hw)
    img = tf.cast(img, tf.float32) / 255.0
    
    # Maske yükle
    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, size_hw, method="nearest")
    mask = tf.cast(mask, tf.float32) / 255.0
    
    return img.numpy(), mask.numpy()


def create_dataset(
    data_dir: str,
    items: List[dict],
    batch_size: int = 16,
    target_size: Tuple[int, int] = (256, 256),
    shuffle: bool = True,
    augment: bool = False,
    seed: int = 42,
    repeat: bool = False,
) -> tf.data.Dataset:
    """
    TensorFlow Dataset oluştur
    
    Args:
        data_dir: Data klasörü yolu
        items: [{"image": "path", "mask": "path"}, ...]
        batch_size: Batch boyutu
        target_size: Hedef görüntü boyutu
        shuffle: Shuffle yapılsın mı
        augment: Data augmentation yapılsın mı
        seed: Random seed
        repeat: True ise veri seti sonsuz tekrarlanır (model.fit çoklu epoch + steps_per_epoch için önerilir)
    
    Returns:
        tf.data.Dataset
    """
    
    def load_data(item_idx):
        item_idx = item_idx.numpy()
        item = items[item_idx]
        img, mask = load_image_mask_pair(data_dir, item, target_size)
        return img, mask
    
    def tf_load_data(item_idx):
        img, mask = tf.py_function(
            load_data,
            [item_idx],
            [tf.float32, tf.float32]
        )
        img.set_shape([target_size[0], target_size[1], 1])
        mask.set_shape([target_size[0], target_size[1], 1])
        return img, mask
    
    def augment_fn(img, mask):
        """Basit data augmentation"""
        # Random horizontal flip
        if tf.random.uniform(()) > 0.5:
            img = tf.image.flip_left_right(img)
            mask = tf.image.flip_left_right(mask)
        
        # Random vertical flip
        if tf.random.uniform(()) > 0.5:
            img = tf.image.flip_up_down(img)
            mask = tf.image.flip_up_down(mask)
        
        # Random rotation (90, 180, 270)
        k = tf.random.uniform((), minval=0, maxval=4, dtype=tf.int32)
        img = tf.image.rot90(img, k)
        mask = tf.image.rot90(mask, k)
        
        # Random brightness/contrast
        img = tf.image.random_brightness(img, 0.1)
        img = tf.image.random_contrast(img, 0.9, 1.1)
        img = tf.clip_by_value(img, 0, 1)
        
        return img, mask
    
    # Dataset oluştur
    indices = tf.data.Dataset.range(len(items))
    
    if shuffle:
        indices = indices.shuffle(buffer_size=len(items), seed=seed)
    
    dataset = indices.map(tf_load_data, num_parallel_calls=tf.data.AUTOTUNE)
    
    if augment:
        dataset = dataset.map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
    
    dataset = dataset.batch(batch_size)
    if repeat:
        dataset = dataset.repeat()
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset
