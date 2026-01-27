"""
TensorFlow/Keras UNet Modeli
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def conv_block(x, filters: int, kernel_size: int = 3):
    """Çift konvolüsyon bloğu"""
    x = layers.Conv2D(filters, kernel_size, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(filters, kernel_size, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    return x


def encoder_block(x, filters: int):
    """Encoder bloğu: conv_block + max pooling"""
    skip = conv_block(x, filters)
    pool = layers.MaxPooling2D(pool_size=(2, 2))(skip)
    return skip, pool


def decoder_block(x, skip, filters: int):
    """Decoder bloğu: upsample + concat + conv_block"""
    x = layers.Conv2DTranspose(filters, (2, 2), strides=(2, 2), padding='same')(x)
    x = layers.Concatenate()([x, skip])
    x = conv_block(x, filters)
    return x


def build_unet(
    input_shape: tuple = (256, 256, 1),
    base_filters: int = 32,
    num_classes: int = 1,
) -> keras.Model:
    """
    U-Net modelini oluştur
    
    Args:
        input_shape: Giriş görüntü boyutu (H, W, C)
        base_filters: İlk katmandaki filtre sayısı
        num_classes: Çıkış sınıf sayısı
    
    Returns:
        Keras Model
    """
    inputs = keras.Input(shape=input_shape)
    
    # Encoder
    s1, p1 = encoder_block(inputs, base_filters)      # 256 -> 128
    s2, p2 = encoder_block(p1, base_filters * 2)      # 128 -> 64
    s3, p3 = encoder_block(p2, base_filters * 4)      # 64 -> 32
    s4, p4 = encoder_block(p3, base_filters * 8)      # 32 -> 16
    
    # Bridge
    bridge = conv_block(p4, base_filters * 16)        # 16
    
    # Decoder
    d4 = decoder_block(bridge, s4, base_filters * 8)  # 16 -> 32
    d3 = decoder_block(d4, s3, base_filters * 4)      # 32 -> 64
    d2 = decoder_block(d3, s2, base_filters * 2)      # 64 -> 128
    d1 = decoder_block(d2, s1, base_filters)          # 128 -> 256
    
    # Output
    outputs = layers.Conv2D(num_classes, (1, 1), padding='same')(d1)
    
    model = keras.Model(inputs, outputs, name='UNet')
    return model


if __name__ == "__main__":
    model = build_unet()
    model.summary()
