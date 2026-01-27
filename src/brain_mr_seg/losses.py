"""
TensorFlow/Keras Loss Fonksiyonları
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow import keras


class BCEDiceLoss(keras.losses.Loss):
    """Binary Cross Entropy + Dice Loss kombinasyonu"""
    
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = keras.losses.BinaryCrossentropy(from_logits=True)
    
    def call(self, y_true, y_pred):
        # BCE loss (logits)
        bce_loss = self.bce(y_true, y_pred)
        
        # Dice loss
        y_pred_sigmoid = tf.nn.sigmoid(y_pred)
        intersection = tf.reduce_sum(y_true * y_pred_sigmoid)
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred_sigmoid)
        dice_score = (2.0 * intersection + 1e-6) / (union + 1e-6)
        dice_loss = 1.0 - dice_score
        
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "bce_weight": self.bce_weight,
            "dice_weight": self.dice_weight,
        })
        return config


class DiceLoss(keras.losses.Loss):
    """Dice Loss"""
    
    def __init__(self, smooth: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        self.smooth = smooth
    
    def call(self, y_true, y_pred):
        y_pred_sigmoid = tf.nn.sigmoid(y_pred)
        intersection = tf.reduce_sum(y_true * y_pred_sigmoid)
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred_sigmoid)
        dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice_score
    
    def get_config(self):
        config = super().get_config()
        config.update({"smooth": self.smooth})
        return config
