"""
TensorFlow/Keras Metrikler
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow import keras


class DiceCoefficient(keras.metrics.Metric):
    """Dice Coefficient metriği"""
    
    def __init__(self, threshold: float = 0.5, smooth: float = 1e-6, name='dice_coeff', **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.smooth = smooth
        self.dice_sum = self.add_weight(name='dice_sum', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_sigmoid = tf.nn.sigmoid(y_pred)
        y_pred_binary = tf.cast(y_pred_sigmoid > self.threshold, tf.float32)
        
        intersection = tf.reduce_sum(y_true * y_pred_binary)
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred_binary)
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        self.dice_sum.assign_add(dice)
        self.count.assign_add(1.0)
    
    def result(self):
        return self.dice_sum / (self.count + 1e-6)
    
    def reset_state(self):
        self.dice_sum.assign(0.0)
        self.count.assign(0.0)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "threshold": self.threshold,
            "smooth": self.smooth,
        })
        return config


class IoUCoefficient(keras.metrics.Metric):
    """IoU (Intersection over Union) metriği"""
    
    def __init__(self, threshold: float = 0.5, smooth: float = 1e-6, name='iou_coeff', **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.smooth = smooth
        self.iou_sum = self.add_weight(name='iou_sum', initializer='zeros')
        self.count = self.add_weight(name='count', initializer='zeros')
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_sigmoid = tf.nn.sigmoid(y_pred)
        y_pred_binary = tf.cast(y_pred_sigmoid > self.threshold, tf.float32)
        
        intersection = tf.reduce_sum(y_true * y_pred_binary)
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred_binary) - intersection
        iou = (intersection + self.smooth) / (union + self.smooth)
        
        self.iou_sum.assign_add(iou)
        self.count.assign_add(1.0)
    
    def result(self):
        return self.iou_sum / (self.count + 1e-6)
    
    def reset_state(self):
        self.iou_sum.assign(0.0)
        self.count.assign(0.0)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "threshold": self.threshold,
            "smooth": self.smooth,
        })
        return config


def dice_coefficient(y_true, y_pred, threshold: float = 0.5, smooth: float = 1e-6):
    """Dice coefficient fonksiyonu"""
    y_pred_sigmoid = tf.nn.sigmoid(y_pred)
    y_pred_binary = tf.cast(y_pred_sigmoid > threshold, tf.float32)
    
    intersection = tf.reduce_sum(y_true * y_pred_binary)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred_binary)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice


def iou_coefficient(y_true, y_pred, threshold: float = 0.5, smooth: float = 1e-6):
    """IoU coefficient fonksiyonu"""
    y_pred_sigmoid = tf.nn.sigmoid(y_pred)
    y_pred_binary = tf.cast(y_pred_sigmoid > threshold, tf.float32)
    
    intersection = tf.reduce_sum(y_true * y_pred_binary)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred_binary) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou
