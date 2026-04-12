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


# ---------------------------------------------------------------------------
# Değerlendirme metrikleri (NumPy - test/raporlama için)
# 1. Grup: Dice, IoU, Sensitivity, Specificity, Precision, F1, HD95, ASD, MCC
# ---------------------------------------------------------------------------

def _ensure_numpy_binary(mask, threshold: float = 0.5):
    """Mask'ı tek kanallı binary numpy array (H, W) yap."""
    import numpy as np
    if hasattr(mask, "numpy"):
        mask = mask.numpy()
    mask = np.asarray(mask).squeeze()
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return (mask > threshold).astype(np.float32)


def compute_dice_np(y_true, y_pred, threshold: float = 0.5, smooth: float = 1e-6):
    """Dice coefficient (NumPy)."""
    import numpy as np
    y_true = _ensure_numpy_binary(y_true, threshold)
    y_pred = _ensure_numpy_binary(y_pred, threshold)
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred)
    return float((2.0 * intersection + smooth) / (union + smooth))


def compute_iou_np(y_true, y_pred, threshold: float = 0.5, smooth: float = 1e-6):
    """IoU - Intersection over Union (NumPy)."""
    import numpy as np
    y_true = _ensure_numpy_binary(y_true, threshold)
    y_pred = _ensure_numpy_binary(y_pred, threshold)
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    return float((intersection + smooth) / (union + smooth))


def compute_sensitivity_np(y_true, y_pred, threshold: float = 0.5, smooth: float = 1e-6):
    """Sensitivity (Recall, TPR) = TP / (TP + FN)."""
    import numpy as np
    y_true = _ensure_numpy_binary(y_true, threshold)
    y_pred = _ensure_numpy_binary(y_pred, threshold)
    tp = np.sum(y_true * y_pred)
    fn = np.sum(y_true * (1 - y_pred))
    return float((tp + smooth) / (tp + fn + smooth))


def compute_specificity_np(y_true, y_pred, threshold: float = 0.5, smooth: float = 1e-6):
    """Specificity (TNR) = TN / (TN + FP)."""
    import numpy as np
    y_true = _ensure_numpy_binary(y_true, threshold)
    y_pred = _ensure_numpy_binary(y_pred, threshold)
    tn = np.sum((1 - y_true) * (1 - y_pred))
    fp = np.sum((1 - y_true) * y_pred)
    return float((tn + smooth) / (tn + fp + smooth))


def compute_precision_np(y_true, y_pred, threshold: float = 0.5, smooth: float = 1e-6):
    """Precision = TP / (TP + FP)."""
    import numpy as np
    y_true = _ensure_numpy_binary(y_true, threshold)
    y_pred = _ensure_numpy_binary(y_pred, threshold)
    tp = np.sum(y_true * y_pred)
    fp = np.sum((1 - y_true) * y_pred)
    return float((tp + smooth) / (tp + fp + smooth))


def compute_f1_np(y_true, y_pred, threshold: float = 0.5, smooth: float = 1e-6):
    """F1 score = 2 * (Precision * Recall) / (Precision + Recall)."""
    precision = compute_precision_np(y_true, y_pred, threshold, smooth)
    sensitivity = compute_sensitivity_np(y_true, y_pred, threshold, smooth)
    return float((2.0 * precision * sensitivity + smooth) / (precision + sensitivity + smooth))


def _surface_points_from_mask(mask_binary, max_points: int = 5000, seed: int = 42):
    """Binary mask'tan sınır (surface) noktalarını döndür. Çok nokta varsa örnekle."""
    import numpy as np
    from scipy import ndimage
    eroded = ndimage.binary_erosion(mask_binary, structure=np.ones((3, 3)))
    boundary = mask_binary.astype(bool) & ~eroded
    ys, xs = np.where(boundary)
    points = np.column_stack((ys, xs)).astype(np.float64)
    if len(points) > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(points), max_points, replace=False)
        points = points[idx]
    return points


def compute_hd95_np(y_true, y_pred, threshold: float = 0.5, percentile: float = 95.0):
    """95th percentile Hausdorff Distance (HD95). Birim: piksel."""
    import numpy as np
    from scipy.spatial.distance import cdist
    y_true = _ensure_numpy_binary(y_true, threshold)
    y_pred = _ensure_numpy_binary(y_pred, threshold)
    # Boş mask kontrolü
    if np.sum(y_true) == 0 and np.sum(y_pred) == 0:
        return 0.0
    if np.sum(y_true) == 0 or np.sum(y_pred) == 0:
        return float("inf")  # veya büyük bir sayı
    pts_true = _surface_points_from_mask(y_true)
    pts_pred = _surface_points_from_mask(y_pred)
    if len(pts_true) == 0 or len(pts_pred) == 0:
        return 0.0
    # pred -> true: her pred noktasının true'ya min mesafesi
    d_pred_to_true = np.min(cdist(pts_pred, pts_true, metric="euclidean"), axis=1)
    d_true_to_pred = np.min(cdist(pts_true, pts_pred, metric="euclidean"), axis=1)
    h1 = np.percentile(d_pred_to_true, percentile)
    h2 = np.percentile(d_true_to_pred, percentile)
    return float(max(h1, h2))


def compute_asd_np(y_true, y_pred, threshold: float = 0.5):
    """Average Surface Distance (ASD). Birim: piksel. İki yüzey arası ortalama mesafe."""
    import numpy as np
    from scipy.spatial.distance import cdist
    y_true = _ensure_numpy_binary(y_true, threshold)
    y_pred = _ensure_numpy_binary(y_pred, threshold)
    if np.sum(y_true) == 0 and np.sum(y_pred) == 0:
        return 0.0
    if np.sum(y_true) == 0 or np.sum(y_pred) == 0:
        return float("inf")
    pts_true = _surface_points_from_mask(y_true)
    pts_pred = _surface_points_from_mask(y_pred)
    if len(pts_true) == 0 or len(pts_pred) == 0:
        return 0.0
    d_pred_to_true = np.min(cdist(pts_pred, pts_true, metric="euclidean"), axis=1)
    d_true_to_pred = np.min(cdist(pts_true, pts_pred, metric="euclidean"), axis=1)
    asd = (np.mean(d_pred_to_true) + np.mean(d_true_to_pred)) / 2.0
    return float(asd)


def compute_mcc_np(y_true, y_pred, threshold: float = 0.5, smooth: float = 1e-6):
    """Matthews Correlation Coefficient (MCC). [-1, 1] aralığı."""
    import numpy as np
    y_true = _ensure_numpy_binary(y_true, threshold)
    y_pred = _ensure_numpy_binary(y_pred, threshold)
    tp = np.sum(y_true * y_pred)
    tn = np.sum((1 - y_true) * (1 - y_pred))
    fp = np.sum((1 - y_true) * y_pred)
    fn = np.sum(y_true * (1 - y_pred))
    denom = np.sqrt((tp + fp + smooth) * (tp + fn + smooth) * (tn + fp + smooth) * (tn + fn + smooth))
    if denom < 1e-10:
        return 0.0
    mcc = (tp * tn - fp * fn) / denom
    return float(np.clip(mcc, -1.0, 1.0))


def compute_all_metrics_np(y_true, y_pred, threshold: float = 0.5):
    """Tüm 1. Grup metriklerini tek seferde hesapla: Dice, IoU, Sensitivity, Specificity, Precision, F1, HD95, ASD, MCC."""
    return {
        "dice": compute_dice_np(y_true, y_pred, threshold),
        "iou": compute_iou_np(y_true, y_pred, threshold),
        "sensitivity": compute_sensitivity_np(y_true, y_pred, threshold),
        "specificity": compute_specificity_np(y_true, y_pred, threshold),
        "precision": compute_precision_np(y_true, y_pred, threshold),
        "f1": compute_f1_np(y_true, y_pred, threshold),
        "hd95": compute_hd95_np(y_true, y_pred, threshold),
        "asd": compute_asd_np(y_true, y_pred, threshold),
        "mcc": compute_mcc_np(y_true, y_pred, threshold),
    }
