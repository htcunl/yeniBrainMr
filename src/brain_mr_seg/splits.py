"""
Train/Val/Test split yönetimi
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Union


def load_split(split_file: Union[str, Path]) -> Dict[str, List[dict]]:
    """
    Split dosyasını yükle
    
    Args:
        split_file: Split JSON dosyasının yolu
    
    Returns:
        {
            "train": [{"image": "...", "mask": "..."}, ...],
            "val": [{"image": "...", "mask": "..."}, ...],
            "test": [{"image": "...", "mask": "..."}, ...]
        }
    """
    with open(split_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Eski format kontrolü
    if "splits" in data:
        return data["splits"]
    
    # Yeni format
    return {
        "train": data.get("train", []),
        "val": data.get("val", []),
        "test": data.get("test", []),
    }


def save_split(
    split_file: Union[str, Path],
    train_items: List[dict],
    val_items: List[dict],
    test_items: List[dict],
    metadata: dict = None,
):
    """Split dosyasını kaydet"""
    data = {
        "meta": metadata or {},
        "splits": {
            "train": train_items,
            "val": val_items,
            "test": test_items,
        }
    }
    
    with open(split_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
