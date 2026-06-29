#!/usr/bin/env python
"""
Önceki splits.json içindeki TEST kümesini (sıra dahil) birebir koruyarak,
kalan örnekleri train:val = 8:1 oranında böler (kalan = eski train ∪ eski val).

Not: Toplam örnek sayısı N iken test sabit T=613 ise küresel tam %80/%10/%10
(train=N*0.8, val=N*0.1, test=N*0.1) aynı anda sağlanamaz; test kümesini
değiştirmeden train/val paylarını ayarlamanın standart yolu, (N-T) havuzunu
8:1 bölmektir (≈ eğitim verisinin %80'i, validasyonun %10'u klasik raporlara yakın).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from brain_mr_seg.splits import load_split, save_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-split", type=str, default="splits.json")
    parser.add_argument("--out", type=str, default="splits_80_10_10_locked_test.json")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base_path = Path(args.baseline_split)
    data = json.loads(base_path.read_text(encoding="utf-8"))
    splits = load_split(base_path)

    test_items = splits["test"]
    pool = list(splits["train"]) + list(splits["val"])
    n_pool = len(pool)
    # 8:1 train:val within pool (sum = n_pool)
    n_train = (n_pool * 8) // 9
    n_val = n_pool - n_train

    rng = np.random.default_rng(args.seed)
    order = np.arange(n_pool)
    rng.shuffle(order)
    shuffled = [pool[i] for i in order]
    train_new = shuffled[:n_train]
    val_new = shuffled[n_train:]

    meta = {
        "data_dir": (data.get("meta") or {}).get("data_dir"),
        "seed": args.seed,
        "baseline_split": str(base_path.resolve()),
        "strategy": "test_unchanged_train_val_8to1_pool",
        "ratios_requested": {"train": 0.8, "val": 0.1, "test": 0.1},
        "note_tr": (
            "Test kümesi baseline ile aynı uzunlukta ve aynı sırada tutuldu. "
            "Global tam %80/%10/%10, test=613 sabitken mümkün olmadığı için "
            f"kalan {n_pool} örnek 8:1 (train:val) bölündü."
        ),
        "counts": {
            "train": len(train_new),
            "val": len(val_new),
            "test": len(test_items),
        },
    }

    out_path = Path(args.out)
    save_split(out_path, train_new, val_new, test_items, metadata=meta)
    print(f"Yazıldı: {out_path.resolve()}")
    print(f"  train={len(train_new)}  val={len(val_new)}  test={len(test_items)}")
    print("Test listesi baseline ile özdeş mi (içerik):", _same_list(splits["test"], test_items))


def _same_list(a, b) -> bool:
    if len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x != y:
            return False
    return True


if __name__ == "__main__":
    main()
