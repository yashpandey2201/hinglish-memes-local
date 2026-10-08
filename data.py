"""CSV + images load karna, labels ko numbers me badalna (original notebooks jaisa hi logic)."""
from __future__ import annotations
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

import config

# Task A: 5 sentiment classes -> 3 (negative / neutral / positive)
TASK_A_MAP = {
    "negative": 0, "very_negative": 0,
    "neutral": 1,
    "positive": 2, "very_positive": 2,
}

# Task B: har emotion binary (0/1), multi-label
TASK_B_MAPS = {
    "humour": {"not_funny": 0, "funny": 1, "very_funny": 1, "hilarious": 1},
    "sarcastic": {"not_sarcastic": 0, "twisted_meaning": 1, "general": 1,
                  "little_sarcastic": 1, "very_sarcastic": 1, "very_twisted": 1},
    "offensive": {"not_offensive": 0, "offensive": 1, "slight": 1,
                  "very_offensive": 1, "hateful_offensive": 1},
    "motivational": {"not_motivational": 0, "motivational": 1},
}

DOWNLOAD_HELP = (
    "Images folder nahi mila / images missing hain.\n"
    "  1) Kaggle pe 'Memotion 3' (memotion-dataset) download karo ya "
    "https://aiisc.ai/defactify2/Memotion.html se.\n"
    "  2) Train images ko  data/trainImages/  me (0.jpg ... 6999.jpg) aur\n"
    "     val images ko  data/valImages/  me (0.jpg ... 1499.jpg) rakho.\n"
    "  Sirf setup test karna hai? --samples flag lagao (5-5 sample images use hongi)."
)


@dataclass
class Split:
    ids: list          # image_id list
    texts: list        # OCR text (text/multimodal ke liye) ya None
    image_paths: list  # image file paths (image/multimodal ke liye) ya None
    y: np.ndarray      # Task A: (N,) int64 | Task B: (N, 4) float32


def _encode_labels(df: pd.DataFrame, task: str) -> pd.DataFrame:
    df = df.copy()
    if task == "A":
        cols_maps = {"overall": TASK_A_MAP}
    else:
        cols_maps = {c: TASK_B_MAPS[c] for c in config.TASK_B_LABELS}

    for i, (col, mapping) in enumerate(cols_maps.items()):
        unknown = set(df[col].dropna().unique()) - set(mapping)
        if unknown:
            raise ValueError(f"CSV ke column '{col}' me unknown labels mile: {sorted(unknown)}")
        df[f"y{i}"] = df[col].map(mapping)
    return df.dropna(subset=[f"y{i}" for i in range(len(cols_maps))])


def load_split(split: str, task: str, model: str, *, samples: bool = False,
               balance: bool = False, limit: int | None = None,
               seed: int = config.SEED) -> Split:
    """split: 'train' | 'val'   task: 'A' | 'B'   model: 'text' | 'image' | 'multimodal'"""
    csv_path = config.TRAIN_CSV if split == "train" else config.VAL_CSV
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV nahi mili: {csv_path}")

    df = _encode_labels(pd.read_csv(csv_path), task)
    needs_text = model in ("text", "multimodal")
    needs_img = model in ("image", "multimodal")

    # --- text: original me text-only model OCR-less rows drop karta tha, multimodal '.' bharta tha
    if needs_text:
        if model == "text":
            df = df[df["ocr"].notna()]
        else:
            df = df.assign(ocr=df["ocr"].fillna("."))
        df = df.assign(ocr=df["ocr"].astype(str).str.replace("\n", " ", regex=False))

    # --- images  (--samples mode me text model ke liye bhi sirf wahi rows jinki sample image hai)
    if needs_img or samples:
        if samples:
            img_dir = config.SAMPLE_TRAIN_IMG_DIR if split == "train" else config.SAMPLE_VAL_IMG_DIR
        else:
            img_dir = config.TRAIN_IMG_DIR if split == "train" else config.VAL_IMG_DIR
        df = df.assign(image_path=[str(img_dir / f"{i}.jpg") for i in df["image_id"]])
        exists = df["image_path"].map(os.path.exists)
        if samples:
            df = df[exists]
        elif not exists.all():
            first_missing = df.loc[~exists, "image_path"].iloc[0]
            raise FileNotFoundError(
                f"{(~exists).sum()} images missing hain (jaise: {first_missing}).\n{DOWNLOAD_HELP}"
            )

    if df.empty:
        raise ValueError(f"'{split}' split me koi row nahi bachi (samples mode me ocr/image check karo).")

    # --- Task A train: classes ko equal karna (original text/image notebooks me tha)
    if balance and split == "train" and task == "A":
        min_count = df["y0"].value_counts().min()
        df = pd.concat([g.sample(min_count, random_state=seed) for _, g in df.groupby("y0")])

    if split == "train":
        df = df.sample(frac=1, random_state=seed)
    if limit is not None and limit < len(df):
        df = df.sample(n=limit, random_state=seed)
    df = df.reset_index(drop=True)

    if task == "A":
        y = df["y0"].to_numpy(dtype=np.int64)
    else:
        y = df[[f"y{i}" for i in range(4)]].to_numpy(dtype=np.float32)

    return Split(
        ids=df["image_id"].astype(int).tolist(),
        texts=df["ocr"].tolist() if needs_text else None,
        image_paths=df["image_path"].tolist() if needs_img else None,
        y=y,
    )


def check_data() -> None:
    """Quick report: kaunsi files mil rahi hain."""
    def count(p):
        return len(list(p.glob("*.jpg"))) if p.exists() else 0

    print("CSV train :", "OK" if config.TRAIN_CSV.exists() else "MISSING", config.TRAIN_CSV)
    print("CSV val   :", "OK" if config.VAL_CSV.exists() else "MISSING", config.VAL_CSV)
    print(f"trainImages: {count(config.TRAIN_IMG_DIR)} / 7000 images  ({config.TRAIN_IMG_DIR})")
    print(f"valImages  : {count(config.VAL_IMG_DIR)} / 1500 images  ({config.VAL_IMG_DIR})")
    print(f"samples    : {count(config.SAMPLE_TRAIN_IMG_DIR)} train + {count(config.SAMPLE_VAL_IMG_DIR)} val")
