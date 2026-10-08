"""Frozen backbones (VGG16, DistilBERT-multilingual) se features nikalna + disk pe cache karna.

Original project me dono backbones frozen the (trainable=False), sirf upar ke chhote Dense layers train hote the.
Isliye backbone ka output ek hi baar nikal ke cache kar lete hain -> training laptop pe bhi tez chalti hai.
"""
from __future__ import annotations

import os

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

import config

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

VGG_FEAT_DIM = 512 * 7 * 7   # VGG16 conv output (224x224 input) flatten
TEXT_FEAT_DIM = 768          # DistilBERT [CLS] hidden size


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def default_workers() -> int:
    return 0 if os.name == "nt" else 2   # Windows pe 0 = sabse safe


# ----------------------------------------------------------------------------- loaders
def load_vgg():
    """ImageNet pretrained VGG16 (sirf conv part). Pehli baar ~528 MB weights download hote hain."""
    import torchvision
    from torchvision.models import VGG16_Weights
    return torchvision.models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features


def load_text_encoder():
    """Returns (tokenizer, model). Pehli baar ~500 MB download hota hai (HuggingFace)."""
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)
    model = AutoModel.from_pretrained(config.TEXT_MODEL_NAME)
    return tok, model


# ----------------------------------------------------------------------------- compute
class _ImageDataset(torch.utils.data.Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB").resize(
            (config.IMG_SIZE, config.IMG_SIZE), Image.BICUBIC)
        arr = (np.asarray(img, dtype=np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        return torch.from_numpy(arr.transpose(2, 0, 1))


@torch.no_grad()
def compute_vgg(paths, device, batch_size=32, workers=None, vgg=None) -> np.ndarray:
    vgg = (vgg or load_vgg()).to(device).eval()
    workers = default_workers() if workers is None else workers
    loader = torch.utils.data.DataLoader(_ImageDataset(paths), batch_size=batch_size,
                                         shuffle=False, num_workers=workers)
    out = []
    for xb in tqdm(loader, desc="VGG16 features", unit="batch"):
        out.append(vgg(xb.to(device)).flatten(1).half().cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def compute_text(texts, device, batch_size=64, encoder=None) -> np.ndarray:
    tok, model = encoder or load_text_encoder()
    model = model.to(device).eval()
    out = []
    for i in tqdm(range(0, len(texts), batch_size), desc="DistilBERT features", unit="batch"):
        enc = tok(texts[i:i + batch_size], padding=True, truncation=True,
                  max_length=config.MAX_SEQ_LEN, return_tensors="pt").to(device)
        cls = model(**enc).last_hidden_state[:, 0, :]
        out.append(cls.half().cpu().numpy())
    return np.concatenate(out)


# ----------------------------------------------------------------------------- cache
def _cached(name, ids, items, compute_fn) -> np.ndarray:
    """Har image_id ka feature cache/<name>.npz me rakhta hai; sirf missing wale compute hote hain."""
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = config.CACHE_DIR / f"{name}.npz"
    cached_ids, cached = np.array([], dtype=np.int64), None
    if path.exists():
        z = np.load(path)
        cached_ids, cached = z["ids"], z["feats"]
    index = {int(i): k for k, i in enumerate(cached_ids)}

    missing = [(i, it) for i, it in zip(ids, items) if int(i) not in index]
    if missing:
        new = compute_fn([it for _, it in missing])
        new_ids = np.array([i for i, _ in missing], dtype=np.int64)
        cached = new if cached is None else np.concatenate([cached, new])
        cached_ids = np.concatenate([cached_ids, new_ids])
        np.savez(path, ids=cached_ids, feats=cached)
        index = {int(i): k for k, i in enumerate(cached_ids)}
    else:
        print(f"[cache] {name}: {len(ids)} features cache se mile")

    return cached[[index[int(i)] for i in ids]]


def vgg_features(split_name, ids, paths, device, samples=False, **kw) -> np.ndarray:
    name = f"vgg16_{split_name}" + ("_samples" if samples else "")
    return _cached(name, ids, paths, lambda p: compute_vgg(p, device, **kw))


def text_features(split_name, ids, texts, device, samples=False, **kw) -> np.ndarray:
    name = f"distilbert_{split_name}" + ("_samples" if samples else "")
    return _cached(name, ids, texts, lambda t: compute_text(t, device, **kw))
