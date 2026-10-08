"""Train + evaluate.

Examples
--------
python train.py --check                                   # setup / data check
python train.py --model text       --task A --samples --epochs 2   # 1-minute test (sample images)
python train.py --model text       --task A               # text only, sentiment
python train.py --model image      --task B               # image only, emotions
python train.py --model multimodal --task A               # text + image, sentiment
"""
from __future__ import annotations

import argparse
import json
import random
import time

import matplotlib
matplotlib.use("Agg")                      # window nahi kholta, plot file me save karta hai
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn

import config
import data
import features
from models import build_head


# ----------------------------------------------------------------------------- args
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Hinglish Memes Classification (Memotion 3)")
    p.add_argument("--model", choices=["text", "image", "multimodal"], help="kaun sa model")
    p.add_argument("--task", choices=["A", "B"], help="A = sentiment, B = emotions")
    p.add_argument("--epochs", type=int, help="default: original notebook wali value")
    p.add_argument("--batch-size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--limit", type=int, help="har split me max itni rows (jaldi test ke liye)")
    p.add_argument("--samples", action="store_true",
                   help="sirf samples/ folder ki 5+5 images use karo (setup test)")
    p.add_argument("--balance", action=argparse.BooleanOptionalAction, default=None,
                   help="Task A train classes equal karo (default: text/image me ON, multimodal me OFF)")
    p.add_argument("--workers", type=int, default=None, help="image loading workers")
    p.add_argument("--seed", type=int, default=config.SEED)
    p.add_argument("--check", action="store_true", help="sirf environment/data check karo")
    args = p.parse_args(argv)
    if not args.check and not (args.model and args.task):
        p.error("--model aur --task dono do (ya --check use karo)")
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ----------------------------------------------------------------------------- data -> tensors
def gather(split, args, device, balance):
    s = data.load_split(split, args.task, args.model, samples=args.samples,
                        balance=balance, limit=args.limit, seed=args.seed)
    xs = []
    if args.model in ("text", "multimodal"):
        xs.append(features.text_features(split, s.ids, s.texts, device, samples=args.samples))
    if args.model in ("image", "multimodal"):
        xs.append(features.vgg_features(split, s.ids, s.image_paths, device,
                                        samples=args.samples, workers=args.workers))
    xs = [torch.from_numpy(x) for x in xs]            # float16, CPU
    y = torch.from_numpy(s.y)
    print(f"[{split}] {len(y)} samples")
    return xs, y


# ----------------------------------------------------------------------------- train / eval
def run_epoch(model, xs, y, batch_size, device, loss_fn, task, optimizer=None):
    training = optimizer is not None
    model.train(training)
    n = len(y)
    order = torch.randperm(n) if training else torch.arange(n)
    total_loss, correct, count = 0.0, 0.0, 0
    all_logits = []
    with torch.set_grad_enabled(training):
        for i in range(0, n, batch_size):
            idx = order[i:i + batch_size]
            xb = [x[idx].to(device).float() for x in xs]
            yb = y[idx].to(device)
            logits = model(*xb)
            loss = loss_fn(logits, yb)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(idx)
            if task == "A":
                correct += (logits.argmax(1) == yb).float().sum().item()
                count += len(idx)
            else:
                correct += ((logits > 0) == (yb > 0.5)).float().sum().item()
                count += yb.numel()
            if not training:
                all_logits.append(logits.detach().cpu())
    out_logits = torch.cat(all_logits).numpy() if all_logits else None
    return total_loss / n, correct / count, out_logits


def compute_metrics(task, y_true, logits):
    """Original notebooks jaisa: weighted average accuracy / precision / recall / F1."""
    def block(yt, yp):
        return dict(
            accuracy=float(accuracy_score(yt, yp)),
            precision=float(precision_score(yt, yp, average="weighted", zero_division=0)),
            recall=float(recall_score(yt, yp, average="weighted", zero_division=0)),
            f1=float(f1_score(yt, yp, average="weighted", zero_division=0)),
        )

    if task == "A":
        pred = logits.argmax(1)
        m = block(y_true, pred)
        m["macro_f1"] = float(f1_score(y_true, pred, average="macro", zero_division=0))
        return m

    pred = (logits > 0).astype(int)                    # sigmoid(logit) > 0.5
    y_true = y_true.astype(int)
    per_label = {name: block(y_true[:, j], pred[:, j]) for j, name in enumerate(config.TASK_B_LABELS)}
    avg = {k: float(np.mean([v[k] for v in per_label.values()])) for k in ("accuracy", "precision", "recall", "f1")}
    return {"per_label": per_label, "average": avg}


def print_metrics(task, m):
    print("\n===== Validation results =====")
    if task == "A":
        print(f"Accuracy : {m['accuracy']:.4f}\nPrecision: {m['precision']:.4f}\n"
              f"Recall   : {m['recall']:.4f}\nF1 (weighted): {m['f1']:.4f}   F1 (macro): {m['macro_f1']:.4f}")
    else:
        for name, v in m["per_label"].items():
            print(f"{name:<13} acc={v['accuracy']:.4f}  prec={v['precision']:.4f}  "
                  f"rec={v['recall']:.4f}  f1={v['f1']:.4f}")
        a = m["average"]
        print(f"{'AVERAGE':<13} acc={a['accuracy']:.4f}  prec={a['precision']:.4f}  "
              f"rec={a['recall']:.4f}  f1={a['f1']:.4f}")


def save_plots(history, out_dir, title):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(history["loss"], label="train")
    ax[0].plot(history["val_loss"], label="val")
    ax[0].set(title=f"{title} - Loss", xlabel="epoch")
    ax[1].plot(history["acc"], label="train")
    ax[1].plot(history["val_acc"], label="val")
    ax[1].set(title=f"{title} - Accuracy", xlabel="epoch")
    for a in ax:
        a.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "history.png", dpi=120)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main(argv=None):
    args = parse_args(argv)
    device = features.get_device()
    print(f"PyTorch {torch.__version__} | device: {device}")
    if args.check:
        data.check_data()
        return

    set_seed(args.seed)
    d = config.DEFAULTS[(args.model, args.task)]
    epochs = args.epochs or d["epochs"]
    batch_size = args.batch_size or d["batch_size"]
    lr = args.lr or d["lr"]
    balance = args.balance
    if balance is None:
        balance = (args.model in ("text", "image") and args.task == "A" and not args.samples)

    print(f"model={args.model} task={args.task} epochs={epochs} batch={batch_size} lr={lr} "
          f"balance={balance} samples={args.samples}")

    x_tr, y_tr = gather("train", args, device, balance)
    x_va, y_va = gather("val", args, device, False)

    model = build_head(args.model, args.task).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss() if args.task == "A" else nn.BCEWithLogitsLoss()
    if args.task == "B":
        y_tr, y_va_t = y_tr.float(), y_va.float()
    else:
        y_va_t = y_va

    history = {"loss": [], "acc": [], "val_loss": [], "val_acc": []}
    t0 = time.time()
    for ep in range(1, epochs + 1):
        tl, ta, _ = run_epoch(model, x_tr, y_tr, batch_size, device, loss_fn, args.task, optimizer)
        vl, va, logits = run_epoch(model, x_va, y_va_t, batch_size, device, loss_fn, args.task)
        for k, v in zip(history, (tl, ta, vl, va)):
            history[k].append(v)
        print(f"Epoch {ep:>3}/{epochs}  loss={tl:.4f} acc={ta:.4f}  |  val_loss={vl:.4f} val_acc={va:.4f}")
    print(f"Training time: {time.time() - t0:.1f}s")

    metrics = compute_metrics(args.task, y_va.numpy(), logits)
    print_metrics(args.task, metrics)

    tag = f"{args.model}_task{args.task}" + ("_samples" if args.samples else "")
    out_dir = config.OUTPUT_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    save_plots(history, out_dir, tag)
    torch.save({"state_dict": model.state_dict(), "model": args.model, "task": args.task},
               out_dir / "model.pt")
    print(f"\nSaved -> {out_dir}  (metrics.json, history.png, model.pt)")


if __name__ == "__main__":
    main()
