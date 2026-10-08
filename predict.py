"""Trained model se ek meme ko classify karo.

Examples
--------
python predict.py --model multimodal --task A --image samples/train_images/326.jpg --text "Tum paas aaye yun muskuraye"
python predict.py --model image      --task B --image my_meme.jpg
python predict.py --model text       --task A --text "bhai ye toh bahut mast hai"

Pehle `python train.py --model <same model> --task <same task>` chala ke model train kar lo.
"""
import argparse

import numpy as np
import torch

import config
import features
from models import build_head


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["text", "image", "multimodal"])
    p.add_argument("--task", required=True, choices=["A", "B"])
    p.add_argument("--image", help="meme image ka path")
    p.add_argument("--text", help="meme ke andar likha text (OCR)")
    p.add_argument("--samples", action="store_true", help="samples wala trained model use karo")
    args = p.parse_args()

    if args.model in ("image", "multimodal") and not args.image:
        p.error("is model ke liye --image chahiye")
    if args.model in ("text", "multimodal") and not args.text:
        p.error("is model ke liye --text chahiye")

    tag = f"{args.model}_task{args.task}" + ("_samples" if args.samples else "")
    ckpt_path = config.OUTPUT_DIR / tag / "model.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"Trained model nahi mila: {ckpt_path}\nPehle train.py chalao.")

    device = features.get_device()
    head = build_head(args.model, args.task)
    head.load_state_dict(torch.load(ckpt_path, map_location="cpu")["state_dict"])
    head.to(device).eval()

    xs = []
    if args.model in ("text", "multimodal"):
        xs.append(features.compute_text([args.text.replace("\n", " ")], device))
    if args.model in ("image", "multimodal"):
        xs.append(features.compute_vgg([args.image], device, workers=0))
    xs = [torch.from_numpy(x).float().to(device) for x in xs]

    with torch.no_grad():
        logits = head(*xs)[0].cpu()

    if args.task == "A":
        probs = torch.softmax(logits, dim=0).numpy()
        print("\nSentiment prediction:")
        for name, pr in zip(config.TASK_A_LABELS, probs):
            print(f"  {name:<9} {pr:6.1%}")
        print(f"=> {config.TASK_A_LABELS[int(np.argmax(probs))].upper()}")
    else:
        probs = torch.sigmoid(logits).numpy()
        print("\nEmotion prediction:")
        for name, pr in zip(config.TASK_B_LABELS, probs):
            print(f"  {name:<13} {pr:6.1%}  {'YES' if pr > 0.5 else 'no'}")


if __name__ == "__main__":
    main()
