"""Chhote trainable heads (original architecture ke hisaab se).

- text      : DistilBERT [CLS] -> Dense(768, relu) -> Dropout(0.2) -> Dense(classes)
- image     : VGG16 flatten    -> Dense(1024, relu) -> Dense(classes)
- multimodal: text -> Dense(768) -> Dense(512)  +  image -> Dense(512)  ->  concat -> Dense(classes)

Output hamesha logits hota hai (Task A: CrossEntropyLoss, Task B: BCEWithLogitsLoss).
"""
import torch
from torch import nn

from features import TEXT_FEAT_DIM, VGG_FEAT_DIM


class TextHead(nn.Module):
    def __init__(self, n_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(TEXT_FEAT_DIM, TEXT_FEAT_DIM), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(TEXT_FEAT_DIM, n_out),
        )

    def forward(self, txt):
        return self.net(txt)


class ImageHead(nn.Module):
    def __init__(self, n_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(VGG_FEAT_DIM, 1024), nn.ReLU(),
            nn.Linear(1024, n_out),
        )

    def forward(self, img):
        return self.net(img)


class MultimodalHead(nn.Module):
    def __init__(self, n_out):
        super().__init__()
        self.txt = nn.Sequential(
            nn.Linear(TEXT_FEAT_DIM, TEXT_FEAT_DIM), nn.ReLU(),
            nn.Linear(TEXT_FEAT_DIM, 512), nn.ReLU(),
        )
        self.img = nn.Sequential(nn.Linear(VGG_FEAT_DIM, 512), nn.ReLU())
        self.out = nn.Linear(1024, n_out)

    def forward(self, txt, img):
        return self.out(torch.cat([self.txt(txt), self.img(img)], dim=1))


def build_head(model: str, task: str) -> nn.Module:
    n_out = 3 if task == "A" else 4
    return {"text": TextHead, "image": ImageHead, "multimodal": MultimodalHead}[model](n_out)
