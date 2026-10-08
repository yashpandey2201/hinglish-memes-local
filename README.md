# Hinglish Memes Classification - VS Code (local) version

Original repo ke Keras/Kaggle notebooks ko **PyTorch + HuggingFace** me convert kiya gaya hai, taaki ye
Windows / Mac / Linux pe VS Code me seedha chale (original me `keras-nlp` use hota tha jo Windows pe install nahi hota).

**Task A** = meme ka sentiment (negative / neutral / positive)
**Task B** = emotions (humour, sarcastic, offensive, motivational) - multi-label

| Model | Kya use karta hai |
|---|---|
| `text` | DistilBERT multilingual (meme ka OCR text) |
| `image` | VGG16 (meme image) |
| `multimodal` | dono combine |

---
## 1. Setup (ek baar)

Python 3.9 - 3.12 chahiye. VS Code me **File > Open Folder** se `hinglish_memes_local` folder kholo, phir terminal (`Ctrl + ~`):

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
(PowerShell me "running scripts is disabled" aaye to: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`)

**Mac / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Phir VS Code me `Ctrl+Shift+P` > **Python: Select Interpreter** > `.venv` wala choose karo.

NVIDIA GPU hai to GPU wala PyTorch lagao (training aur feature extraction kaafi tez): https://pytorch.org/get-started/locally/

## 2. Dataset

CSV files `data/` me pehle se hain. **Images nahi hain** (repo me sirf 10 samples the). Memotion 3 dataset download karo
(Kaggle pe "memotion-dataset" ya https://aiisc.ai/defactify2/Memotion.html) aur rakho:

```
data/trainImages/0.jpg ... 6999.jpg     (7000 images)
data/valImages/0.jpg   ... 1499.jpg     (1500 images)
```
Check karne ke liye: `python train.py --check`

## 3. Chalana

**Pehle quick test** (sirf 5+5 sample images, ~1 min, setup verify karne ke liye; pehli baar models download honge, ~1 GB):
```bash
python train.py --model text --task A --samples --epochs 2
python train.py --model multimodal --task A --samples --epochs 2
```

**Poora training** (full dataset ke saath):
```bash
python train.py --model text       --task A
python train.py --model image      --task A
python train.py --model multimodal --task A
python train.py --model text       --task B     # image / multimodal ke liye bhi same
```
VS Code me `Run and Debug` panel (Ctrl+Shift+D) me bane hue configs bhi hain, bas dropdown se chuno aur Play dabao.

Useful options: `--epochs N`, `--batch-size N`, `--lr X`, `--limit N` (har split me N rows), `--no-balance`.

**Kitna time?** Pehli run me backbone features nikalte hain (8500 images VGG16 se): GPU pe ~2-5 min, CPU pe ~20-40 min.
Ye `cache/` me save ho jaata hai, dobara chalane pe training sirf kuch minute leti hai (poora 25-70 epochs bhi).

## 4. Results kahan milenge

`outputs/<model>_task<A|B>/` me: `metrics.json` (accuracy / precision / recall / F1), `history.png` (loss & accuracy graph), `model.pt` (trained model).

## 5. Apne meme pe predict

```bash
python predict.py --model multimodal --task A --image my_meme.jpg --text "meme me jo likha hai"
python predict.py --model image --task B --image my_meme.jpg
```
(Pehle usi model + task ka `train.py` chal chuka hona chahiye.)

---
## Original notebooks se kya alag hai (jaanna zaroori)

- **Framework**: Keras/keras-nlp -> PyTorch/transformers. Architecture wahi hai (frozen DistilBERT-multilingual + frozen VGG16 + chhote Dense heads), epochs/batch size/learning rate bhi original wale defaults. Numbers original README se thode alag aa sakte hain (random seed, framework, preprocessing).
- **Backbones frozen the**, isliye unke features ek baar nikaal ke cache kiye hain. Training mathematically wahi hai, bas bahut tez.
- Original DistilBERT notebook me pehle 1 epoch poore model ko fine-tune karta tha, phir freeze karta tha; yahan seedha frozen backbone ke saath train hota hai.
- Original multimodal Task A me softmax output ke saath `from_logits=True` loss laga tha (galti); yahan sahi loss (CrossEntropy on logits) use hua hai.
- Validation set pe hi evaluate hota hai (test set repo me nahi hai), jaise original me.

## Folder structure
```
train.py      training + evaluation        predict.py   single meme predict
config.py     paths & defaults             data.py      CSV/labels loading
features.py   VGG16 / DistilBERT features   models.py    classification heads
data/         CSVs + (aapki) images        samples/     10 sample images
original_notebooks/   original .ipynb files (reference ke liye)
```
