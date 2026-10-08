"""Saari paths aur default settings yahan hain. Zaroorat ho to sirf is file ko edit karo."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---- Data paths -----------------------------------------------------------
DATA_DIR = ROOT / "data"
TRAIN_CSV = DATA_DIR / "updated_train.csv"
VAL_CSV = DATA_DIR / "updated_val.csv"
TRAIN_IMG_DIR = DATA_DIR / "trainImages"   # <- yahan 7000 train images (0.jpg ... 6999.jpg)
VAL_IMG_DIR = DATA_DIR / "valImages"       # <- yahan 1500 val images

# Chhote sample images (repo ke saath aaye the) - sirf quick test ke liye
SAMPLE_TRAIN_IMG_DIR = ROOT / "samples" / "train_images"
SAMPLE_VAL_IMG_DIR = ROOT / "samples" / "val_images"

CACHE_DIR = ROOT / "cache"      # extracted features yahan save hote hain (dobara fast chalne ke liye)
OUTPUT_DIR = ROOT / "outputs"   # results, plots, trained model yahan milenge

# ---- Models ---------------------------------------------------------------
# Original project ka "distil_bert_base_multi" = HuggingFace ka ye model
TEXT_MODEL_NAME = "distilbert-base-multilingual-cased"
MAX_SEQ_LEN = 128
IMG_SIZE = 224
SEED = 42

# ---- Labels ---------------------------------------------------------------
TASK_A_LABELS = ["negative", "neutral", "positive"]
TASK_B_LABELS = ["humour", "sarcastic", "offensive", "motivational"]

# ---- Training defaults (original notebooks wali values) -------------------
DEFAULTS = {
    ("text", "A"):       dict(epochs=25, batch_size=32, lr=5e-5),
    ("text", "B"):       dict(epochs=25, batch_size=32, lr=5e-5),
    ("image", "A"):      dict(epochs=50, batch_size=16, lr=1e-4),
    ("image", "B"):      dict(epochs=70, batch_size=16, lr=1e-4),
    ("multimodal", "A"): dict(epochs=25, batch_size=16, lr=1e-3),
    ("multimodal", "B"): dict(epochs=25, batch_size=12, lr=1e-3),
}
