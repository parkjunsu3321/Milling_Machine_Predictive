from pathlib import Path
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

def data_load(split="train"):
    return pd.read_csv(DATA_DIR / f"{split}_data.csv")