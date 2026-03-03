# src/config.py
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = PROJECT_ROOT / "data" / "raw" / "train.csv"

TARGET = "SalePrice"
ID_COLS = ["Order", "PID"]

RANDOM_STATE = 42
TEST_SIZE = 0.2