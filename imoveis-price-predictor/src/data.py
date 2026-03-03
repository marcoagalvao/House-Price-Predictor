# src/data.py
import pandas as pd
from sklearn.model_selection import train_test_split
from .config import DATA_RAW, TARGET, ID_COLS, TEST_SIZE, RANDOM_STATE

def load_raw_data(path=DATA_RAW) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def make_xy(df: pd.DataFrame):
    y = df[TARGET].copy()
    X = df.drop(columns=[TARGET] + ID_COLS, errors="ignore").copy()
    return X, y

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test

def handle_structural_missing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    structural_cols = [
        "Pool QC",
        "Misc Feature",
        "Alley",
        "Fence",
        "Fireplace Qu"
    ]
    
    for col in structural_cols:
        if col in df.columns:
            df[col] = df[col].fillna("NoFeature")
    
    return df

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Idade da casa
    if "Yr Sold" in df.columns and "Year Built" in df.columns:
        df["HouseAge"] = df["Yr Sold"] - df["Year Built"]

    # Área total
    if "Total Bsmt SF" in df.columns and "Gr Liv Area" in df.columns:
        df["TotalSF"] = df["Total Bsmt SF"] + df["Gr Liv Area"]

    return df