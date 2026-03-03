# src/train.py
from asyncio.windows_utils import pipe
import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, KFold

from .data import handle_structural_missing
from .data import load_raw_data, make_xy, split_data, handle_structural_missing, add_features

from sklearn.model_selection import KFold, cross_val_score
from xgboost import XGBRegressor

from .transformers import QuantileClipper

def build_baseline_pipeline(X):
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object"]).columns

    numeric_pipe = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("clip", QuantileClipper(lower_q=0.01, upper_q=0.99)),
])

    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop"
    )

    model = Ridge(alpha=1.0)

    pipe = Pipeline(steps=[
        ("prep", preprocessor),
        ("model", model)
    ])
    return pipe

def build_xgb_pipeline(X):
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = X.select_dtypes(include=["object"]).columns

    numeric_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("clip", QuantileClipper(lower_q=0.01, upper_q=0.99)),
    ])

    categorical_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop"
    )

    # XGBoost regressão (tabular)
    model = XGBRegressor(
        n_estimators=800,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        objective="reg:squarederror",
    )

    pipe = Pipeline(steps=[
        ("prep", preprocessor),
        ("model", model)
    ])

    return pipe

def evaluate(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)  # sem squared=
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    return {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2)}

def cross_validate_model(pipe, X, y):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # usamos MAE como métrica principal
    scores = cross_val_score(
        pipe,
        X,
        y,
        scoring="neg_mean_absolute_error",
        cv=kf
    )

    mae_scores = -scores
    return mae_scores

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def main():
    df = load_raw_data()
    df = handle_structural_missing(df)
    df = add_features(df)

    X, y = make_xy(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    y_train_log = np.log1p(y_train)

    xgb_pipe = build_xgb_pipeline(X_train)
    xgb_pipe.fit(X_train, y_train_log)

    # prever
    preds_log = xgb_pipe.predict(X_test)
    preds = np.expm1(preds_log)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print("\n=== Avaliação na Escala Real ===")
    print(f"MAE : {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2  : {r2:.4f}")

    joblib.dump(xgb_pipe, "models/xgb_final.pkl")
    print("\nModelo salvo em models/xgb_final.pkl")

if __name__ == "__main__":
    main()