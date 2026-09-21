from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    if "date_recorded" in data:
        date = pd.to_datetime(data["date_recorded"], errors="coerce")
        data["recorded_year"] = date.dt.year
        data["recorded_month"] = date.dt.month
        data = data.drop(columns="date_recorded")
    if {"recorded_year", "construction_year"}.issubset(data.columns):
        year = data["construction_year"].replace(0, np.nan)
        data["pump_age"] = (data["recorded_year"] - year).clip(lower=0)
    if "population" in data:
        population = data["population"].fillna(0)
        data["has_population"] = (population > 0).astype("int8")
    if "amount_tsh" in data:
        data["is_zero_tsh"] = data["amount_tsh"].fillna(0).eq(0).astype("int8")
    if {"amount_tsh", "population"}.issubset(data.columns):
        denom = data["population"].replace(0, np.nan)
        data["water_per_person"] = (data["amount_tsh"] / denom).replace([np.inf, -np.inf], np.nan)
    if "id" in data:
        data = data.drop(columns="id")
    return data


def build_model(features: pd.DataFrame) -> Pipeline:
    categorical = features.select_dtypes(exclude=np.number).columns.tolist()
    numeric = features.select_dtypes(include=np.number).columns.tolist()
    preprocessing = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), numeric),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore", min_frequency=5)),
                    ]
                ),
                categorical,
            ),
        ]
    )
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=42,
    )
    xgb = XGBClassifier(
        n_estimators=900,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        gamma=0.05,
        min_child_weight=2,
        reg_alpha=0.05,
        reg_lambda=1.0,
        objective="multi:softprob",
        eval_metric="mlogloss",
        n_jobs=-1,
        random_state=42,
    )
    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("xgb", xgb)],
        voting="soft",
        weights=[1, 2],
    )
    return Pipeline([("preprocess", preprocessing), ("model", ensemble)])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-values", type=Path, required=True)
    parser.add_argument("--train-labels", type=Path, required=True)
    parser.add_argument("--test-values", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_values = pd.read_csv(args.train_values)
    train_labels = pd.read_csv(args.train_labels)
    test_values = pd.read_csv(args.test_values)
    merged = train_values.merge(train_labels[["id", "status_group"]], on="id", validate="one_to_one")
    x = engineer_features(merged.drop(columns="status_group"))
    y = merged["status_group"]
    x_train, x_valid, y_train, y_valid = train_test_split(
        x, y, test_size=0.2, stratify=y, random_state=42
    )
    model = build_model(x_train)
    model.fit(x_train, y_train)
    print(f"Validation accuracy: {model.score(x_valid, y_valid):.4f}")
    model.fit(x, y)
    predictions = model.predict(engineer_features(test_values))
    pd.DataFrame({"id": test_values["id"], "status_group": predictions}).to_csv(
        args.output, index=False
    )
    print(f"Submission written to {args.output}")


if __name__ == "__main__":
    main()

