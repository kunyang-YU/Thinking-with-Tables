from __future__ import annotations

"""
Single-modality baselines for the four datasets in `data/multimodalQA-used`.

- Tabular baselines: train on `train.csv`, predict `test.csv`
  with RandomForest / LightGBM / XGBoost.
- Image baselines: load the provided fine-tuned ResNet50 weights and evaluate
  directly on the test split without retraining.

Notes
- `skinca` uses `diagnostic` as the tabular target because the provided image
  weights and folder labels are six-way diagnostic classification.
- The adoption/skinca image baselines evaluate the folder-based test split
  shipped with the ResNet checkpoints.
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms


REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = REPO_ROOT / "data" / "multimodalQA-used"
WEIGHT_ROOT = Path(__file__).resolve().parent / "ResNet_weights"


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    task_type: str
    target_col: str
    id_col: str
    train_csv: Path
    test_csv: Path
    tabular_drop_cols: tuple[str, ...]
    image_eval_type: str
    image_root: Path
    image_test_root: Path | None
    image_col: str | None
    image_suffix: str
    resnet_weight_path: Path
    num_classes: int | None = None


DATASET_CONFIGS: dict[str, DatasetConfig] = {
    "adoption": DatasetConfig(
        name="adoption",
        task_type="classification",
        target_col="AdoptionSpeed",
        id_col="PetID",
        train_csv=DATA_ROOT / "adoption" / "train.csv",
        test_csv=DATA_ROOT / "adoption" / "test.csv",
        tabular_drop_cols=("PetID", "RescuerID", "Description", "Name"),
        image_eval_type="folder_classification",
        image_root=DATA_ROOT / "adoption" / "images",
        image_test_root=DATA_ROOT / "adoption" / "images" / "test",
        image_col=None,
        image_suffix="",
        resnet_weight_path=WEIGHT_ROOT / "adoption" / "resnet" / "resnet50_finetuned.pth",
        num_classes=5,
    ),
    "pawpularity": DatasetConfig(
        name="pawpularity",
        task_type="regression",
        target_col="Pawpularity",
        id_col="Id",
        train_csv=DATA_ROOT / "pawpularity" / "train.csv",
        test_csv=DATA_ROOT / "pawpularity" / "test.csv",
        tabular_drop_cols=("Id",),
        image_eval_type="csv_regression",
        image_root=DATA_ROOT / "pawpularity" / "dataset" / "images",
        image_test_root=None,
        image_col="Id",
        image_suffix=".jpg",
        resnet_weight_path=WEIGHT_ROOT / "pawpularity" / "resnet50" / "resnet50_regression.pth",
    ),
    "paintings": DatasetConfig(
        name="paintings",
        task_type="regression",
        target_col="price",
        id_col="image_url",
        train_csv=DATA_ROOT / "paintings" / "train.csv",
        test_csv=DATA_ROOT / "paintings" / "test.csv",
        tabular_drop_cols=("image_url", "styles"),
        image_eval_type="csv_regression",
        image_root=DATA_ROOT / "paintings" / "dataset" / "images",
        image_test_root=None,
        image_col="image_url",
        image_suffix="",
        resnet_weight_path=WEIGHT_ROOT / "paintings" / "resnet50" / "resnet50_regression.pth",
    ),
    "skinca": DatasetConfig(
        name="skinca",
        task_type="classification",
        target_col="diagnostic",
        id_col="img_id",
        train_csv=DATA_ROOT / "skinca" / "train.csv",
        test_csv=DATA_ROOT / "skinca" / "test.csv",
        tabular_drop_cols=("patient_id", "lesion_id", "img_id"),
        image_eval_type="folder_classification",
        image_root=DATA_ROOT / "skinca" / "dataset",
        image_test_root=DATA_ROOT / "skinca" / "dataset" / "test",
        image_col=None,
        image_suffix="",
        resnet_weight_path=WEIGHT_ROOT / "skinca" / "resnet" / "resnet50_finetuned.pth",
        num_classes=6,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-modality baselines for the four multimodalQA-used datasets."
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        choices=["all", *DATASET_CONFIGS.keys()],
        default=["all"],
        help="Datasets to evaluate. Default: all.",
    )
    parser.add_argument(
        "--modality",
        nargs="+",
        choices=["all", "tabular", "image"],
        default=["all"],
        help="Which baseline families to run. Default: all.",
    )
    parser.add_argument(
        "--tabular-models",
        nargs="+",
        choices=["random_forest", "lightgbm", "xgboost"],
        default=["random_forest", "lightgbm", "xgboost"],
        help="Tabular models to train/evaluate.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "evaluation" / "single_modality_baselines_results.json",
        help="Where to write the summary JSON.",
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=None,
        help="Optional directory for saving per-run prediction CSVs.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Image evaluation batch size.")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Execution device. Image evaluation uses this directly; XGBoost/LightGBM use GPU when requested and supported.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Dataloader workers for image evaluation. Windows default is 0.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def selected_datasets(args: argparse.Namespace) -> list[DatasetConfig]:
    if "all" in args.dataset:
        return [DATASET_CONFIGS[name] for name in DATASET_CONFIGS]
    return [DATASET_CONFIGS[name] for name in args.dataset]


def selected_modalities(args: argparse.Namespace) -> set[str]:
    if "all" in args.modality:
        return {"tabular", "image"}
    return set(args.modality)


def build_onehot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = features.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [col for col in features.columns if col not in numeric_cols]

    transformers: list[tuple[str, Any, list[str]]] = []
    if numeric_cols:
        transformers.append(
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                numeric_cols,
            )
        )
    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", build_onehot_encoder()),
                    ]
                ),
                categorical_cols,
            )
        )
    return ColumnTransformer(transformers=transformers, remainder="drop")


def build_tabular_model(
    model_name: str,
    task_type: str,
    n_classes: int | None,
    seed: int,
    use_gpu: bool,
) -> Any:
    if model_name == "random_forest":
        if task_type == "classification":
            return RandomForestClassifier(
                n_estimators=500,
                random_state=seed,
                n_jobs=-1,
                class_weight="balanced_subsample",
            )
        return RandomForestRegressor(
            n_estimators=500,
            random_state=seed,
            n_jobs=-1,
        )

    if model_name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier, LGBMRegressor
        except ImportError as exc:
            raise ImportError("lightgbm is required for the LightGBM baseline.") from exc

        common_kwargs = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "random_state": seed,
        }
        if use_gpu:
            common_kwargs["device_type"] = "gpu"
        if task_type == "classification":
            if n_classes == 2:
                return LGBMClassifier(objective="binary", **common_kwargs)
            return LGBMClassifier(objective="multiclass", num_class=n_classes, **common_kwargs)
        return LGBMRegressor(objective="regression", **common_kwargs)

    if model_name == "xgboost":
        try:
            from xgboost import XGBClassifier, XGBRegressor
        except ImportError as exc:
            raise ImportError("xgboost is required for the XGBoost baseline.") from exc

        common_kwargs = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "random_state": seed,
            "n_jobs": -1,
            "tree_method": "hist",
            "verbosity": 0,
        }
        if use_gpu:
            common_kwargs["device"] = "cuda"
        if task_type == "classification":
            if n_classes == 2:
                return XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    **common_kwargs,
                )
            return XGBClassifier(
                objective="multi:softprob",
                num_class=n_classes,
                eval_metric="mlogloss",
                **common_kwargs,
            )
        return XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            **common_kwargs,
        )

    raise ValueError(f"Unsupported tabular model: {model_name}")


def postprocess_regression_predictions(dataset_name: str, preds: np.ndarray) -> np.ndarray:
    preds = np.asarray(preds, dtype=float)
    if dataset_name == "pawpularity":
        return np.clip(preds, 0.0, 100.0)
    if dataset_name == "paintings":
        return np.clip(preds, 0.0, None)
    return preds


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    value_range = float(np.max(y_true) - np.min(y_true))
    if value_range == 0.0:
        nrmse = 0.0 if rmse == 0.0 else float("inf")
    else:
        nrmse = rmse / value_range
    return {
        "nrmse": float(nrmse),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def compute_metrics(task_type: str, dataset_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    if task_type == "classification":
        return classification_metrics(y_true.astype(int), y_pred.astype(int))
    preds = postprocess_regression_predictions(dataset_name, y_pred.astype(float))
    return regression_metrics(y_true.astype(float), preds)


def prepare_tabular_data(config: DatasetConfig) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    train_df = pd.read_csv(config.train_csv)
    test_df = pd.read_csv(config.test_csv)

    drop_cols = [config.target_col, *config.tabular_drop_cols]
    X_train = train_df.drop(columns=drop_cols, errors="ignore")
    X_test = test_df.drop(columns=drop_cols, errors="ignore")

    if config.task_type == "classification":
        y_train = train_df[config.target_col].astype(int).to_numpy()
        y_test = test_df[config.target_col].astype(int).to_numpy()
    else:
        y_train = train_df[config.target_col].astype(float).to_numpy()
        y_test = test_df[config.target_col].astype(float).to_numpy()
    return X_train, X_test, y_train, y_test


def save_prediction_frame(
    ids: pd.Series | list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Path,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "id": ids,
            "y_true": y_true,
            "y_pred": y_pred,
        }
    )
    frame.to_csv(save_path, index=False)


def run_tabular_baseline(
    config: DatasetConfig,
    model_name: str,
    seed: int,
    use_gpu: bool,
    predictions_dir: Path | None,
) -> dict[str, Any]:
    train_df = pd.read_csv(config.train_csv)
    test_df = pd.read_csv(config.test_csv)
    X_train, X_test, y_train, y_test = prepare_tabular_data(config)
    n_classes = int(np.unique(y_train).size) if config.task_type == "classification" else None

    pipeline = Pipeline(
        steps=[
            ("preprocess", build_preprocessor(X_train)),
            ("model", build_tabular_model(model_name, config.task_type, n_classes, seed, use_gpu)),
        ]
    )
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    if config.task_type == "classification":
        y_pred = np.asarray(y_pred, dtype=int)
    else:
        y_pred = postprocess_regression_predictions(config.name, np.asarray(y_pred, dtype=float))

    metrics = compute_metrics(config.task_type, config.name, y_test, y_pred)
    result = {
        "status": "ok",
        "task_type": config.task_type,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "device": "gpu" if (use_gpu and model_name in {"lightgbm", "xgboost"}) else "cpu",
        "metrics": metrics,
    }

    if predictions_dir is not None:
        save_prediction_frame(
            ids=test_df[config.id_col].astype(str),
            y_true=y_test,
            y_pred=y_pred,
            save_path=predictions_dir / f"{config.name}_tabular_{model_name}.csv",
        )
    return result


def build_resnet50_classifier(num_classes: int) -> nn.Module:
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_resnet50_regressor(target_dim: int = 1) -> nn.Module:
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, target_dim)
    return model


def build_eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


class CSVImageRegressionDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        image_root: Path,
        image_col: str,
        target_col: str,
        transform: transforms.Compose,
        image_suffix: str = "",
    ) -> None:
        self.frame = frame.reset_index(drop=True).copy()
        self.image_root = image_root
        self.image_col = image_col
        self.target_col = target_col
        self.transform = transform
        self.image_suffix = image_suffix

    def __len__(self) -> int:
        return len(self.frame)

    def _image_path(self, raw_value: Any) -> Path:
        raw_str = str(raw_value)
        raw_path = Path(raw_str)
        filename = raw_path.name if raw_path.suffix else f"{raw_str}{self.image_suffix}"
        return self.image_root / filename

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        row = self.frame.iloc[index]
        image_path = self._image_path(row[self.image_col])
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)
        target_tensor = torch.tensor(float(row[self.target_col]), dtype=torch.float32)
        return image_tensor, target_tensor, str(row[self.image_col])


def evaluate_image_classification(
    config: DatasetConfig,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    predictions_dir: Path | None,
) -> dict[str, Any]:
    if config.image_test_root is None or config.num_classes is None:
        raise ValueError(f"{config.name} is missing image classification metadata.")

    dataset = datasets.ImageFolder(str(config.image_test_root), transform=build_eval_transform())
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_resnet50_classifier(config.num_classes)
    state_dict = torch.load(config.resnet_weight_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    y_true: list[int] = []
    y_pred: list[int] = []
    image_ids: list[str] = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(device)
            logits = model(images)
            preds = logits.argmax(dim=1).cpu().numpy()

            y_true.extend(targets.numpy().tolist())
            y_pred.extend(preds.tolist())

    image_ids = [Path(path).name for path, _ in dataset.samples]
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    metrics = classification_metrics(y_true_arr, y_pred_arr)

    if predictions_dir is not None:
        save_prediction_frame(
            ids=image_ids,
            y_true=y_true_arr,
            y_pred=y_pred_arr,
            save_path=predictions_dir / f"{config.name}_image_resnet50.csv",
        )

    return {
        "status": "ok",
        "task_type": config.task_type,
        "n_test": int(len(dataset)),
        "device": str(device),
        "metrics": metrics,
        "weight_path": str(config.resnet_weight_path),
    }


def evaluate_image_regression(
    config: DatasetConfig,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    predictions_dir: Path | None,
) -> dict[str, Any]:
    if config.image_col is None:
        raise ValueError(f"{config.name} is missing image regression metadata.")

    test_df = pd.read_csv(config.test_csv)
    dataset = CSVImageRegressionDataset(
        frame=test_df,
        image_root=config.image_root,
        image_col=config.image_col,
        target_col=config.target_col,
        transform=build_eval_transform(),
        image_suffix=config.image_suffix,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_resnet50_regressor(target_dim=1)
    state_dict = torch.load(config.resnet_weight_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    y_true: list[float] = []
    y_pred: list[float] = []
    sample_ids: list[str] = []

    with torch.no_grad():
        for images, targets, ids in loader:
            images = images.to(device)
            preds = model(images).squeeze(-1).cpu().numpy()

            y_true.extend(targets.numpy().tolist())
            y_pred.extend(np.asarray(preds, dtype=float).tolist())
            sample_ids.extend(list(ids))

    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = postprocess_regression_predictions(config.name, np.asarray(y_pred, dtype=float))
    metrics = regression_metrics(y_true_arr, y_pred_arr)

    if predictions_dir is not None:
        save_prediction_frame(
            ids=sample_ids,
            y_true=y_true_arr,
            y_pred=y_pred_arr,
            save_path=predictions_dir / f"{config.name}_image_resnet50.csv",
        )

    return {
        "status": "ok",
        "task_type": config.task_type,
        "n_test": int(len(dataset)),
        "device": str(device),
        "metrics": metrics,
        "weight_path": str(config.resnet_weight_path),
    }


def run_image_baseline(
    config: DatasetConfig,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    predictions_dir: Path | None,
) -> dict[str, Any]:
    if config.image_eval_type == "folder_classification":
        return evaluate_image_classification(config, device, batch_size, num_workers, predictions_dir)
    if config.image_eval_type == "csv_regression":
        return evaluate_image_regression(config, device, batch_size, num_workers, predictions_dir)
    raise ValueError(f"Unsupported image eval type: {config.image_eval_type}")


def safe_run(fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    try:
        return fn()
    except Exception as exc:
        return {
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
        }


def main() -> None:
    args = parse_args()
    modalities = selected_modalities(args)
    datasets_to_run = selected_datasets(args)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    tabular_use_gpu = device.type == "cuda"

    results: dict[str, Any] = {
        "_meta": {
            "repo_root": str(REPO_ROOT),
            "data_root": str(DATA_ROOT),
            "weight_root": str(WEIGHT_ROOT),
            "device": str(device),
            "datasets": [config.name for config in datasets_to_run],
            "modalities": sorted(modalities),
            "tabular_models": args.tabular_models,
        }
    }

    for config in datasets_to_run:
        dataset_results: dict[str, Any] = {}
        if "tabular" in modalities:
            tabular_results: dict[str, Any] = {}
            for model_name in args.tabular_models:
                tabular_results[model_name] = safe_run(
                    lambda config=config, model_name=model_name: run_tabular_baseline(
                        config=config,
                        model_name=model_name,
                        seed=args.seed,
                        use_gpu=tabular_use_gpu,
                        predictions_dir=args.predictions_dir,
                    )
                )
            dataset_results["tabular"] = tabular_results

        if "image" in modalities:
            dataset_results["image"] = {
                "resnet50": safe_run(
                    lambda config=config: run_image_baseline(
                        config=config,
                        device=device,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        predictions_dir=args.predictions_dir,
                    )
                )
            }

        results[config.name] = dataset_results

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
