

import os
import sys
import json
import glob
import logging
import argparse
import math
from typing import List, Dict, Optional, Any, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans

# TabSTAR imports
TABSTAR_DIR = "/playpen-nvme/scribble/shbhat/TabSTAR"
sys.path.insert(0, os.path.join(TABSTAR_DIR, "src"))

from tabstar.tabstar_model import TabSTARClassifier, TabSTARRegressor

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Data loading utilities
# ─────────────────────────────────────────────────────────────────────

def load_npz_dataset(npz_path: str, metadata_path: str = None) -> Tuple[pd.DataFrame, Dict]:
    data = np.load(npz_path, allow_pickle=True)
    if 'data' in data:
        arr = data['data']
    elif 'X' in data:
        arr = data['X']
    elif 'features' in data:
        arr = data['features']
    else:
        arr = data[list(data.keys())[0]]

    metadata = None
    if metadata_path and os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

    if metadata and 'features' in metadata:
        feature_names = [f['name'] for f in metadata['features']]
        if len(feature_names) >= arr.shape[1]:
            col_names = feature_names[:arr.shape[1]]
        else:
            col_names = feature_names + [f'feature_{i}' for i in range(len(feature_names), arr.shape[1])]
    else:
        col_names = [f'feature_{i}' for i in range(arr.shape[1])]

    return pd.DataFrame(arr, columns=col_names), metadata


def infer_feature_types(df: pd.DataFrame) -> Dict[str, str]:
    """Return {col_name: 'continuous'|'categorical'} for each column."""
    types = {}
    for col in df.columns:
        col_data = df[col].dropna()
        if len(col_data) == 0:
            types[col] = "categorical"
            continue
        if col_data.dtype in ['object', 'category']:
            types[col] = "categorical"
        else:
            try:
                vals = pd.to_numeric(col_data, errors='coerce').dropna()
                if len(vals) >= len(col_data) * 0.8:
                    types[col] = "continuous"
                else:
                    types[col] = "categorical"
            except Exception:
                types[col] = "categorical"
    return types


def discover_datasets(data_dirs: List[str], max_datasets: int = 0,
                      data_dirs_file: str = None) -> List[str]:
    paths = []
    if data_dirs_file and os.path.exists(data_dirs_file):
        with open(data_dirs_file) as f:
            for line in f:
                line = line.strip()
                if line and os.path.exists(line):
                    paths.append(line)
    elif data_dirs:
        for d in data_dirs:
            paths.extend(glob.glob(os.path.join(d, "**/*.csv"), recursive=True))
            for npz in glob.glob(os.path.join(d, "**/dataset.npz"), recursive=True):
                paths.append(npz)
    if max_datasets > 0:
        paths = paths[:max_datasets]
    return paths


# ─────────────────────────────────────────────────────────────────────
# TabSTAR prediction helpers
# ─────────────────────────────────────────────────────────────────────

def build_tabstar_for_target(
    target_col: str,
    is_regression: bool,
    device: str = "cuda:0",
    max_epochs: int = 10,
    patience: int = 3,
    pretrain_path: Optional[str] = None,
) -> Any:
    """Build a TabSTARClassifier or TabSTARRegressor for predicting target_col."""
    # Use unique checkpoint dir to avoid conflicts between parallel jobs on same GPU
    device_tag = device.replace(":", "_")
    pretrain_tag = os.path.basename(os.path.dirname(pretrain_path)) if pretrain_path else "default"
    cp_dir = f".tabstar_checkpoint_{device_tag}_{pretrain_tag}"
    kwargs = dict(
        max_epochs=max_epochs,
        patience=patience,
        device=device,
        verbose=False,
        keep_model=False,
        output_dir=cp_dir,
    )
    if pretrain_path:
        kwargs["pretrain_dataset_or_path"] = pretrain_path

    if is_regression:
        model = TabSTARRegressor(**kwargs)
    else:
        model = TabSTARClassifier(**kwargs)

    return model


def predict_feature_vanilla(
    df: pd.DataFrame,
    target_col: str,
    observed_cols: List[str],
    is_regression: bool,
    device: str = "cuda:0",
    max_epochs: int = 10,
    patience: int = 3,
    n_shots: int = 0,
    pretrain_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vanilla prediction: fit TabSTAR on training data, predict on all rows.
    Returns (predictions, confidences).
    """
    X = df[observed_cols].copy()
    y = df[target_col].copy()

    if is_regression:
        y = pd.to_numeric(y, errors='coerce')
    else:
        y = y.astype(str)

    y = pd.Series(y, name=target_col)

    # Few-shot subsample — TabSTAR needs >=20 samples for internal train/val split
    MIN_TABSTAR_SAMPLES = 20
    effective_shots = max(n_shots, MIN_TABSTAR_SAMPLES) if n_shots > 0 else 0

    if effective_shots > 0 and effective_shots < len(df):
        if is_regression:
            idx = np.random.choice(len(df), min(effective_shots, len(df)), replace=False)
        else:
            # Balanced sampling across classes
            idx = []
            classes = y.unique()
            per_class = max(1, effective_shots // len(classes))
            for c in classes:
                c_idx = np.where(y.values == c)[0]
                np.random.shuffle(c_idx)
                idx.extend(c_idx[:per_class].tolist())
            idx = idx[:effective_shots]
        X_train = X.iloc[idx]
        y_train = y.iloc[idx]
    else:
        X_train = X
        y_train = y

    # Need at least 2 classes for classification, or 5 samples for regression
    if not is_regression and y_train.nunique() < 2:
        # Fallback: mode prediction
        mode_val = y.mode().iloc[0]
        preds = np.full(len(df), mode_val)
        confs = np.full(len(df), 0.3)
        return preds, confs

    if len(X_train) < 3:
        if is_regression:
            preds = np.full(len(df), float(y.mean()))
        else:
            preds = np.full(len(df), str(y.mode().iloc[0]))
        confs = np.full(len(df), 0.2)
        return preds, confs

    def _fit_predict(X_tr, y_tr):
        model = build_tabstar_for_target(
            target_col, is_regression, device, max_epochs, patience, pretrain_path,
        )
        model.fit(X_tr, y_tr)
        if is_regression:
            preds = model.predict(X)
            confs = np.full(len(preds), 0.8)
        else:
            preds = model.predict(X)
            try:
                proba = model.predict_proba(X)
                confs = np.max(proba, axis=1) if proba.ndim > 1 else proba
            except Exception:
                confs = np.full(len(preds), 0.7)
        del model
        torch.cuda.empty_cache()
        return preds, confs

    try:
        return _fit_predict(X_train, y_train)
    except Exception as e:
        logger.warning(f"TabSTAR fit failed for {target_col} with {len(X_train)} samples: {e}")
        # Retry with full data
        if len(X_train) < len(X):
            logger.info(f"  Retrying {target_col} with full data ({len(X)} samples)")
            try:
                return _fit_predict(X, y)
            except Exception as e2:
                logger.warning(f"  Retry also failed for {target_col}: {e2}")

        if is_regression:
            preds = np.full(len(df), float(y.mean()))
        else:
            preds = np.full(len(df), str(y.mode().iloc[0]))
        confs = np.full(len(df), 0.1)
        return preds, confs


# ─────────────────────────────────────────────────────────────────────
# CAST prediction
# ─────────────────────────────────────────────────────────────────────

def predict_feature_cast(
    df: pd.DataFrame,
    target_col: str,
    observed_cols: List[str],
    is_regression: bool,
    device: str = "cuda:0",
    max_epochs: int = 10,
    patience: int = 3,
    n_shots: int = 0,
    pretrain_path: Optional[str] = None,
    cast_rounds: int = 3,
    confidence_threshold: float = 0.85,
    n_clusters: int = 8,
    cast_k: int = 10,
    cast_alpha: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    CAST prediction: iterative self-training with confidence-based sample selection.
    For classification targets only — falls back to vanilla for regression.
    """
    if is_regression:
        return predict_feature_vanilla(
            df, target_col, observed_cols, is_regression,
            device, max_epochs, patience, n_shots, pretrain_path,
        )

    X = df[observed_cols].copy()
    y = df[target_col].copy().astype(str)
    y = pd.Series(y, name=target_col)

    # Initial few-shot subsample
    if n_shots > 0 and n_shots < len(df):
        classes = y.unique()
        per_class = max(1, n_shots // len(classes))
        idx = []
        for c in classes:
            c_idx = np.where(y.values == c)[0]
            np.random.shuffle(c_idx)
            idx.extend(c_idx[:per_class].tolist())
        idx = idx[:n_shots]
        labeled_mask = np.zeros(len(df), dtype=bool)
        labeled_mask[idx] = True
    else:
        labeled_mask = np.ones(len(df), dtype=bool)

    if y[labeled_mask].nunique() < 2:
        mode_val = y.mode().iloc[0]
        return np.full(len(df), mode_val), np.full(len(df), 0.3)

    best_preds = np.full(len(df), str(y.mode().iloc[0]))
    best_confs = np.full(len(df), 0.3)

    for round_idx in range(cast_rounds):
        X_train = X[labeled_mask]
        y_train = y[labeled_mask]

        if len(X_train) < 3 or y_train.nunique() < 2:
            break

        try:
            model = build_tabstar_for_target(
                target_col, False, device, max_epochs, patience, pretrain_path,
            )
            model.fit(X_train, y_train)

            preds = model.predict(X)
            try:
                proba = model.predict_proba(X)
                confs = np.max(proba, axis=1) if proba.ndim > 1 else proba
            except Exception:
                confs = np.full(len(preds), 0.7)

            best_preds = preds
            best_confs = confs

            # Select confident unlabeled samples
            unlabeled = ~labeled_mask
            confident = confs >= confidence_threshold
            new_labeled = unlabeled & confident

            if new_labeled.sum() == 0:
                del model
                torch.cuda.empty_cache()
                break

            # Cluster confident samples and select top from each cluster
            if new_labeled.sum() > n_clusters:
                try:
                    # Use predicted probabilities as features for clustering
                    if proba.ndim > 1:
                        cluster_features = proba[new_labeled]
                    else:
                        cluster_features = confs[new_labeled].reshape(-1, 1)
                    km = KMeans(n_clusters=min(n_clusters, len(cluster_features)),
                                random_state=42, n_init=3)
                    clusters = km.fit_predict(cluster_features)

                    # Select top confident from each cluster
                    new_idx = np.where(new_labeled)[0]
                    selected = []
                    for c in range(km.n_clusters):
                        c_mask = clusters == c
                        c_indices = new_idx[c_mask]
                        c_confs = confs[c_indices]
                        top = c_indices[np.argsort(c_confs)[-max(1, len(c_indices) // 2):]]
                        selected.extend(top.tolist())
                    new_labeled = np.zeros(len(df), dtype=bool)
                    new_labeled[selected] = True
                except Exception:
                    pass

            # Expand labeled set: use pseudo-labels for newly selected samples
            labeled_mask = labeled_mask | new_labeled
            # Override y with predictions for pseudo-labeled samples
            y = y.copy()
            y[new_labeled] = pd.Series(preds, index=df.index)[new_labeled]

            del model
            torch.cuda.empty_cache()

            logger.debug(f"  CAST round {round_idx+1}: {labeled_mask.sum()} labeled, "
                         f"{new_labeled.sum()} new")

        except Exception as e:
            logger.warning(f"CAST round {round_idx+1} failed for {target_col}: {e}")
            break

    return best_preds, best_confs


# ─────────────────────────────────────────────────────────────────────
# Autoregressive processing
# ─────────────────────────────────────────────────────────────────────

def process_dataset(
    file_path: str,
    output_dir: str,
    method: str = "vanilla",
    max_rows: Optional[int] = None,
    device: str = "cuda:0",
    max_epochs: int = 10,
    patience: int = 3,
    n_shots: int = 5,
    pretrain_path: Optional[str] = None,
    cast_rounds: int = 3,
    confidence_threshold: float = 0.85,
    n_clusters: int = 8,
    cast_k: int = 10,
    cast_alpha: float = 1.0,
) -> Optional[str]:
    """
    Autoregressively pseudolabel all features in a dataset using TabSTAR.

    Strategy: Leave-one-out — for each feature, use ground truth of ALL other
    features as input (not previously predicted values) to avoid error cascading.
    """
    try:
        if file_path.endswith('.npz'):
            npz_dir = os.path.dirname(file_path)
            meta_path = os.path.join(npz_dir, 'final_metadata.json')
            df, _ = load_npz_dataset(file_path, meta_path if os.path.exists(meta_path) else None)
            dataset_name = os.path.basename(npz_dir)
        else:
            df = pd.read_csv(file_path)
            dataset_name = os.path.splitext(os.path.basename(file_path))[0]

        if max_rows:
            df = df.head(max_rows)

        df = df.dropna()
        if len(df) < 10:
            logger.warning(f"Too few rows ({len(df)}), skipping {dataset_name}")
            return None

        feature_types = infer_feature_types(df)
        columns = list(df.columns)

        if len(columns) < 2:
            logger.warning(f"Too few columns, skipping {dataset_name}")
            return None

        logger.info(f"  {len(df)} rows, {len(columns)} features")

        results_per_feature = {}

        for step, target_col in enumerate(tqdm(columns, desc=f"AR-{method}")):
            # Leave-one-out: use ALL other features' ground truth as input
            observed_cols = [c for c in columns if c != target_col]
            is_regression = feature_types.get(target_col) == "continuous"

            if not observed_cols:
                # Single-column dataset — use dataset statistics
                if is_regression:
                    preds = np.full(len(df), float(df[target_col].mean()))
                    confs = np.full(len(df), 0.3)
                else:
                    mode_val = str(df[target_col].mode().iloc[0])
                    preds = np.full(len(df), mode_val)
                    confs = np.full(len(df), 0.3)

                results_per_feature[target_col] = {
                    "predictions": preds, "confidences": confs, "is_regression": is_regression,
                }
                continue

            # Build sub-DataFrame: ground truth of other features + target
            sub_df = df[observed_cols].copy()
            sub_df[target_col] = df[target_col].values

            # Ensure correct dtypes
            for col in observed_cols:
                if feature_types.get(col) == "continuous":
                    sub_df[col] = pd.to_numeric(sub_df[col], errors='coerce')
            sub_df = sub_df.dropna()

            if len(sub_df) < 5:
                if is_regression:
                    preds = np.full(len(df), float(df[target_col].mean()))
                else:
                    preds = np.full(len(df), str(df[target_col].mode().iloc[0]))
                confs = np.full(len(df), 0.1)
                results_per_feature[target_col] = {
                    "predictions": preds, "confidences": confs, "is_regression": is_regression,
                }
                continue

            if method == "cast":
                preds, confs = predict_feature_cast(
                    sub_df, target_col, observed_cols, is_regression,
                    device, max_epochs, patience, n_shots, pretrain_path,
                    cast_rounds, confidence_threshold, n_clusters,
                    cast_k, cast_alpha,
                )
            else:
                preds, confs = predict_feature_vanilla(
                    sub_df, target_col, observed_cols, is_regression,
                    device, max_epochs, patience, n_shots, pretrain_path,
                )

            results_per_feature[target_col] = {
                "predictions": preds, "confidences": confs, "is_regression": is_regression,
            }

        # Build output DataFrame
        rows = []
        for idx in df.index:
            row_result = {"row_id": idx}
            iloc_idx = df.index.get_loc(idx)
            for col in columns:
                original = df.at[idx, col]
                pseudo = results_per_feature[col]["predictions"]
                if hasattr(pseudo, '__len__'):
                    pseudo_val = pseudo[iloc_idx] if iloc_idx < len(pseudo) else original
                else:
                    pseudo_val = pseudo
                conf = results_per_feature[col]["confidences"]
                if hasattr(conf, '__len__'):
                    conf_val = conf[iloc_idx] if iloc_idx < len(conf) else 0.0
                else:
                    conf_val = conf

                row_result[f"{col}_original"] = original
                row_result[f"{col}_pseudo"] = pseudo_val
                row_result[f"{col}_confidence"] = conf_val
                if not results_per_feature[col]["is_regression"]:
                    row_result[f"{col}_correct"] = (str(pseudo_val) == str(original))
            rows.append(row_result)

        results_df = pd.DataFrame(rows)
        suffix = f"_ar_pseudolabels.csv"
        output_path = os.path.join(output_dir, f"{dataset_name}{suffix}")
        results_df.to_csv(output_path, index=False)

        # Report accuracy
        for col in columns:
            if not results_per_feature[col]["is_regression"]:
                corr_col = f"{col}_correct"
                if corr_col in results_df.columns:
                    acc = results_df[corr_col].mean()
                    logger.info(f"  {col}: {acc:.2%}")

        conf_cols = [c for c in results_df.columns if c.endswith('_confidence')]
        if conf_cols:
            avg_conf = results_df[conf_cols].mean().mean()
            logger.info(f"  Avg confidence: {avg_conf:.3f}")

        logger.info(f"Saved {len(results_df)} rows to {output_path}")
        return output_path

    except Exception as e:
        logger.error(f"Error processing {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Autoregressive pseudolabelling with TabSTAR")
    parser.add_argument('--data_dirs', type=str, nargs='+', default=None, help='Data directories (CSV/NPZ)')
    parser.add_argument('--data_dirs_file', type=str, default=None,
                        help='File listing dataset paths (one per line)')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--method', type=str, default='vanilla', choices=['vanilla', 'cast'])
    parser.add_argument('--max_datasets', type=int, default=0, help='Max datasets (0=all)')
    parser.add_argument('--max_rows', type=int, default=None, help='Max rows per dataset')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--pretrain_path', type=str, default=None,
                        help='Path to pretrained TabSTAR model (default: alana89/TabSTAR from HF)')
    parser.add_argument('--n_shots', type=int, default=5,
                        help='Number of shots for finetuning per feature (0=all)')
    parser.add_argument('--max_epochs', type=int, default=10, help='Max finetuning epochs per feature')
    parser.add_argument('--patience', type=int, default=3, help='Early stopping patience')
    # CAST parameters
    parser.add_argument('--cast_rounds', type=int, default=3)
    parser.add_argument('--confidence_threshold', type=float, default=0.85)
    parser.add_argument('--n_clusters', type=int, default=8)
    parser.add_argument('--cast_k', type=int, default=10)
    parser.add_argument('--cast_alpha', type=float, default=1.0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Discover datasets
    if not args.data_dirs and not args.data_dirs_file:
        parser.error('Must provide --data_dirs or --data_dirs_file')
    dataset_paths = discover_datasets(args.data_dirs or [], args.max_datasets, args.data_dirs_file)
    logger.info(f"Found {len(dataset_paths)} datasets")
    logger.info(f"TabSTAR pretrain: {args.pretrain_path or 'alana89/TabSTAR (default)'}")
    logger.info(f"Method: {args.method}, n_shots: {args.n_shots}")

    processed = 0
    for i, path in enumerate(dataset_paths):
        logger.info(f"\n[{i+1}/{len(dataset_paths)}] {os.path.basename(path)}")
        result = process_dataset(
            path, args.output_dir, method=args.method,
            max_rows=args.max_rows, device=args.device,
            max_epochs=args.max_epochs, patience=args.patience,
            n_shots=args.n_shots, pretrain_path=args.pretrain_path,
            cast_rounds=args.cast_rounds,
            confidence_threshold=args.confidence_threshold,
            n_clusters=args.n_clusters,
            cast_k=args.cast_k, cast_alpha=args.cast_alpha,
        )
        if result:
            processed += 1

    logger.info(f"\nDone. Processed {processed}/{len(dataset_paths)} datasets.")
    logger.info(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
