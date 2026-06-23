#!/usr/bin/env python3
"""
Vanilla pseudolabeling with TabSTAR.

For each feature in each dataset:
  1. Train TabSTAR to predict that feature from ALL other features
  2. Generate pseudolabels (predictions + confidence scores)

Output: {dataset}_pseudolabels.csv with predictions for each feature
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import torch
from pandas import DataFrame, Series

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# TabSTAR imports (lazy)
TABSTAR_DIR = "/playpen-nvme/scribble/shbhat/TabSTAR"
sys.path.insert(0, os.path.join(TABSTAR_DIR, "src"))
TabSTARClassifier = None
TabSTARRegressor = None

def _load_tabstar():
    global TabSTARClassifier, TabSTARRegressor
    if TabSTARClassifier is None:
        from tabstar.tabstar_model import TabSTARClassifier as _C, TabSTARRegressor as _R
        TabSTARClassifier = _C
        TabSTARRegressor = _R


def predict_feature_vanilla(
    df: pd.DataFrame,
    target_col: str,
    is_regression: bool,
    device: str = "cuda:0",
    max_epochs: int = 10,
    patience: int = 3,
    n_shots: int = 0,
    pretrain_path: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vanilla pseudolabeling for a single feature:
    Train on all other features → predict target feature.
    
    Returns (predictions, confidences).
    """
    _load_tabstar()
    
    # Features = all columns except target
    feature_cols = [c for c in df.columns if c != target_col]
    if len(feature_cols) == 0:
        logger.warning(f"No features available for {target_col}")
        if is_regression:
            return np.full(len(df), df[target_col].mean()), np.full(len(df), 0.1)
        else:
            return np.full(len(df), str(df[target_col].mode().iloc[0])), np.full(len(df), 0.1)
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    if is_regression:
        y = pd.to_numeric(y, errors='coerce')
    else:
        y = y.astype(str)
    y = pd.Series(y.values, name=target_col, index=X.index)
    
    # Few-shot subsample
    if n_shots > 0 and n_shots < len(X):
        if is_regression:
            idx = np.random.choice(len(X), min(n_shots, len(X)), replace=False)
        else:
            idx = []
            classes = y.unique()
            per_class = max(1, n_shots // len(classes))
            for c in classes:
                c_idx = np.where(y.values == c)[0]
                np.random.shuffle(c_idx)
                idx.extend(c_idx[:per_class].tolist())
            idx = idx[:n_shots]
        X_train = X.iloc[idx].reset_index(drop=True)
        y_train = y.iloc[idx].reset_index(drop=True)
    else:
        X_train = X.copy()
        y_train = y.copy()
    
    # Validate we have enough data
    if not is_regression and y_train.nunique() < 2:
        mode_val = str(df[target_col].mode().iloc[0])
        return np.full(len(df), mode_val), np.full(len(df), 0.2)
    
    if len(X_train) < 3:
        if is_regression:
            return np.full(len(df), float(df[target_col].mean())), np.full(len(df), 0.2)
        else:
            mode_val = str(df[target_col].mode().iloc[0])
            return np.full(len(df), mode_val), np.full(len(df), 0.2)
    
    try:
        import tempfile
        import shutil
        tmp_dir = tempfile.mkdtemp(prefix="tabstar_vanilla_")
        
        kwargs = dict(
            max_epochs=max_epochs,
            patience=patience,
            device=device,
            verbose=False,
            keep_model=False,
            is_paper_version=True,
            output_dir=tmp_dir,
        )
        if pretrain_path:
            kwargs["pretrain_dataset_or_path"] = pretrain_path
        
        if is_regression:
            model = TabSTARRegressor(**kwargs)
        else:
            model = TabSTARClassifier(**kwargs)
        
        # Fit
        if len(X_train) < 10:
            model.fit(X_train, y_train, x_val=X_train.copy(), y_val=y_train.copy())
        else:
            model.fit(X_train, y_train)
        
        # Predict on full dataset
        X_pred = df[feature_cols].copy()
        for col in feature_cols:
            if X_train[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                X_pred[col] = pd.to_numeric(X_pred[col], errors='coerce')
        
        if is_regression:
            preds = model.predict(X_pred)
            pred_std = max(np.std(preds), 1e-6)
            preds = preds + np.random.normal(0, pred_std * 0.05, len(preds))
            confs = np.full(len(preds), 0.8)
        else:
            try:
                proba = model.predict_proba(X_pred)
                classes = model.classes_ if hasattr(model, 'classes_') else np.arange(proba.shape[1])
                preds = np.array([
                    np.random.choice(classes, p=p / p.sum())
                    for p in proba
                ])
                confs = np.array([proba[i, np.where(classes == preds[i])[0][0]]
                                  for i in range(len(preds))])
            except Exception:
                preds = model.predict(X_pred)
                confs = np.full(len(preds), 0.7)
        
        del model
        torch.cuda.empty_cache()
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return preds, confs
    
    except Exception as e:
        logger.warning(f"TabSTAR pseudolabel failed for {target_col}: {e}")
        if 'tmp_dir' in locals():
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)
        if is_regression:
            return np.full(len(df), float(df[target_col].mean())), np.full(len(df), 0.1)
        else:
            mode_val = str(df[target_col].mode().iloc[0])
            return np.full(len(df), mode_val), np.full(len(df), 0.1)


def pseudolabel_dataset(
    csv_path: str,
    output_dir: str,
    device: str = "cuda:0",
    max_epochs: int = 10,
    patience: int = 3,
    n_shots: int = 0,
    pretrain_path: Optional[str] = None,
    mode: str = "lora",
) -> Optional[str]:
    """
    Generate vanilla pseudolabels for all features in a dataset.
    
    Returns output path or None if failed.
    """
    try:
        dataset_name = Path(csv_path).stem
        df = pd.read_csv(csv_path)
        df = df.dropna()
        
        if len(df) < 5:
            logger.warning(f"Too few rows ({len(df)}), skipping {dataset_name}")
            return None
        
        columns = list(df.columns)
        if len(columns) < 2:
            logger.warning(f"Too few columns, skipping {dataset_name}")
            return None
        
        logger.info(f"  {len(columns)} features, {len(df)} rows")
        
        # Create output DataFrames
        pseudolabel_df = pd.DataFrame(index=df.index)
        confidence_df = pd.DataFrame(index=df.index)
        
        # Infer dtypes for each column
        for i, col in enumerate(columns):
            col_data = df[col].dropna()
            is_regression = False
            
            if col_data.dtype not in ['object', 'category']:
                try:
                    vals = pd.to_numeric(col_data, errors='coerce').dropna()
                    if len(vals) >= len(col_data) * 0.8:
                        # Check if truly continuous (not categorical integers)
                        if vals.nunique() > 20 or len(vals) <= vals.nunique() * 10:
                            is_regression = True
                except Exception:
                    pass
            
            task_type = "regression" if is_regression else "classification"
            logger.info(f"  [{i+1}/{len(columns)}] {col} ({task_type}, mode={mode})")
            
            if mode == "lora":
                preds, confs = predict_feature_vanilla(
                    df, col, is_regression, device, max_epochs, patience, n_shots, pretrain_path
                )
            else:
                # Fast sklearn mode
                from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
                feature_cols = [c for c in columns if c != col]
                X = df[feature_cols].copy()
                y = df[col].copy()
                
                # Encode categorical features
                for fcol in feature_cols:
                    if X[fcol].dtype in ['object', 'category']:
                        X[fcol] = pd.Categorical(X[fcol]).codes
                
                if is_regression:
                    y = pd.to_numeric(y, errors='coerce')
                    model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
                    model.fit(X, y)
                    preds = model.predict(X)
                    confs = np.full(len(preds), 0.7)
                else:
                    y = y.astype(str)
                    model = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
                    model.fit(X, y)
                    proba = model.predict_proba(X)
                    preds = model.predict(X)
                    confs = proba.max(axis=1)
            
            pseudolabel_df[col] = preds
            confidence_df[col] = confs
        
        # Save outputs
        pseudo_path = os.path.join(output_dir, f"{dataset_name}_pseudolabels.csv")
        pseudolabel_df.to_csv(pseudo_path, index=False)
        
        conf_path = os.path.join(output_dir, f"{dataset_name}_pseudolabels_confidence.csv")
        confidence_df.to_csv(conf_path, index=False)
        
        avg_conf = confidence_df.mean().mean()
        logger.info(f"  ✓ Avg confidence: {avg_conf:.3f}")
        logger.info(f"  Saved: {pseudo_path}")
        
        return pseudo_path
    
    except Exception as e:
        logger.error(f"Failed to pseudolabel {csv_path}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Vanilla pseudolabeling with TabSTAR (predict each feature from all others)"
    )
    parser.add_argument('--dataset_dir', type=str, required=True,
                        help='Directory containing CSV files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for pseudolabels')
    parser.add_argument('--pretrain_path', type=str, default=None,
                        help='Path to pretrained TabSTAR model (for LoRA init)')
    parser.add_argument('--max_datasets', type=int, default=0,
                        help='Max datasets to process (0=all)')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--n_shots', type=int, default=0,
                        help='Few-shot for per-feature fitting (0=all training data)')
    parser.add_argument('--max_epochs', type=int, default=10,
                        help='Max finetuning epochs per feature model')
    parser.add_argument('--patience', type=int, default=3,
                        help='Early stopping patience')
    parser.add_argument('--dataset_pattern', type=str, default='*.csv')
    parser.add_argument('--mode', type=str, default='lora', choices=['lora', 'fast'],
                        help='lora=TabSTAR LoRA, fast=sklearn RandomForest')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Discover datasets
    dataset_dir = Path(args.dataset_dir)
    csv_paths = sorted(dataset_dir.glob(args.dataset_pattern))
    if args.max_datasets > 0:
        csv_paths = csv_paths[:args.max_datasets]
    
    logger.info(f"Found {len(csv_paths)} datasets in {dataset_dir}")
    logger.info(f"Pretrain: {args.pretrain_path or 'None (TabSTAR default)'}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Output: {args.output_dir}")
    
    processed = 0
    for i, path in enumerate(csv_paths):
        logger.info(f"\n[{i+1}/{len(csv_paths)}] {path.name}")
        result = pseudolabel_dataset(
            str(path), args.output_dir,
            device=args.device,
            max_epochs=args.max_epochs,
            patience=args.patience,
            n_shots=args.n_shots,
            pretrain_path=args.pretrain_path,
            mode=args.mode,
        )
        if result:
            processed += 1
    
    logger.info(f"\nDone. Pseudolabeled {processed}/{len(csv_paths)} datasets.")
    logger.info(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
