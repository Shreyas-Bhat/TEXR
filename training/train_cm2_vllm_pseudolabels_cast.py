#!/usr/bin/env python3
"""
Train CM2 model on vLLM synthetic cast pseudo-labeled datasets.

The pseudo-labeled CSVs contain:
- {feature}_original: Original values
- {feature}_pseudo: Pseudo label predictions  
- {feature}_confidence: Confidence scores
- {feature}_correct: Whether prediction was correct (for categorical)

This script trains CM2 to predict pseudo labels, implementing self-supervised learning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import glob
import logging
import argparse
import numpy as np
import pandas as pd
import random
from typing import List, Dict, Optional
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

# Add CM2 to path
sys.path.insert(0, "/playpen-nvme/scribble/shbhat/CM2")

from CM2.modeling_CM2 import CM2Model, CM2LinearClassifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_raw_csv(csv_path: str):
    """
    Parse a raw CSV file (without pseudolabels).
    Randomly selects one column as target.
    
    Returns:
        Tuple of (feature_names, X_data, y_data, task_type)
    """
    df = pd.read_csv(csv_path)
    
    # Remove any index/ID columns
    drop_cols = [col for col in df.columns if col.lower() in ['id', 'index', 'row_id', 'unnamed: 0']]
    df = df.drop(columns=drop_cols, errors='ignore')
    
    if len(df.columns) < 2 or len(df) < 10:
        return None, None, None, None
    
    # Randomly select a target column
    target_col = random.choice(df.columns.tolist())
    input_cols = [col for col in df.columns if col != target_col]
    
    X = df[input_cols].copy()
    y = df[target_col].copy()
    
    # Drop rows with NaN
    valid_mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[valid_mask]
    y = y[valid_mask]
    
    if len(X) < 10:
        return None, None, None, None
    
    # Determine task type
    try:
        y_numeric = pd.to_numeric(y, errors='coerce')
        if y_numeric.notna().sum() < len(y) * 0.8 or y.nunique() <= 20:
            task_type = 'classification'
        else:
            task_type = 'regression'
    except:
        task_type = 'classification'
    
    return input_cols, X, y, task_type


def parse_pseudolabel_csv(csv_path: str, confidence_threshold: float = 0.0):
    """
    Parse a pseudo-labeled CSV and extract features and training data.
    
    Returns:
        Tuple of (feature_names, X_data, y_data, task_type)
    """
    df = pd.read_csv(csv_path)
    
    # Extract unique feature names (remove suffixes)
    feature_names = set()
    for col in df.columns:
        if col.endswith('_original'):
            feature_names.add(col.replace('_original', ''))
    
    feature_names = sorted(feature_names)
    if 'row_id' in feature_names:
        feature_names.remove('row_id')
    
    if len(feature_names) < 2:
        return None, None, None, None
    
    # Select a random target feature
    target_feature = random.choice(feature_names)
    input_features = [f for f in feature_names if f != target_feature]
    
    if len(input_features) < 1:
        return None, None, None, None
    
    # Build X from original values of input features
    X_cols = []
    for feat in input_features:
        orig_col = f"{feat}_original"
        if orig_col in df.columns:
            X_cols.append(orig_col)
    
    if not X_cols:
        return None, None, None, None
    
    X = df[X_cols].copy()
    X.columns = [col.replace('_original', '') for col in X.columns]
    
    # Build y from pseudo labels of target feature
    pseudo_col = f"{target_feature}_pseudo"
    conf_col = f"{target_feature}_confidence"
    
    if pseudo_col not in df.columns:
        return None, None, None, None
    
    y = df[pseudo_col].copy()
    
    # Filter by confidence if threshold is set
    if confidence_threshold > 0.0 and conf_col in df.columns:
        mask = df[conf_col] >= confidence_threshold
        X = X[mask]
        y = y[mask]
    
    # Drop rows with NaN
    valid_mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[valid_mask]
    y = y[valid_mask]
    
    if len(X) < 10:
        return None, None, None, None
    
    # Determine task type
    try:
        y_numeric = pd.to_numeric(y, errors='coerce')
        if y_numeric.notna().sum() < len(y) * 0.8 or y.nunique() <= 20:
            task_type = 'classification'
        else:
            task_type = 'regression'
    except:
        task_type = 'classification'
    
    return input_features, X, y, task_type


def prepare_cm2_data(X: pd.DataFrame, y: pd.Series, task_type: str):
    """
    Prepare data for CM2 model.
    
    Returns:
        X_processed, y_processed, cat_cols, num_cols, bin_cols, num_classes
    """
    # Identify column types
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    bin_cols = []
    
    # Check for binary numerical columns
    for col in num_cols.copy():
        if X[col].nunique() <= 2:
            bin_cols.append(col)
            num_cols.remove(col)
    
    # Encode target
    if task_type == 'classification':
        le = LabelEncoder()
        y_encoded = le.fit_transform(y.astype(str))
        num_classes = len(le.classes_)
        return X, y_encoded, cat_cols, num_cols, bin_cols, num_classes
    else:
        y_numeric = pd.to_numeric(y, errors='coerce').fillna(y.mean())
        return X, y_numeric.values, cat_cols, num_cols, bin_cols, None


def load_pseudolabel_datasets(
    data_dirs,
    max_datasets: int = None,
    confidence_threshold: float = 0.0,
    max_examples_per_dataset: int = 100
):
    """
    Load pseudo-labeled datasets from one or more directories.
    
    Args:
        data_dirs: Single directory string or list of directory strings
    
    Returns:
        List of (dataset_name, X, y, task_type, cat_cols, num_cols, bin_cols, num_classes)
    """
    # Handle both single directory and list of directories
    if isinstance(data_dirs, str):
        data_dirs = [data_dirs]
    
    # Collect CSV files from all directories
    csv_files = []
    for data_dir in data_dirs:
        logger.info(f"Searching in: {data_dir}")
        # Auto-detect file naming pattern: cast, vanilla, or plain
        dir_csv_files = glob.glob(os.path.join(data_dir, "*_pseudolabels_cast.csv"))
        if not dir_csv_files:
            dir_csv_files = glob.glob(os.path.join(data_dir, "*_pseudolabels_vanilla.csv"))
        if not dir_csv_files:
            dir_csv_files = glob.glob(os.path.join(data_dir, "*_pseudolabels.csv"))
        if not dir_csv_files:
            dir_csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
        csv_files.extend(dir_csv_files)
        logger.info(f"  Found {len(dir_csv_files)} files in {data_dir}")
    
    logger.info(f"Total: {len(csv_files)} pseudo-labeled CSV files across {len(data_dirs)} directories")
    
    if max_datasets and max_datasets < len(csv_files):
        random.seed(42)
        csv_files = random.sample(csv_files, max_datasets)
        logger.info(f"Limited to {max_datasets} datasets")
    
    datasets = []
    
    for csv_path in tqdm(csv_files, desc="Loading datasets"):
        try:
            basename = os.path.basename(csv_path)
            for suffix in ['_pseudolabels_cast.csv', '_pseudolabels_vanilla.csv', '_pseudolabels.csv', '.csv']:
                if basename.endswith(suffix):
                    dataset_name = basename[:-len(suffix)]
                    break
            else:
                dataset_name = basename
            
            # Auto-detect: pseudolabeled (has _original columns) or raw CSV
            df_peek = pd.read_csv(csv_path, nrows=1)
            is_pseudolabeled = any('_original' in col for col in df_peek.columns)
            
            if is_pseudolabeled:
                feature_names, X, y, task_type = parse_pseudolabel_csv(csv_path, confidence_threshold)
            else:
                feature_names, X, y, task_type = parse_raw_csv(csv_path)
            
            if X is None or len(X) < 10:
                continue
            
            # Limit examples per dataset
            if max_examples_per_dataset and len(X) > max_examples_per_dataset:
                indices = random.sample(range(len(X)), max_examples_per_dataset)
                X = X.iloc[indices]
                y = y.iloc[indices]
            
            X_proc, y_proc, cat_cols, num_cols, bin_cols, num_classes = prepare_cm2_data(X, y, task_type)
            
            datasets.append((
                dataset_name,
                X_proc,
                y_proc,
                task_type,
                cat_cols,
                num_cols,
                bin_cols,
                num_classes
            ))
            
        except Exception as e:
            logger.warning(f"Error loading {csv_path}: {e}")
            continue
    
    logger.info(f"Successfully loaded {len(datasets)} datasets")
    return datasets


def train_cm2_on_pseudolabels():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Train CM2 on pseudo-labeled datasets")
    parser.add_argument('--data_dir', type=str, nargs='+', default=['./pseudolabels_vllm_synthetic_cast'],
                        help='Directory(ies) containing pseudo-labeled CSVs (can specify multiple)')
    parser.add_argument('--max_datasets', type=int, default=5000,
                        help='Maximum number of datasets to use')
    parser.add_argument('--max_examples_per_dataset', type=int, default=100,
                        help='Maximum examples per dataset')
    parser.add_argument('--num_epochs', type=int, default=40,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--confidence_threshold', type=float, default=0.0,
                        help='Minimum confidence threshold (0.0-1.0)')
    parser.add_argument('--save_dir', type=str, default='./checkpoints/cm2_vllm_pseudolabels_cast',
                        help='Directory to save checkpoints')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use (cuda:0 or cpu)')
    parser.add_argument('--pretrained_checkpoint', type=str, 
                        default='/playpen-nvme/scribble/shbhat/CM2/cm2_best_model',
                        help='Pretrained CM2 checkpoint directory')
    
    args = parser.parse_args()
    
    set_seed(42)
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Load datasets
    logger.info("Loading pseudo-labeled datasets...")
    datasets = load_pseudolabel_datasets(
        data_dirs=args.data_dir,
        max_datasets=args.max_datasets,
        confidence_threshold=args.confidence_threshold,
        max_examples_per_dataset=args.max_examples_per_dataset
    )
    
    if not datasets:
        logger.error("No datasets loaded. Exiting.")
        return
    
    total_examples = sum(len(X) for _, X, _, _, _, _, _, _ in datasets)
    logger.info(f"Total training examples: {total_examples}")
    
    # Load pretrained CM2 model
    logger.info(f"Loading CM2 from directory: {args.pretrained_checkpoint}")
    try:
        import CM2
        # Build CM2 classifier without loading checkpoint first
        cm2_model = CM2.build_classifier(
            checkpoint=None,  # Don't load checkpoint yet
            device=str(device),
            num_class=2,
            num_layer=2,
            hidden_dropout_prob=0.1,
            vocab_freeze=False,
            use_bert=True,
        )
        
        # Load only the encoder weights, not the classifier head
        if args.pretrained_checkpoint and os.path.exists(args.pretrained_checkpoint):
            checkpoint_path = os.path.join(args.pretrained_checkpoint, 'pytorch_model.bin')
            if os.path.exists(checkpoint_path):
                state_dict = torch.load(checkpoint_path, map_location='cpu')
                # Filter out classifier head weights
                encoder_state_dict = {k: v for k, v in state_dict.items() if not k.startswith('clf.')}
                cm2_model.load_state_dict(encoder_state_dict, strict=False)
                logger.info(f"Loaded encoder weights from {checkpoint_path}")
            else:
                logger.warning(f"No checkpoint found at {checkpoint_path}, using random initialization")
        
        cm2_model = cm2_model.to(device)
        logger.info("CM2 model initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize CM2 model: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Training loop
    logger.info(f"\n{'='*80}")
    logger.info("Starting CM2 training on pseudo-labeled data")
    logger.info(f"{'='*80}")
    
    optimizer = optim.AdamW(cm2_model.parameters(), lr=args.lr)
    best_loss = float('inf')
    
    for epoch in range(args.num_epochs):
        cm2_model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        # Shuffle datasets
        random.shuffle(datasets)
        
        pbar = tqdm(datasets, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        
        for dataset_name, X, y, task_type, cat_cols, num_cols, bin_cols, num_classes in pbar:
            try:
                # Skip datasets with too many categories (can cause CUDA errors)
                if len(cat_cols) > 20 or num_classes > 100:
                    continue
                
                # Update CM2 model with column types for this dataset
                try:
                    cm2_model.update({
                        "cat": [cat_cols],
                        "num": [num_cols],
                        "bin": [bin_cols]
                    })
                except:
                    continue
                
                # Update classifier head for this task
                if task_type == 'classification':
                    cm2_model.num_class = num_classes
                    cm2_model.clf = CM2LinearClassifier(num_class=num_classes, hidden_dim=128).to(device)
                    if num_classes > 2:
                        cm2_model.loss_fn = nn.CrossEntropyLoss(reduction='none')
                    else:
                        cm2_model.loss_fn = nn.BCEWithLogitsLoss(reduction='none')
                
                # Process in batches
                indices = list(range(len(X)))
                random.shuffle(indices)
                
                for i in range(0, len(indices), args.batch_size):
                    batch_indices = indices[i:i+args.batch_size]
                    X_batch = X.iloc[batch_indices]
                    y_batch = y[batch_indices]
                    
                    try:
                        optimizer.zero_grad()
                        
                        # CM2 forward pass - takes DataFrame and target directly
                        if task_type == 'classification':
                            y_series = pd.Series(y_batch, index=X_batch.index)
                        else:
                            y_series = pd.Series(y_batch, index=X_batch.index)
                        
                        logits, loss = cm2_model(X_batch, y_series)
                        
                        if loss is None or torch.isnan(loss) or torch.isinf(loss):
                            continue
                        
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(cm2_model.parameters(), 1.0)
                        optimizer.step()
                        
                        epoch_loss += loss.item()
                        num_batches += 1
                        
                        pbar.set_postfix({'loss': f"{loss.item():.4f}"})
                        
                    except RuntimeError as e:
                        if "CUDA" in str(e):
                            logger.warning(f"CUDA error in batch, resetting GPU")
                            torch.cuda.empty_cache()
                            break  # Skip rest of this dataset
                        continue
                    except Exception as e:
                        continue
                
            except Exception as e:
                continue
        
        avg_loss = epoch_loss / max(num_batches, 1)
        logger.info(f"Epoch {epoch+1}/{args.num_epochs} - Avg Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_path = os.path.join(args.save_dir, 'best_model.pt')
            try:
                # Move model to CPU before saving to avoid CUDA errors
                cm2_model_cpu = cm2_model.cpu()
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': cm2_model_cpu.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                }, checkpoint_path)
                cm2_model = cm2_model.to(device)
                logger.info(f"✓ Saved best model (loss: {avg_loss:.4f})")
            except Exception as e:
                logger.warning(f"Failed to save checkpoint: {e}")
        
        # Save periodic checkpoint
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(args.save_dir, f'checkpoint_epoch_{epoch+1}.pt')
            try:
                cm2_model_cpu = cm2_model.cpu()
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': cm2_model_cpu.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                }, checkpoint_path)
                cm2_model = cm2_model.to(device)
                logger.info(f"Saved checkpoint at epoch {epoch+1}")
            except Exception as e:
                logger.warning(f"Failed to save periodic checkpoint: {e}")
        
        # Clear cache
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except:
                pass
    
    logger.info(f"\n{'='*80}")
    logger.info("Training Complete!")
    logger.info(f"Best Loss: {best_loss:.4f}")
    logger.info(f"Checkpoints saved to: {args.save_dir}")
    logger.info(f"{'='*80}")


if __name__ == "__main__":
    train_cm2_on_pseudolabels()
