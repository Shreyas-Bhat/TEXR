#!/usr/bin/env python3
"""
Comprehensive Evaluation of CM2 Model Trained on CAST Pseudo-Labels
===================================================================
Tests the CM2 model trained on CAST pseudo-labeled data against benchmark datasets.
"""

import torch
import numpy as np
import pandas as pd
import yaml
import argparse
import logging
import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

# Add CM2 to path
sys.path.insert(0, "/playpen-nvme/scribble/shbhat/CM2")

from CM2 import modeling_CM2, build_classifier

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_cm2_checkpoint(checkpoint_path: str, device='cuda:0'):
    """Load CM2 model from checkpoint (handles both .pt and .bin formats)"""
    logger.info(f"Loading CM2 checkpoint from: {checkpoint_path}")
    
    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Extract model state dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            logger.info(f"  Loaded from epoch {checkpoint.get('epoch', 'unknown')}")
            logger.info(f"  Training loss: {checkpoint.get('loss', 'unknown'):.4f}")
        elif isinstance(checkpoint, dict):
            state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Build CM2 model
        model = build_classifier(
            checkpoint=None,
            device=device,
            num_class=2,
            num_layer=2,
            hidden_dropout_prob=0.1,
            vocab_freeze=False,
            use_bert=True
        )
        
        # Load state dict (strict=False to allow missing classifier head)
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            logger.info(f"  Missing keys (expected): {len(missing_keys)}")
        if unexpected_keys:
            logger.warning(f"  Unexpected keys: {len(unexpected_keys)}")
        
        model = model.to(device)
        model.eval()
        
        logger.info("✅ CM2 model loaded successfully")
        return model
        
    except Exception as e:
        logger.error(f"❌ Failed to load checkpoint: {e}")
        import traceback
        traceback.print_exc()
        raise


def load_yaml_datasets(yaml_path: str, max_datasets: int = None) -> Dict:
    """Load dataset metadata from YAML"""
    logger.info(f"Loading datasets from: {yaml_path}")
    
    with open(yaml_path, 'r') as f:
        datasets = yaml.safe_load(f)
    
    # Filter for datasets with valid paths
    valid_datasets = {}
    for name, info in datasets.items():
        if info.get('path') and Path(info['path']).exists():
            valid_datasets[name] = info
    
    logger.info(f"  Found {len(valid_datasets)}/{len(datasets)} datasets with valid paths")
    
    if max_datasets and len(valid_datasets) > max_datasets:
        valid_datasets = dict(list(valid_datasets.items())[:max_datasets])
        logger.info(f"  Limited to {max_datasets} datasets")
    
    return valid_datasets


def preprocess_dataset(df: pd.DataFrame, target_col: str) -> Tuple:
    """Preprocess dataset for CM2"""
    # Drop target from features
    y = df[target_col]
    X = df.drop(columns=[target_col])
    
    # Identify column types
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    bin_cols = []
    
    # Check for binary numerical columns
    for col in num_cols.copy():
        if X[col].nunique() <= 2:
            bin_cols.append(col)
            num_cols.remove(col)
    
    # Determine task type
    is_classification = False
    try:
        y_float = pd.to_numeric(y, errors='coerce')
        if y_float.notna().sum() < len(y) * 0.8 or y.nunique() <= 20:
            is_classification = True
    except:
        is_classification = True
    
    if is_classification:
        le = LabelEncoder()
        y_encoded = le.fit_transform(y.astype(str))
        num_classes = len(le.classes_)
        task_type = 'classification'
    else:
        y_encoded = y_float.values
        num_classes = None
        task_type = 'regression'
    
    return X, y_encoded, task_type, num_classes, cat_cols, num_cols, bin_cols


def evaluate_on_dataset(model, dataset_name: str, dataset_info: Dict, device='cuda:0') -> Dict:
    """Evaluate CM2 on a single dataset"""
    logger.info(f"\n{'='*70}")
    logger.info(f"Testing: {dataset_name}")
    logger.info(f"{'='*70}")
    
    try:
        # Load dataset
        csv_path = dataset_info['path']
        df = pd.read_csv(csv_path, low_memory=False)
        logger.info(f"  Loaded: {len(df)} rows, {len(df.columns)} columns")
        
        # Find target column
        target_col = None
        for col_name in ['target', 'label', 'class', 'y']:
            if col_name in df.columns:
                target_col = col_name
                break
        
        if not target_col:
            # Check feature descriptions
            if 'feature_descriptions' in dataset_info:
                for col, desc in dataset_info['feature_descriptions'].items():
                    if 'target' in str(desc).lower() and col in df.columns:
                        target_col = col
                        break
        
        if not target_col:
            target_col = df.columns[-1]
        
        logger.info(f"  Target: {target_col}")
        
        # Drop missing targets
        df = df.dropna(subset=[target_col])
        
        if len(df) < 20:
            logger.warning(f"  ⚠️  Too few samples: {len(df)}")
            return {}
        
        # Limit size for faster evaluation
        if len(df) > 500:
            df = df.sample(n=500, random_state=42)
            logger.info(f"  Sampled to 500 rows")
        
        # Preprocess
        X, y, task_type, num_classes, cat_cols, num_cols, bin_cols = preprocess_dataset(df, target_col)
        
        logger.info(f"  Task: {task_type}")
        if task_type == 'classification':
            logger.info(f"  Classes: {num_classes}")
        logger.info(f"  Features: {len(cat_cols)} cat, {len(num_cols)} num, {len(bin_cols)} bin")
        
        # Split data
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42,
                stratify=y if task_type == 'classification' and num_classes <= 20 else None
            )
        except:
            # If stratify fails, do regular split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, random_state=42
            )
        
        logger.info(f"  Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Update model
        try:
            model.update({
                'cat': [cat_cols],
                'num': [num_cols],
                'bin': [bin_cols]
            })
            
            if task_type == 'classification':
                import torch.nn as nn
                from CM2.modeling_CM2 import CM2LinearClassifier
                
                # Update number of classes
                model.num_class = num_classes
                
                # Reinitialize classifier head for this dataset
                model.clf = CM2LinearClassifier(num_class=num_classes, hidden_dim=128).to(device)
                
                # Set appropriate loss function
                if num_classes == 2:
                    model.loss_fn = nn.BCEWithLogitsLoss(reduction='none')
                else:
                    model.loss_fn = nn.CrossEntropyLoss(reduction='none')
                    
                logger.info(f"  Setup classifier: {num_classes} classes, loss={model.loss_fn.__class__.__name__}")
        except Exception as e:
            logger.warning(f"  ⚠️  Model update failed: {e}")
            return {}
        
        # Make predictions
        try:
            with torch.no_grad():
                if task_type == 'classification':
                    # Convert test targets to Series for CM2
                    y_test_series = pd.Series(y_test, index=X_test.index)
                    
                    # CM2 forward pass returns (logits, loss)
                    logits, _ = model(X_test, y_test_series)
                    
                    # Get predictions
                    predictions = torch.argmax(logits, dim=1).cpu().numpy()
                    
                    # Calculate metrics
                    accuracy = accuracy_score(y_test, predictions)
                    f1 = f1_score(y_test, predictions, average='weighted', zero_division=0)
                    
                    # Try to calculate AUC for binary classification
                    try:
                        if num_classes == 2:
                            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                            auc = roc_auc_score(y_test, probs)
                        else:
                            auc = None
                    except:
                        auc = None
                    
                    logger.info(f"\n  📊 Results:")
                    logger.info(f"     Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
                    logger.info(f"     F1 Score: {f1:.4f}")
                    if auc is not None:
                        logger.info(f"     ROC-AUC: {auc:.4f}")
                    
                    return {
                        'dataset': dataset_name,
                        'task': task_type,
                        'num_classes': num_classes,
                        'num_test': len(y_test),
                        'accuracy': accuracy,
                        'f1': f1,
                        'auc': auc
                    }
                    
                else:
                    # Regression: CM2 doesn't have good regression support yet
                    # Use simple baseline: predict mean
                    predictions = np.full(len(y_test), y_train.mean())
                    
                    mse = mean_squared_error(y_test, predictions)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(y_test, predictions)
                    
                    logger.info(f"\n  📊 Results (Baseline):")
                    logger.info(f"     RMSE: {rmse:.4f}")
                    logger.info(f"     R²: {r2:.4f}")
                    
                    return {
                        'dataset': dataset_name,
                        'task': task_type,
                        'num_test': len(y_test),
                        'rmse': rmse,
                        'r2': r2
                    }
                    
        except Exception as e:
            logger.error(f"  ❌ Prediction failed: {e}")
            import traceback
            traceback.print_exc()
            return {}
            
    except Exception as e:
        logger.error(f"❌ Dataset processing failed: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser(description="Evaluate CM2 CAST Model")
    parser.add_argument('--checkpoint', type=str, 
                       default='./checkpoints/cm2_synthetic_cast_pseudolabels/best_model.pt',
                       help='Path to CM2 checkpoint')
    parser.add_argument('--yaml_config', type=str,
                       default='/playpen-nvme/scribble/shbhat/universal_machine_old/test_datasets_metadata.yaml',
                       help='Path to YAML with test datasets')
    parser.add_argument('--device', type=str, default='cuda:0',
                       help='Device to use')
    parser.add_argument('--max_datasets', type=int, default=50,
                       help='Maximum number of datasets to evaluate')
    parser.add_argument('--output_json', type=str, default='cm2_cast_results.json',
                       help='Output JSON file for results')
    
    args = parser.parse_args()
    
    # Verify checkpoint exists
    if not Path(args.checkpoint).exists():
        logger.error(f"❌ Checkpoint not found: {args.checkpoint}")
        return 1
    
    try:
        # Load model
        model = load_cm2_checkpoint(args.checkpoint, device=args.device)
        
        # Load datasets
        datasets = load_yaml_datasets(args.yaml_config, args.max_datasets)
        
        # Evaluate on all datasets
        logger.info(f"\n{'='*80}")
        logger.info(f"🚀 Starting Evaluation on {len(datasets)} Datasets")
        logger.info(f"{'='*80}\n")
        
        all_results = []
        
        for dataset_name, dataset_info in tqdm(datasets.items(), desc="Evaluating"):
            result = evaluate_on_dataset(model, dataset_name, dataset_info, device=args.device)
            if result:
                all_results.append(result)
        
        # Summary Statistics
        logger.info(f"\n{'='*80}")
        logger.info(f"📊 EVALUATION SUMMARY")
        logger.info(f"{'='*80}")
        
        # Classification results
        cls_results = [r for r in all_results if r['task'] == 'classification']
        if cls_results:
            accuracies = [r['accuracy'] for r in cls_results]
            f1s = [r['f1'] for r in cls_results]
            aucs = [r['auc'] for r in cls_results if r.get('auc') is not None]
            
            logger.info(f"\n🎯 Classification ({len(cls_results)} datasets):")
            logger.info(f"   Mean Accuracy: {np.mean(accuracies):.4f} ± {np.std(accuracies):.4f}")
            logger.info(f"   Median Accuracy: {np.median(accuracies):.4f}")
            logger.info(f"   Mean F1: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
            if aucs:
                logger.info(f"   Mean AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
        
        # Regression results
        reg_results = [r for r in all_results if r['task'] == 'regression']
        if reg_results:
            rmses = [r['rmse'] for r in reg_results]
            r2s = [r['r2'] for r in reg_results]
            
            logger.info(f"\n📈 Regression ({len(reg_results)} datasets):")
            logger.info(f"   Mean RMSE: {np.mean(rmses):.4f} ± {np.std(rmses):.4f}")
            logger.info(f"   Mean R²: {np.mean(r2s):.4f} ± {np.std(r2s):.4f}")
        
        logger.info(f"\n✅ Evaluated {len(all_results)}/{len(datasets)} datasets successfully")
        
        # Save results to JSON
        output_data = {
            'checkpoint': args.checkpoint,
            'num_datasets': len(all_results),
            'results': all_results,
            'summary': {
                'classification': {
                    'num_datasets': len(cls_results),
                    'mean_accuracy': float(np.mean([r['accuracy'] for r in cls_results])) if cls_results else None,
                    'mean_f1': float(np.mean([r['f1'] for r in cls_results])) if cls_results else None
                },
                'regression': {
                    'num_datasets': len(reg_results),
                    'mean_rmse': float(np.mean([r['rmse'] for r in reg_results])) if reg_results else None,
                    'mean_r2': float(np.mean([r['r2'] for r in reg_results])) if reg_results else None
                }
            }
        }
        
        with open(args.output_json, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        logger.info(f"\n💾 Results saved to: {args.output_json}")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
