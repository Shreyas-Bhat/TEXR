#!/usr/bin/env python3
"""
Pretrain TabSTAR from scratch on CAST-generated synthetic datasets.

This script trains a TabSTAR foundation model using the autoregressive CAST
synthetic data as the pretraining corpus.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm

# TabSTAR imports
TABSTAR_DIR = "/playpen-nvme/scribble/shbhat/TabSTAR"
sys.path.insert(0, os.path.join(TABSTAR_DIR, "src"))

from tabstar.arch.arch import TabStarModel
from tabstar.preprocessing.nulls import raise_if_null_target
from tabstar.preprocessing.splits import split_to_val
from tabstar.tabstar_verbalizer import TabSTARVerbalizer, TabSTARData
from tabstar.training.dataloader import get_dataloader
from tabstar.training.devices import get_device
from tabstar.training.early_stopping import EarlyStopping
from tabstar.training.metrics import apply_loss_fn, calculate_metric, calculate_loss
from tabstar.training.utils import fix_seed, concat_predictions
from tabstar_paper.pretraining.config import TabStarConfig
from tabstar_paper.pretraining.unfreezing import unfreeze_text_encoder

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
)
logger = logging.getLogger(__name__)


class CASTPretrainer:
    """Pretrain TabSTAR on CAST-generated synthetic datasets."""
    
    def __init__(
        self,
        dataset_csvs: List[str],
        output_dir: str,
        device: str = "cuda:0",
        batch_size: int = 32,
        global_batch_size: int = 256,
        max_epochs: int = 10,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        patience: int = 3,
        max_datasets: int = 0,
        tabular_layers: int = 6,
        unfreeze_layers: int = 3,
        val_ratio: float = 0.2,
    ):
        self.dataset_csvs = dataset_csvs[:max_datasets] if max_datasets > 0 else dataset_csvs
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = get_device(device)
        self.batch_size = batch_size
        self.global_batch_size = global_batch_size
        self.accumulation_steps = max(1, global_batch_size // batch_size)
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.patience = patience
        self.val_ratio = val_ratio
        
        self.use_amp = bool(self.device.type == "cuda")
        self.scaler = GradScaler(enabled=self.use_amp)
        
        # Initialize model
        logger.info("Initializing TabSTAR model from scratch...")
        config = TabStarConfig(num_layers=tabular_layers, unfreeze_layers=unfreeze_layers)
        self.model = TabStarModel(config=config)
        unfreeze_text_encoder(self.model.text_encoder, layers_to_unfreeze=config.unfreeze_layers)
        self.model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(f"Model initialized: {trainable_params:,} / {total_params:,} trainable parameters")
        
        # Will be set during training
        self.optimizer = None
        self.scheduler = None
        self.early_stopper = EarlyStopping(patience=patience)
        
        fix_seed()
    
    def load_and_prepare_dataset(self, csv_path: str) -> Optional[Tuple[TabSTARData, TabSTARData, bool]]:
        """Load a CAST-generated CSV and prepare for TabSTAR training."""
        try:
            df = pd.read_csv(csv_path)
            df = df.dropna()
            
            if len(df) < 10:
                logger.warning(f"Skipping {Path(csv_path).name}: too few rows ({len(df)})")
                return None
            
            if len(df.columns) < 2:
                logger.warning(f"Skipping {Path(csv_path).name}: too few columns")
                return None
            
            # Use last column as target (arbitrary choice for pretraining)
            target_col = df.columns[-1]
            X = df.drop(columns=[target_col])
            y = df[target_col]
            
            # Infer task type
            is_cls = y.dtype in ['object', 'category'] or y.nunique() <= 20
            
            if is_cls:
                y = y.astype(str)
            else:
                y = pd.to_numeric(y, errors='coerce')
            
            raise_if_null_target(y)
            
            # Split train/val
            X_train, X_val, y_train, y_val = split_to_val(
                x=X, y=y, is_cls=is_cls, val_ratio=self.val_ratio
            )
            
            # Preprocess
            preprocessor = TabSTARVerbalizer(is_cls=is_cls, verbose=False)
            preprocessor.fit(X_train, y_train)
            train_data = preprocessor.transform(X_train, y_train)
            val_data = preprocessor.transform(X_val, y_val)
            
            return train_data, val_data, is_cls
        
        except Exception as e:
            logger.warning(f"Failed to load {Path(csv_path).name}: {e}")
            return None
    
    def train_epoch(self, dataloaders: List[torch.utils.data.DataLoader]) -> float:
        """Train one epoch on all datasets."""
        self.model.train()
        total_loss = 0.0
        total_batches = 0
        
        # Interleave batches from all datasets
        all_iters = [iter(dl) for dl in dataloaders]
        
        # Estimate total batches
        total_expected = sum(len(dl) for dl in dataloaders)
        
        pbar = tqdm(total=total_expected, desc="Training", leave=False)
        
        self.optimizer.zero_grad()
        
        dataset_idx = 0
        finished = [False] * len(all_iters)
        
        while not all(finished):
            # Round-robin through datasets
            if finished[dataset_idx]:
                dataset_idx = (dataset_idx + 1) % len(all_iters)
                continue
            
            try:
                batch = next(all_iters[dataset_idx])
            except StopIteration:
                finished[dataset_idx] = True
                dataset_idx = (dataset_idx + 1) % len(all_iters)
                continue
            
            # Forward pass
            with autocast(device_type=self.device.type, enabled=self.use_amp):
                predictions = self.model(
                    x_txt=batch.x_txt,
                    x_num=batch.x_num,
                    d_output=batch.d_output
                )
                loss = calculate_loss(
                    predictions=predictions,
                    y=batch.y,
                    d_output=batch.d_output
                )
                loss_scaled = loss / self.accumulation_steps
            
            # Backward pass
            self.scaler.scale(loss_scaled).backward()
            
            # Update weights every accumulation_steps
            if (total_batches + 1) % self.accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            total_loss += loss.item()
            total_batches += 1
            pbar.update(1)
            pbar.set_postfix({"loss": f"{total_loss / total_batches:.4f}"})
            
            dataset_idx = (dataset_idx + 1) % len(all_iters)
        
        pbar.close()
        
        # Final update if there are remaining gradients
        if total_batches % self.accumulation_steps != 0:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
        
        return total_loss / max(total_batches, 1)
    
    def validate(self, dataloaders: List[torch.utils.data.DataLoader]) -> Tuple[float, float]:
        """Validate on all datasets."""
        self.model.eval()
        total_loss = 0.0
        total_metric = 0.0
        total_batches = 0
        
        with torch.no_grad():
            for dl in dataloaders:
                for batch in dl:
                    with autocast(device_type=self.device.type, enabled=self.use_amp):
                        predictions = self.model(
                            x_txt=batch.x_txt,
                            x_num=batch.x_num,
                            d_output=batch.d_output
                        )
                        loss = calculate_loss(
                            predictions=predictions,
                            y=batch.y,
                            d_output=batch.d_output
                        )
                        preds_transformed = apply_loss_fn(
                            prediction=predictions,
                            d_output=batch.d_output
                        )
                    
                    total_loss += loss.item()
                    
                    # Calculate metric
                    y_pred = preds_transformed.cpu().detach()
                    y_true = batch.y
                    if isinstance(y_true, torch.Tensor):
                        y_true = y_true.cpu().detach()
                    metrics = calculate_metric(y_true=y_true, y_pred=y_pred, d_output=batch.d_output)
                    total_metric += metrics.score
                    total_batches += 1
        
        return total_loss / max(total_batches, 1), total_metric / max(total_batches, 1)
    
    def pretrain(self):
        """Run full pretraining loop."""
        logger.info(f"\n{'='*80}")
        logger.info(f"CAST TabSTAR Pretraining")
        logger.info(f"{'='*80}")
        logger.info(f"Datasets: {len(self.dataset_csvs)}")
        logger.info(f"Max epochs: {self.max_epochs}")
        logger.info(f"Batch size: {self.batch_size} (global: {self.global_batch_size})")
        logger.info(f"Learning rate: {self.learning_rate}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Output: {self.output_dir}")
        logger.info(f"{'='*80}\n")
        
        # Load all datasets
        logger.info("Loading datasets...")
        train_dataloaders = []
        val_dataloaders = []
        
        for csv_path in tqdm(self.dataset_csvs, desc="Loading datasets"):
            result = self.load_and_prepare_dataset(csv_path)
            if result is None:
                continue
            
            train_data, val_data, is_cls = result
            
            train_dl = get_dataloader(train_data, is_train=True, batch_size=self.batch_size)
            val_dl = get_dataloader(val_data, is_train=False, batch_size=self.batch_size)
            
            train_dataloaders.append(train_dl)
            val_dataloaders.append(val_dl)
        
        if len(train_dataloaders) == 0:
            logger.error("No valid datasets loaded!")
            return
        
        logger.info(f"Loaded {len(train_dataloaders)} valid datasets\n")
        
        # Initialize optimizer and scheduler
        total_batches = sum(len(dl) for dl in train_dataloaders)
        total_steps = (total_batches // self.accumulation_steps) * self.max_epochs
        
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        self.scheduler = OneCycleLR(
            self.optimizer,
            max_lr=self.learning_rate,
            total_steps=total_steps,
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(1, self.max_epochs + 1):
            logger.info(f"\nEpoch {epoch}/{self.max_epochs}")
            logger.info("-" * 80)
            
            # Train
            epoch_start = time.time()
            train_loss = self.train_epoch(train_dataloaders)
            
            # Validate
            val_loss, val_metric = self.validate(val_dataloaders)
            
            epoch_time = time.time() - epoch_start
            
            logger.info(f"Train Loss: {train_loss:.4f}")
            logger.info(f"Val Loss: {val_loss:.4f}")
            logger.info(f"Val Metric: {val_metric:.4f}")
            logger.info(f"Time: {epoch_time:.1f}s")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                logger.info(f"🏆 New best model! (val_loss={val_loss:.4f})")
                
                save_path = self.output_dir / "best_model"
                self.model.save_pretrained(str(save_path))
                logger.info(f"Saved to: {save_path}")
                
                # Save metadata
                metadata = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_metric": val_metric,
                    "num_datasets": len(train_dataloaders),
                    "total_params": sum(p.numel() for p in self.model.parameters()),
                    "trainable_params": sum(p.numel() for p in self.model.parameters() if p.requires_grad),
                }
                
                with open(self.output_dir / "best_model" / "pretrain_args.json", 'w') as f:
                    json.dump(metadata, f, indent=2)
            
            # Early stopping
            self.early_stopper.update(val_loss)
            if self.early_stopper.should_stop:
                logger.info(f"\n⏸️ Early stopping triggered at epoch {epoch}")
                break
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Pretraining complete!")
        logger.info(f"Best val loss: {best_val_loss:.4f}")
        logger.info(f"Model saved to: {self.output_dir / 'best_model'}")
        logger.info(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Pretrain TabSTAR from scratch on CAST-generated datasets"
    )
    parser.add_argument('--dataset_dir', type=str, default=None,
                        help='Directory containing CAST-generated CSV files')
    parser.add_argument('--csv_list', type=str, default=None,
                        help='Text file with one CSV path per line')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for pretrained model')
    parser.add_argument('--dataset_pattern', type=str, default='*.csv',
                        help='File pattern for datasets (default: *.csv)')
    parser.add_argument('--max_datasets', type=int, default=0,
                        help='Max datasets to use (0=all, default: 0)')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Per-device batch size')
    parser.add_argument('--global_batch_size', type=int, default=256,
                        help='Global batch size (uses gradient accumulation)')
    parser.add_argument('--max_epochs', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--patience', type=int, default=3,
                        help='Early stopping patience')
    parser.add_argument('--tabular_layers', type=int, default=6,
                        help='Number of tabular encoder layers')
    parser.add_argument('--unfreeze_layers', type=int, default=3,
                        help='Number of text encoder layers to unfreeze')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='Validation split ratio')
    parser.add_argument('--num_partitions', type=int, default=1,
                        help='Split datasets into N equal partitions (default: 1 = all)')
    parser.add_argument('--partition', type=int, default=0,
                        help='Which partition to train on, 0-indexed (default: 0)')

    args = parser.parse_args()
    
    # Find CSVs from list file or directory
    if args.csv_list:
        with open(args.csv_list) as f:
            csv_paths = [line.strip() for line in f if line.strip() and line.strip().endswith('.csv')]
    elif args.dataset_dir:
        dataset_dir = Path(args.dataset_dir)
        csv_paths = sorted([str(p) for p in dataset_dir.rglob(args.dataset_pattern)])
        # Filter out confidence and schema files
        csv_paths = [p for p in csv_paths if not ('confidence' in p or 'schema' in p or 'metadata' in p)]
    else:
        logger.error("Must specify --dataset_dir or --csv_list")
        return
    
    if not csv_paths:
        logger.error("No CSV files found")
        return
    
    logger.info(f"Found {len(csv_paths)} datasets")

    if args.num_partitions > 1:
        size = len(csv_paths) // args.num_partitions
        start = args.partition * size
        end = start + size if args.partition < args.num_partitions - 1 else len(csv_paths)
        csv_paths = csv_paths[start:end]
        logger.info(f"Partition {args.partition}/{args.num_partitions}: using datasets [{start}:{end}] ({len(csv_paths)} total)")

    # Create pretrainer
    pretrainer = CASTPretrainer(
        dataset_csvs=csv_paths,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        global_batch_size=args.global_batch_size,
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        max_datasets=args.max_datasets,
        tabular_layers=args.tabular_layers,
        unfreeze_layers=args.unfreeze_layers,
        val_ratio=args.val_ratio,
    )
    
    # Run pretraining
    pretrainer.pretrain()


if __name__ == "__main__":
    main()
