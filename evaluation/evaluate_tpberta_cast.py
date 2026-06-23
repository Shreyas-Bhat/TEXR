"""
Evaluate CAST-trained TP-BERTa on benchmark datasets.
Supports 5-shot and finetuning evaluation modes.
Uses the same 15 benchmark datasets as CM2 evaluation.
"""
import argparse
import os
import sys
import json
import shutil
import tempfile
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

# ---- TP-BERTa imports ----
TPBERTA_DIR = "/playpen-nvme/scribble/shbhat/tp-berta"
sys.path.insert(0, TPBERTA_DIR)
from transformers import RobertaConfig
from bin.tpberta_modeling import TPBertaForClassification
from lib.data import Dataset2
from lib.data_utils import DataConfig, data_preproc, encode_single_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def pad_and_cat(tensors_a, tensors_b):
    """Concatenate two dicts of tensors, padding dim=1 to match if needed."""
    result = {}
    for k in tensors_a:
        a, b = tensors_a[k], tensors_b[k]
        if a.dim() >= 2 and b.dim() >= 2 and a.shape[1] != b.shape[1]:
            max_len = max(a.shape[1], b.shape[1])
            if a.shape[1] < max_len:
                pad_size = max_len - a.shape[1]
                a = F.pad(a, (0, pad_size), value=0)
            if b.shape[1] < max_len:
                pad_size = max_len - b.shape[1]
                b = F.pad(b, (0, pad_size), value=0)
        result[k] = torch.cat([a, b], dim=0)
    return result

# ---- Benchmark datasets (same as CM2 evaluation) ----
BENCHMARK_DATASETS = {
    'Breast': {
        'path': '/playpen-nvme/scribble/shbhat/OpenTabs/OpenTabs-Latest/clean_labeled_dataset/0142_BNG(breast-w).csv',
        'format': 'csv',
        'max_rows': 10000,
    },
    'Cmc': {
        'path': '/playpen-nvme/scribble/shbhat/OpenTabs/OpenTabs-Latest/clean_labeled_dataset/0146_BNG(cmc).csv',
        'format': 'csv',
        'max_rows': 10000,
    },
    'Diabetes': {
        'path': '/playpen-nvme/scribble/shbhat/OpenTabs/OpenTabs-Latest/clean_labeled_dataset/diabetes.csv',
        'format': 'csv',
    },
    'Vehicle': {
        'path': '/playpen-nvme/scribble/shbhat/OpenTabs/OpenTabs-Latest/clean_labeled_dataset/1347_BNG(vehicle).csv',
        'format': 'csv',
        'max_rows': 10000,
    },
    'Satimage': {
        'path': '/playpen-nvme/scribble/shbhat/OpenTabs/sat.tst',
        'format': 'satimage',
    },
    'Sick': {
        'path': '/playpen-nvme/scribble/shbhat/OpenTabs/OpenTabs-Latest/clean_labeled_dataset/1671_sick.csv',
        'format': 'csv',
    },
    'Pc1': {
        'path': '/playpen-nvme/scribble/shbhat/OpenTabs/OpenTabs-Latest/clean_labeled_dataset/0649_pc1.csv',
        'format': 'csv',
    },
    'Adult': {
        'path': '/playpen-nvme/scribble/shbhat/OpenTabs/OpenTabs-Latest/clean_labeled_dataset/1674_adult.csv',
        'format': 'csv',
        'max_rows': 10000,
    },
    'PhishingWebsites': {
        'path': '/playpen-nvme/scribble/shbhat/OpenTabs/OpenTabs-Latest/clean_labeled_dataset/0929_PhishingWebsites.csv',
        'format': 'csv',
        'max_rows': 10000,
    },
    'Cylinder-bands': {
        'path': '/playpen-nvme/scribble/shbhat/OpenTabs/OpenTabs-Latest/clean_labeled_dataset/0943_cylinder-bands.csv',
        'format': 'csv',
    },
    'MiceProtein': {
        'path': '/playpen-nvme/scribble/shbhat/OpenTabs/OpenTabs-Latest/clean_labeled_dataset/1021_MiceProtein.csv',
        'format': 'csv',
    },
    'Car': {
        'path': '/playpen-nvme/scribble/shbhat/OpenTabs/OpenTabs-Latest/clean_labeled_dataset/1023_car.csv',
        'format': 'csv',
    },
    'Segment': {
        'path': '/playpen-nvme/scribble/shbhat/OpenTabs/OpenTabs-Latest/clean_labeled_dataset/Segmentation.csv',
        'format': 'csv',
    },
    'Porto-seguro': {
        'path': '/playpen-nvme/scribble/shbhat/OpenTabs/OpenTabs-Latest/clean_labeled_dataset/1355_porto-seguro.csv',
        'format': 'csv',
        'max_rows': 10000,
    },
    'Amazon': {
        'path': '/playpen-nvme/scribble/shbhat/OpenTabs/OpenTabs-Latest/clean_labeled_dataset/0919_Amazon_employee_access.csv',
        'format': 'csv',
        'max_rows': 10000,
    },
}


def load_dataset_to_csv(name, config, tmp_dir):
    """Load a benchmark dataset and save as CSV in tmp_dir for TP-BERTa pipeline."""
    path = config['path']
    fmt = config.get('format', 'csv')

    if fmt == 'satimage':
        # Space-separated, last column is target
        data = []
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) > 1:
                    data.append([float(x) for x in parts])
        df = pd.DataFrame(data)
        n_cols = len(df.columns)
        df.columns = [f'feat_{i}' for i in range(n_cols - 1)] + ['target']
        df['target'] = df['target'].astype(int)
    elif fmt == 'csv':
        df = pd.read_csv(path, low_memory=False)
    else:
        raise ValueError(f"Unknown format: {fmt}")

    # Subsample large datasets
    max_rows = config.get('max_rows', None)
    if max_rows and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=42)

    # Save to tmp dir
    csv_path = os.path.join(tmp_dir, f"{name}.csv")
    df.to_csv(csv_path, index=False)

    return csv_path, df


def load_pretrained_backbone(pretrain_dir, model_suffix, num_classes, device, backbone_pt=None):
    """Load pretrained TP-BERTa backbone into a TPBertaForClassification model.
    
    If backbone_pt is provided, load weights from that raw .pt file instead of
    pretrain_dir/model_suffix/pytorch_model.bin.
    """
    config = RobertaConfig.from_pretrained(pretrain_dir)
    model = TPBertaForClassification(config, num_class=num_classes)

    if backbone_pt and os.path.exists(backbone_pt):
        pretrained_path = backbone_pt
    else:
        pretrained_path = Path(pretrain_dir) / model_suffix / "pytorch_model.bin"

    pretrained_path = Path(pretrained_path)
    if pretrained_path.exists():
        state_dict = torch.load(pretrained_path, map_location="cpu")
        backbone_keys = {
            k: v for k, v in state_dict.items()
            if (k.startswith("tpberta.") or k.startswith("ranker."))
            and "position_ids" not in k
        }
        # Resize embeddings if checkpoint has different vocab size
        emb_key = "tpberta.embeddings.word_embeddings.weight"
        if emb_key in backbone_keys:
            ckpt_vocab = backbone_keys[emb_key].shape[0]
            model_vocab = config.vocab_size
            if ckpt_vocab != model_vocab:
                logger.info(f"Resizing vocab embeddings: {model_vocab} -> {ckpt_vocab}")
                model.tpberta.resize_token_embeddings(ckpt_vocab)
        missing, unexpected = model.load_state_dict(backbone_keys, strict=False)
        logger.info(f"Loaded backbone from {pretrained_path}: {len(backbone_keys)} keys, "
                     f"{len(missing)} missing, {len(unexpected)} unexpected")
    else:
        logger.warning(f"Pretrained weights not found at {pretrained_path}")

    return model.to(device)


def finetune_and_evaluate(
    encoded, dataset, pretrain_dir, model_suffix, device,
    train_split_name="train", eval_split_name="val",
    finetune_epochs=50, lr=1e-4, batch_size=32,
    n_shot=None, seed=42, backbone_pt=None,
):
    """
    Fine-tune TP-BERTa on a dataset and evaluate.
    
    If n_shot is set, only use n_shot samples per class from train split.
    Remaining train samples are merged into eval split.
    """
    task_type = dataset.task_type.value
    n_classes = dataset.n_classes if task_type != "regression" else 1

    if train_split_name not in encoded:
        return {"error": "no train split"}

    # Get train data
    train_tensors = {k: torch.as_tensor(v) for k, v in encoded[train_split_name].items()}
    train_labels = train_tensors.pop("labels")
    n_train_full = len(train_labels)

    # Get eval data
    eval_tensors = None
    eval_labels = None
    if eval_split_name in encoded:
        eval_tensors = {k: torch.as_tensor(v) for k, v in encoded[eval_split_name].items()}
        eval_labels = eval_tensors.pop("labels")

    # For n-shot: subsample train, merge rest into eval
    if n_shot is not None and task_type != "regression":
        labels_np = train_labels.numpy()
        unique_classes = np.unique(labels_np)
        
        shot_indices = []
        remaining_indices = []
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        for cls in unique_classes:
            cls_idx = np.where(labels_np == cls)[0]
            if len(cls_idx) < n_shot:
                # Use all available for this class
                shot_indices.extend(cls_idx.tolist())
            else:
                np.random.shuffle(cls_idx)
                shot_indices.extend(cls_idx[:n_shot].tolist())
                remaining_indices.extend(cls_idx[n_shot:].tolist())
        
        shot_indices = sorted(shot_indices)
        remaining_indices = sorted(remaining_indices)
        
        # Remaining train samples -> merge into eval
        if remaining_indices and eval_tensors is not None:
            remaining_t = {k: v[remaining_indices] for k, v in train_tensors.items()}
            eval_tensors = pad_and_cat(eval_tensors, remaining_t)
            eval_labels = torch.cat([eval_labels, train_labels[remaining_indices]], dim=0)
        elif remaining_indices:
            eval_tensors = {k: v[remaining_indices] for k, v in train_tensors.items()}
            eval_labels = train_labels[remaining_indices]
        
        # Subset train to shot_indices
        train_tensors = {k: v[shot_indices] for k, v in train_tensors.items()}
        train_labels = train_labels[shot_indices]
        
        logger.info(f"  n-shot: {len(shot_indices)} train, {len(eval_labels)} eval")

    n_train = len(train_labels)
    if n_train == 0:
        return {"error": "empty train set"}

    # Also merge test split into eval if available (for more comprehensive eval)
    if "test" in encoded and eval_tensors is not None:
        test_t = {k: torch.as_tensor(v) for k, v in encoded["test"].items()}
        test_l = test_t.pop("labels")
        eval_tensors = pad_and_cat(eval_tensors, test_t)
        eval_labels = torch.cat([eval_labels, test_l], dim=0)

    if eval_tensors is None or len(eval_labels) == 0:
        return {"error": "no eval data"}

    # Load model
    model = load_pretrained_backbone(pretrain_dir, model_suffix, n_classes, device, backbone_pt=backbone_pt)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Fine-tune
    best_loss = float('inf')
    best_state = None
    patience = 10
    patience_counter = 0

    for epoch in range(finetune_epochs):
        model.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            batch = {k: v[idx].to(device) for k, v in train_tensors.items()}
            labels = train_labels[idx].to(device)

            optimizer.zero_grad()
            logits, _ = model(**batch)

            if task_type == "regression":
                loss = F.mse_loss(logits.squeeze(-1), labels.float())
            elif n_classes == 1:  # binclass
                loss = F.binary_cross_entropy_with_logits(logits.squeeze(-1), labels.float())
            else:
                loss = F.cross_entropy(logits, labels.long())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        # Early stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            logger.info(f"  Early stopping at epoch {epoch + 1}")
            break

    # Load best state
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    # Evaluate
    model.eval()
    n_eval = len(eval_labels)
    all_logits = []

    with torch.no_grad():
        for i in range(0, n_eval, batch_size):
            batch = {k: v[i:i + batch_size].to(device) for k, v in eval_tensors.items()}
            logits, _ = model(**batch)
            all_logits.append(logits.cpu())

    logits_all = torch.cat(all_logits)
    y_true = eval_labels.numpy()

    # Compute predictions and metrics
    results = {"n_train": n_train, "n_test": n_eval, "task_type": task_type, "n_classes": n_classes}

    if task_type == "regression":
        preds = logits_all.squeeze(-1).numpy()
        from sklearn.metrics import mean_squared_error, r2_score
        results["mse"] = float(mean_squared_error(y_true, preds))
        results["r2"] = float(r2_score(y_true, preds))
    elif n_classes == 1:  # binclass
        probs = torch.sigmoid(logits_all.squeeze(-1)).numpy()
        preds = (probs > 0.5).astype(int)
        results["accuracy"] = float(accuracy_score(y_true.astype(int), preds))
        results["f1_score"] = float(f1_score(y_true.astype(int), preds, average='macro'))
        try:
            results["auc"] = float(roc_auc_score(y_true.astype(int), probs))
        except ValueError:
            results["auc"] = None
    else:  # multiclass
        probs = torch.softmax(logits_all, dim=1).numpy()
        preds = np.argmax(probs, axis=1)
        results["accuracy"] = float(accuracy_score(y_true.astype(int), preds))
        results["f1_score"] = float(f1_score(y_true.astype(int), preds, average='macro'))
        try:
            results["auc"] = float(roc_auc_score(y_true.astype(int), probs, multi_class='ovr'))
        except ValueError:
            results["auc"] = None

    if n_shot is not None:
        results["n_shot"] = n_shot
    results["epochs_trained"] = epoch + 1

    del model
    torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate CAST-trained TP-BERTa")
    parser.add_argument('--pretrain_dir', type=str, required=True,
                        help='Path to TP-BERTa pretrained checkpoint directory')
    parser.add_argument('--model_suffix', type=str, default='pytorch_models/best',
                        help='Subdirectory containing pytorch_model.bin')
    parser.add_argument('--settings', type=str, nargs='+', default=['few_shot', 'finetuning'],
                        choices=['few_shot', 'finetuning'],
                        help='Evaluation settings to run')
    parser.add_argument('--num_shots', type=int, default=5)
    parser.add_argument('--fewshot_epochs', type=int, default=50)
    parser.add_argument('--finetune_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--output_dir', type=str, default='./tpberta_cast_eval_results')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--backbone_pt', type=str, default=None,
                        help='Path to raw .pt backbone weights file (alternative to model_suffix)')
    parser.add_argument('--datasets', type=str, nargs='+', default=None,
                        help='Specific datasets to evaluate (default: all)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    logger.info(f"Pretrain dir: {args.pretrain_dir}")
    logger.info(f"Settings: {args.settings}")

    # Select datasets
    if args.datasets:
        eval_datasets = {k: v for k, v in BENCHMARK_DATASETS.items() if k in args.datasets}
    else:
        eval_datasets = BENCHMARK_DATASETS

    all_results = {s: {} for s in args.settings}

    for ds_name, ds_config in eval_datasets.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Dataset: {ds_name}")
        logger.info(f"{'='*60}")

        # Create temp dir for this dataset
        tmp_dir = tempfile.mkdtemp(prefix=f"tpberta_eval_{ds_name}_")
        # Create feature_names.json
        json.dump({}, open(os.path.join(tmp_dir, 'feature_names.json'), 'w'))

        try:
            # Load dataset and save as CSV
            csv_path, df = load_dataset_to_csv(ds_name, ds_config, tmp_dir)
            logger.info(f"  Loaded: {len(df)} rows, {len(df.columns)} cols")

            # Build DataConfig for this temp directory
            data_config = DataConfig.from_pretrained(
                args.pretrain_dir,
                data_dir=tmp_dir,
                train_ratio=0.8,
                pre_train=False,
                batch_size=args.batch_size,
                preproc_type="lm",
            )

            # Preprocess and encode
            dataset = data_preproc(ds_name, data_config, tt=None)
            encoded, dataset = encode_single_dataset(dataset, data_config)

            task_type = dataset.task_type.value
            n_classes = dataset.n_classes if task_type != "regression" else 1
            logger.info(f"  Task: {task_type}, Classes: {n_classes}")

            for setting in args.settings:
                logger.info(f"\n  --- {setting} ---")

                if setting == 'few_shot':
                    result = finetune_and_evaluate(
                        encoded, dataset,
                        args.pretrain_dir, args.model_suffix, device,
                        finetune_epochs=args.fewshot_epochs,
                        lr=args.lr, batch_size=args.batch_size,
                        n_shot=args.num_shots, seed=args.seed,
                        backbone_pt=args.backbone_pt,
                    )
                elif setting == 'finetuning':
                    result = finetune_and_evaluate(
                        encoded, dataset,
                        args.pretrain_dir, args.model_suffix, device,
                        finetune_epochs=args.finetune_epochs,
                        lr=args.lr, batch_size=args.batch_size,
                        n_shot=None, seed=args.seed,
                        backbone_pt=args.backbone_pt,
                    )

                if "error" in result:
                    logger.warning(f"  SKIP: {result['error']}")
                else:
                    acc = result.get('accuracy', result.get('r2', 'N/A'))
                    f1 = result.get('f1_score', 'N/A')
                    auc = result.get('auc', 'N/A')
                    logger.info(f"  Acc={acc}, F1={f1}, AUC={auc}")

                all_results[setting][ds_name] = result

        except Exception as e:
            logger.error(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            for setting in args.settings:
                all_results[setting][ds_name] = {"error": str(e)}
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("RESULTS SUMMARY")
    logger.info(f"{'='*60}")

    for setting in args.settings:
        logger.info(f"\n--- {setting.upper()} ---")
        accs, f1s = [], []
        for ds_name, result in all_results[setting].items():
            if "error" in result:
                logger.info(f"  {ds_name:20s}: FAILED ({result['error']})")
                continue
            acc = result.get('accuracy', None)
            f1 = result.get('f1_score', None)
            auc = result.get('auc', None)
            if acc is not None:
                accs.append(acc)
            if f1 is not None:
                f1s.append(f1)
            logger.info(f"  {ds_name:20s}: Acc={acc:.4f}  F1={f1:.4f}  AUC={auc if auc else 'N/A'}")
        if accs:
            logger.info(f"  {'MEAN':20s}: Acc={np.mean(accs):.4f}  F1={np.mean(f1s):.4f}")

    # Save results
    for setting in args.settings:
        out_path = os.path.join(args.output_dir, f"tpberta_cast_{setting}_results.json")
        with open(out_path, 'w') as f:
            json.dump(all_results[setting], f, indent=2)
        logger.info(f"Saved {setting} results to {out_path}")

    # Save combined
    combined_path = os.path.join(args.output_dir, "tpberta_cast_all_results.json")
    with open(combined_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Saved combined results to {combined_path}")


if __name__ == "__main__":
    main()
