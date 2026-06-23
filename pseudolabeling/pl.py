import argparse
import glob
import json
import math
import os
import random
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import norm as scipy_norm
from tqdm import tqdm


def _add_module_root(module_root: Optional[str]) -> None:
    if module_root:
        sys.path.insert(0, module_root)


def _import_model(module_name: str, class_name: str):
    mod = __import__(module_name, fromlist=[class_name, "Feature", "Example"])
    model_cls = getattr(mod, class_name)
    feat_cls = getattr(mod, "Feature")
    ex_cls = getattr(mod, "Example") if hasattr(mod, "Example") else None
    return model_cls, feat_cls, ex_cls


def load_npz_dataset(npz_path: str, metadata_path: Optional[str] = None) -> Tuple[pd.DataFrame, Optional[Dict[str, Any]]]:
    data = np.load(npz_path, allow_pickle=True)

    if "data" in data:
        arr = data["data"]
    elif "X" in data:
        arr = data["X"]
    elif "features" in data:
        arr = data["features"]
    else:
        keys = list(data.keys())
        if not keys:
            raise ValueError("NPZ file contains no arrays.")
        arr = data[keys[0]]

    if arr.ndim != 2:
        raise ValueError("Expected a 2D array in NPZ dataset.")

    metadata = None
    if metadata_path is not None:
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

    if metadata and isinstance(metadata, dict) and "features" in metadata and isinstance(metadata["features"], list):
        feature_names = []
        for f in metadata["features"]:
            if isinstance(f, dict) and "name" in f:
                feature_names.append(str(f["name"]))
        if len(feature_names) == 0:
            col_names = [f"feature_{i}" for i in range(arr.shape[1])]
        elif len(feature_names) >= arr.shape[1]:
            col_names = feature_names[: arr.shape[1]]
        else:
            col_names = feature_names + [f"feature_{i}" for i in range(len(feature_names), arr.shape[1])]
    else:
        col_names = [f"feature_{i}" for i in range(arr.shape[1])]

    return pd.DataFrame(arr, columns=col_names), metadata


def infer_features_from_csv(df: pd.DataFrame, feature_cls, min_rows: int = 10) -> List[Any]:
    features: List[Any] = []
    for col in df.columns:
        col_data = df[col].dropna()
        if len(col_data) < min_rows:
            continue

        numeric_vals = pd.to_numeric(col_data, errors="coerce").dropna()
        if len(numeric_vals) >= len(col_data) * 0.8:
            min_val = float(numeric_vals.min())
            max_val = float(numeric_vals.max())
            if not (math.isfinite(min_val) and math.isfinite(max_val) and min_val < max_val):
                raise ValueError(f"Invalid numeric range for column: {col}")
            features.append(
                feature_cls(
                    name=str(col),
                    description=f"Feature {col}",
                    dtype="continuous",
                    value_range=(min_val, max_val),
                )
            )
            continue

        unique_vals = col_data.astype(str).unique()
        if 2 <= len(unique_vals) <= 50:
            choices = [str(v) for v in unique_vals]
            features.append(
                feature_cls(
                    name=str(col),
                    description=f"Feature {col}",
                    dtype="categorical",
                    choices=choices,
                )
            )

    return features


def discover_datasets(data_dirs: List[str], max_datasets: int = 0) -> List[str]:
    paths: List[str] = []
    for d in data_dirs:
        paths.extend(glob.glob(os.path.join(d, "**", "*.csv"), recursive=True))
        paths.extend(glob.glob(os.path.join(d, "**", "dataset.npz"), recursive=True))
    if max_datasets > 0:
        paths = paths[:max_datasets]
    return paths


@torch.no_grad()
def compute_target_repr(
    model,
    target_feature,
    observed_features: List[Any],
    observed_values: List[Any],
    dataset_context: str,
) -> torch.Tensor:
    device = next(model.parameters()).device

    obs_atoms_list: List[torch.Tensor] = []
    for feat, val in zip(observed_features, observed_values):
        if val is None:
            raise ValueError("Observed value is None in strict mode.")
        if isinstance(val, float) and math.isnan(val):
            raise ValueError("Observed value is NaN in strict mode.")
        phi_feat = model.semantic_grounding(feat, device)
        atom = model.atom_processing(feat, val, phi_feat, device)
        obs_atoms_list.append(atom)

    phi_target = model.semantic_grounding(target_feature, device)
    if target_feature.dtype == "continuous":
        placeholder = 0.0
    else:
        if not getattr(target_feature, "choices", None):
            raise ValueError("Categorical target feature has no choices in strict mode.")
        placeholder = target_feature.choices[0]
    target_atom = model.atom_processing(target_feature, placeholder, phi_target, device)

    if obs_atoms_list:
        query_atoms = torch.stack(obs_atoms_list, dim=0)
    else:
        query_atoms = torch.zeros(0, model.model_dim, device=device)

    target_atoms = target_atom.unsqueeze(0)

    if getattr(model, "use_intra_set2set", False) and getattr(model, "intra_set2set", None) is not None:
        if query_atoms.size(0) > 0:
            query_atoms = model.intra_set2set(query_atoms)
        target_atoms = model.intra_set2set(target_atoms)

    context_data_tokens = model._encode_context(dataset_context, device)
    desc = getattr(target_feature, "description", None) or getattr(target_feature, "name", "")
    context_target_tokens = model.shared_text.encode_text_sequence(desc, is_context=False, device=device)

    target_reprs = model.inter_aggregator(
        query_atoms=query_atoms,
        target_atoms=target_atoms,
        support_atoms=[],
        context_data_tokens=context_data_tokens,
        context_target_tokens=context_target_tokens,
    )
    if not isinstance(target_reprs, (list, tuple)) and not torch.is_tensor(target_reprs):
        raise ValueError("Unexpected inter_aggregator output type.")
    return target_reprs[0]


@torch.no_grad()
def predict_feature_vanilla(
    model,
    repr_vec: torch.Tensor,
    feature,
    temperature: float = 1.0,
) -> Tuple[Any, float]:
    device = next(model.parameters()).device

    if feature.dtype == "continuous":
        pi, mu, logvar = model.reg_head(repr_vec.unsqueeze(0))
        pi = pi.squeeze(0)
        mu = mu.squeeze(0)
        logvar = logvar.squeeze(0)

        if pi.ndim != 1 or mu.ndim != 1 or logvar.ndim != 1:
            raise ValueError("reg_head outputs must be 1D per example.")

        comp = torch.multinomial(pi, 1).item()
        std = torch.exp(0.5 * logvar[comp]).item()
        y_norm = mu[comp].item() + random.gauss(0.0, max(std * temperature, 1e-4))
        y_norm = float(max(0.0, min(1.0, y_norm)))

        if not getattr(feature, "value_range", None):
            raise ValueError("Continuous feature missing value_range in strict mode.")
        vmin, vmax = feature.value_range
        pred = y_norm * (vmax - vmin) + vmin
        conf = max(0.0, 1.0 - std)
        return float(pred), float(conf)

    categories = list(getattr(feature, "choices", None) or [])
    if len(categories) == 0:
        raise ValueError("Categorical feature has no choices in strict mode.")
    if "[UNK]" in categories:
        raise ValueError("Choices already contain [UNK]; strict mode expects raw categories only.")
    categories = categories + ["[UNK]"]

    label_vecs = []
    for cat in categories:
        emb = model.shared_text.encode_text(str(cat), is_context=False, device=device)
        label_vecs.append(emb)
    label_vecs = torch.stack(label_vecs, dim=0)
    label_embs = model.cls_head.category_proj(label_vecs)

    logits = model.cls_head.logits(repr_vec.unsqueeze(0), label_embs)
    probs = F.softmax(logits / max(1e-8, temperature), dim=-1).squeeze(0)

    unk_idx = categories.index("[UNK]")
    probs[unk_idx] = 0.0
    if probs.sum().item() <= 0:
        raise ValueError("All categorical probability mass collapsed to UNK/zero in strict mode.")
    probs = probs / probs.sum()

    sampled_idx = torch.multinomial(probs, 1).item()
    pred_cat = categories[sampled_idx]
    if pred_cat == "[UNK]":
        raise ValueError("Sampled UNK in strict mode.")
    conf = float(probs[sampled_idx].item())
    return pred_cat, conf


@torch.no_grad()
def predict_feature_quartile(
    model,
    repr_vec: torch.Tensor,
    feature,
    temperature: float = 1.0,
) -> Tuple[Any, float]:
    if feature.dtype != "continuous":
        return predict_feature_vanilla(model, repr_vec, feature, temperature)

    pi, mu, logvar = model.reg_head(repr_vec.unsqueeze(0))
    pi = pi.squeeze(0).detach().cpu().numpy()
    mu = mu.squeeze(0).detach().cpu().numpy()
    logvar = logvar.squeeze(0).detach().cpu().numpy()
    std = np.exp(0.5 * logvar)

    if not getattr(feature, "value_range", None):
        raise ValueError("Continuous feature missing value_range in strict mode.")
    vmin, vmax = feature.value_range

    bin_edges = [0.0, 0.25, 0.50, 0.75, 1.0]
    bin_probs = np.zeros(4, dtype=np.float64)

    for b in range(4):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        for k in range(len(pi)):
            s = max(float(std[k]), 1e-6)
            cdf_hi = scipy_norm.cdf(hi, loc=float(mu[k]), scale=s)
            cdf_lo = scipy_norm.cdf(lo, loc=float(mu[k]), scale=s)
            bin_probs[b] += float(pi[k]) * float(cdf_hi - cdf_lo)

    if not np.isfinite(bin_probs).all():
        raise ValueError("Non-finite quartile probabilities computed.")
    bin_probs = np.maximum(bin_probs, 1e-12)

    log_probs = np.log(bin_probs) / max(float(temperature), 1e-8)
    log_probs -= log_probs.max()
    bin_probs = np.exp(log_probs)
    bin_probs /= bin_probs.sum()

    chosen_bin = int(np.random.choice(4, p=bin_probs))
    lo_norm = bin_edges[chosen_bin]
    hi_norm = bin_edges[chosen_bin + 1]
    y_norm = float(np.random.uniform(lo_norm, hi_norm))

    pred = y_norm * (vmax - vmin) + vmin
    conf = float(bin_probs[chosen_bin])
    return float(pred), conf


@torch.no_grad()
def predict_feature_cast(
    model,
    repr_vec: torch.Tensor,
    feature,
    cast_pool: Dict[str, torch.Tensor],
    temperature: float = 1.0,
    cast_k: int = 10,
    cast_alpha: float = 1.0,
    cast_metric: str = "cosine",
) -> Tuple[Any, float]:
    device = next(model.parameters()).device

    if feature.dtype == "continuous":
        return predict_feature_vanilla(model, repr_vec, feature, temperature)

    categories = list(getattr(feature, "choices", None) or [])
    if len(categories) == 0:
        raise ValueError("Categorical feature has no choices in strict mode.")
    if "[UNK]" in categories:
        raise ValueError("Choices already contain [UNK]; strict mode expects raw categories only.")
    categories = categories + ["[UNK]"]

    label_vecs = []
    for cat in categories:
        emb = model.shared_text.encode_text(str(cat), is_context=False, device=device)
        label_vecs.append(emb)
    label_vecs = torch.stack(label_vecs, dim=0)
    label_embs = model.cls_head.category_proj(label_vecs)

    logits = model.cls_head.logits(repr_vec.unsqueeze(0), label_embs)
    probs = F.softmax(logits / max(1e-8, temperature), dim=-1).squeeze(0)

    unk_idx = categories.index("[UNK]")
    probs[unk_idx] = 0.0
    if probs.sum().item() <= 0:
        raise ValueError("All categorical probability mass collapsed to UNK/zero in strict mode.")
    probs = probs / probs.sum()

    densities: List[torch.Tensor] = []
    for cat in categories:
        if cat == "[UNK]":
            densities.append(torch.tensor(0.0, device=device))
            continue

        if str(cat) not in cast_pool:
            raise KeyError(f"Missing CAST pool for category: {cat}")

        pool = cast_pool[str(cat)]
        if pool.ndim != 2 or pool.size(0) == 0:
            raise ValueError(f"Invalid CAST pool tensor for category: {cat}")

        if cast_metric == "cosine":
            q = F.normalize(repr_vec, dim=0)
            P = F.normalize(pool.to(device), dim=1)
            sims = torch.matmul(P, q)
            k = min(int(cast_k), int(sims.numel()))
            if k <= 0:
                raise ValueError("Invalid cast_k for cosine metric.")
            topk = sims.topk(k=k).values
            density = (topk.mean() + 1.0) / 2.0
        elif cast_metric == "euclidean":
            diff = pool.to(device) - repr_vec.unsqueeze(0)
            dists = torch.linalg.norm(diff, dim=1)
            k = min(int(cast_k), int(dists.numel()))
            if k <= 0:
                raise ValueError("Invalid cast_k for euclidean metric.")
            topk = dists.topk(k=k, largest=False).values
            density = 1.0 / (1e-6 + topk.mean())
        else:
            raise ValueError("cast_metric must be 'cosine' or 'euclidean'.")

        densities.append(density)

    dens = torch.stack(densities) + 1e-12
    cast_weights = dens.pow(float(cast_alpha))
    cast_scores = probs * cast_weights
    if cast_scores.sum().item() <= 0:
        raise ValueError("CAST calibration produced zero total mass in strict mode.")
    cast_probs = cast_scores / cast_scores.sum()

    cast_probs[unk_idx] = 0.0
    if cast_probs.sum().item() <= 0:
        raise ValueError("CAST calibration collapsed to UNK/zero in strict mode.")
    cast_probs = cast_probs / cast_probs.sum()

    sampled_idx = torch.multinomial(cast_probs, 1).item()
    pred_cat = categories[sampled_idx]
    if pred_cat == "[UNK]":
        raise ValueError("Sampled UNK in strict mode.")
    conf = float(cast_probs[sampled_idx].item())
    return pred_cat, conf


@torch.no_grad()
def build_cast_indices(
    df: pd.DataFrame,
    features: List[Any],
    model,
    dataset_context: str,
    max_rows: Optional[int],
    max_per_class: int,
) -> Dict[int, Dict[str, torch.Tensor]]:
    device = next(model.parameters()).device
    cat_feature_indices = [i for i, f in enumerate(features) if f.dtype == "categorical"]
    if len(cat_feature_indices) == 0:
        raise ValueError("CAST requested but no categorical features were inferred.")

    usable = df if max_rows is None else df.head(int(max_rows))
    indices: Dict[int, Dict[str, List[torch.Tensor]]] = {i: defaultdict(list) for i in cat_feature_indices}

    for _, row in tqdm(usable.iterrows(), total=len(usable), desc="cast_index"):
        vals: List[Any] = []
        for feat in features:
            if feat.name not in row:
                raise KeyError(f"Missing column for feature: {feat.name}")
            val = row[feat.name]
            if pd.isna(val):
                raise ValueError("NaN encountered while building CAST indices in strict mode.")
            if feat.dtype == "continuous":
                v = float(val)
                if not math.isfinite(v):
                    raise ValueError("Non-finite continuous value encountered in CAST indexing.")
                vals.append(v)
            else:
                sval = str(val)
                if getattr(feat, "choices", None) and sval not in feat.choices:
                    raise ValueError("Value not in declared choices encountered in CAST indexing.")
                vals.append(sval)

        phi_list = [model.semantic_grounding(f, device) for f in features]
        atom_list = [model.atom_processing(f, v, phi_list[i], device) for i, (f, v) in enumerate(zip(features, vals))]

        for f_idx in cat_feature_indices:
            true_label = str(vals[f_idx])
            if len(indices[f_idx][true_label]) >= int(max_per_class):
                continue

            obs_atoms = [atom_list[i] for i in range(len(features)) if i != f_idx]
            query_atoms = torch.stack(obs_atoms, dim=0) if obs_atoms else torch.zeros(0, model.model_dim, device=device)
            target_atom = atom_list[f_idx].unsqueeze(0)

            if getattr(model, "use_intra_set2set", False) and getattr(model, "intra_set2set", None) is not None:
                if query_atoms.size(0) > 0:
                    query_atoms = model.intra_set2set(query_atoms)
                target_atom = model.intra_set2set(target_atom)

            context_data_tokens = model._encode_context(dataset_context, device)
            desc = getattr(features[f_idx], "description", None) or getattr(features[f_idx], "name", "")
            context_target_tokens = model.shared_text.encode_text_sequence(desc, is_context=False, device=device)

            target_reprs = model.inter_aggregator(
                query_atoms=query_atoms,
                target_atoms=target_atom,
                support_atoms=[],
                context_data_tokens=context_data_tokens,
                context_target_tokens=context_target_tokens,
            )
            indices[f_idx][true_label].append(target_reprs[0].detach())

    stacked: Dict[int, Dict[str, torch.Tensor]] = {}
    for f_idx, class_map in indices.items():
        stacked[f_idx] = {}
        for cls, vecs in class_map.items():
            if len(vecs) == 0:
                raise ValueError("Empty CAST class pool encountered in strict mode.")
            stacked[f_idx][cls] = torch.stack(vecs, dim=0)

    return stacked


@torch.no_grad()
def autoregressive_pseudolabel_row(
    model,
    features: List[Any],
    original_values: List[Any],
    dataset_context: str,
    method: str,
    cast_indices: Optional[Dict[int, Dict[str, torch.Tensor]]],
    temperature: float,
    cast_k: int,
    cast_alpha: float,
    cast_metric: str,
    use_quartile: bool,
) -> Dict[str, Any]:
    if len(original_values) != len(features):
        raise ValueError("original_values length mismatch.")

    predicted_values: List[Any] = [None] * len(features)
    confidences: List[float] = [0.0] * len(features)

    order = list(range(len(features)))
    random.shuffle(order)

    for step, f_idx in enumerate(order):
        feat = features[f_idx]

        obs_feats = [features[prev_idx] for prev_idx in order[:step]]
        obs_vals = [predicted_values[prev_idx] for prev_idx in order[:step]]

        if any(v is None for v in obs_vals):
            raise ValueError("Autoregressive context contains None in strict mode.")

        repr_vec = compute_target_repr(model, feat, obs_feats, obs_vals, dataset_context)

        if use_quartile and feat.dtype == "continuous":
            pred_val, conf = predict_feature_quartile(model, repr_vec, feat, temperature)
        elif method == "cast":
            if cast_indices is None or f_idx not in cast_indices:
                raise ValueError("CAST method requested but missing cast indices for feature.")
            pred_val, conf = predict_feature_cast(
                model=model,
                repr_vec=repr_vec,
                feature=feat,
                cast_pool=cast_indices[f_idx],
                temperature=temperature,
                cast_k=cast_k,
                cast_alpha=cast_alpha,
                cast_metric=cast_metric,
            )
        else:
            pred_val, conf = predict_feature_vanilla(model, repr_vec, feat, temperature)

        predicted_values[f_idx] = pred_val
        confidences[f_idx] = float(conf)

    return {"predictions": predicted_values, "confidences": confidences}


def _load_dataframe(file_path: str) -> pd.DataFrame:
    if file_path.endswith(".npz"):
        npz_dir = os.path.dirname(file_path)
        meta_path = os.path.join(npz_dir, "final_metadata.json")
        df, _ = load_npz_dataset(file_path, meta_path)
        return df

    if file_path.endswith(".csv"):
        return pd.read_csv(file_path)

    raise ValueError("Unsupported file type. Expected .csv or .npz")


def process_dataset(
    file_path: str,
    model,
    feature_cls,
    output_dir: str,
    method: str,
    max_rows: Optional[int],
    temperature: float,
    cast_k: int,
    cast_alpha: float,
    cast_metric: str,
    cast_index_rows: Optional[int],
    cast_max_per_class: int,
    use_quartile: bool,
) -> str:
    df = _load_dataframe(file_path)
    if max_rows is not None:
        df = df.head(int(max_rows))

    features = infer_features_from_csv(df, feature_cls)
    if len(features) < 2:
        raise ValueError("Too few inferred features.")

    dataset_context = "Dataset"

    cast_indices = None
    if method == "cast":
        rows_for_index = cast_index_rows if cast_index_rows is not None else len(df)
        cast_indices = build_cast_indices(
            df=df,
            features=features,
            model=model,
            dataset_context=dataset_context,
            max_rows=int(rows_for_index),
            max_per_class=int(cast_max_per_class),
        )

    results: List[Dict[str, Any]] = []
    for row_idx, row in tqdm(df.iterrows(), total=len(df), desc="pseudolabel"):
        vals: List[Any] = []
        for feat in features:
            if feat.name not in row:
                raise KeyError(f"Missing column for feature: {feat.name}")
            val = row[feat.name]
            if pd.isna(val):
                raise ValueError("NaN encountered in strict mode.")
            if feat.dtype == "continuous":
                v = float(val)
                if not math.isfinite(v):
                    raise ValueError("Non-finite continuous value encountered in strict mode.")
                vals.append(v)
            else:
                sval = str(val)
                if getattr(feat, "choices", None) and sval not in feat.choices:
                    raise ValueError("Value not in declared choices encountered in strict mode.")
                vals.append(sval)

        out = autoregressive_pseudolabel_row(
            model=model,
            features=features,
            original_values=vals,
            dataset_context=dataset_context,
            method=method,
            cast_indices=cast_indices,
            temperature=temperature,
            cast_k=cast_k,
            cast_alpha=cast_alpha,
            cast_metric=cast_metric,
            use_quartile=use_quartile,
        )

        row_result: Dict[str, Any] = {"row_id": int(row_idx)}
        for i, feat in enumerate(features):
            row_result[f"{feat.name}_original"] = vals[i]
            row_result[f"{feat.name}_pseudo"] = out["predictions"][i]
            row_result[f"{feat.name}_confidence"] = float(out["confidences"][i])
            if feat.dtype == "categorical":
                row_result[f"{feat.name}_correct"] = (str(out["predictions"][i]) == str(vals[i]))
        results.append(row_result)

    if len(results) == 0:
        raise ValueError("No results produced.")

    results_df = pd.DataFrame(results)
    os.makedirs(output_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(file_path))[0]
    out_path = os.path.join(output_dir, f"{base}_ar_pseudolabels_{method}.csv")
    results_df.to_csv(out_path, index=False)

    return out_path


def _build_model_from_checkpoint(model_cls, checkpoint: Dict[str, Any], device: torch.device):
    model = model_cls(
        model_dim=checkpoint.get("model_dim", 768),
        num_heads=checkpoint.get("num_heads", 8),
        num_inds=checkpoint.get("num_inds", 32),
        mask_prob=checkpoint.get("mask_prob", 0.40),
        max_targets=checkpoint.get("max_targets", 3),
        intra_layers=checkpoint.get("intra_layers", 2),
        inter_layers=checkpoint.get("inter_layers", 2),
        use_intra_set2set=checkpoint.get("use_intra_set2set", True),
        use_dataset_description=checkpoint.get("use_dataset_description", True),
    ).to(device)
    if "model_state_dict" not in checkpoint:
        raise KeyError("Checkpoint missing 'model_state_dict'.")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--data_dirs", type=str, nargs="+", default=None)
    parser.add_argument("--data_dirs_file", type=str, default=None)
    parser.add_argument("--max_datasets", type=int, default=0)
    parser.add_argument("--max_rows", type=int, default=None)

    parser.add_argument("--method", type=str, default="vanilla", choices=["vanilla", "cast"])
    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--cast_k", type=int, default=10)
    parser.add_argument("--cast_alpha", type=float, default=1.0)
    parser.add_argument("--cast_metric", type=str, default="cosine", choices=["cosine", "euclidean"])
    parser.add_argument("--cast_index_rows", type=int, default=30)
    parser.add_argument("--cast_max_per_class", type=int, default=30)

    parser.add_argument("--use_quartile", action="store_true")

    parser.add_argument("--module_root", type=str, default=None)
    parser.add_argument("--model_module", type=str, default="aspire_enhaced_v2_icml")
    parser.add_argument("--model_class", type=str, default="ASPIREEnhanced")

    args = parser.parse_args()

    _add_module_root(args.module_root)
    model_cls, feature_cls, _ = _import_model(args.model_module, args.model_class)

    os.makedirs(args.output_dir, exist_ok=True)

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = _build_model_from_checkpoint(model_cls, checkpoint, device)

    if args.data_dirs_file:
        with open(args.data_dirs_file, "r", encoding="utf-8") as f:
            raw_paths = [line.strip() for line in f if line.strip()]
        dataset_paths: List[str] = []
        for p in raw_paths:
            if os.path.isdir(p):
                npz = os.path.join(p, "dataset.npz")
                if os.path.exists(npz):
                    dataset_paths.append(npz)
                else:
                    dataset_paths.extend(glob.glob(os.path.join(p, "*.csv")))
            else:
                dataset_paths.append(p)
        if args.max_datasets > 0:
            dataset_paths = dataset_paths[: args.max_datasets]
    elif args.data_dirs:
        dataset_paths = discover_datasets(args.data_dirs, args.max_datasets)
    else:
        raise ValueError("Must provide --data_dirs or --data_dirs_file.")

    if len(dataset_paths) == 0:
        raise ValueError("No datasets discovered.")

    for path in dataset_paths:
        process_dataset(
            file_path=path,
            model=model,
            feature_cls=feature_cls,
            output_dir=args.output_dir,
            method=args.method,
            max_rows=args.max_rows,
            temperature=args.temperature,
            cast_k=args.cast_k,
            cast_alpha=args.cast_alpha,
            cast_metric=args.cast_metric,
            cast_index_rows=args.cast_index_rows if args.method == "cast" else None,
            cast_max_per_class=args.cast_max_per_class,
            use_quartile=args.use_quartile,
        )


if __name__ == "__main__":
    main()