#!/usr/bin/env python3
"""
Compare Vanilla vs CAST pseudolabelling on a single filtered dataset.
Shows pre-labelling (original) vs post-labelling (pseudo) values for both methods.
"""

import torch
import torch.nn.functional as F
import sys
import os
import json
import math
import numpy as np
import pandas as pd
import random
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src", "universal_inference_machine"))

# Reuse functions from the autoregressive pseudolabel script
from autoregressive_pseudolabel_aspire_new import (
    infer_features_from_csv,
    compute_target_repr,
    predict_feature_vanilla,
    predict_feature_cast,
    build_cast_indices,
    autoregressive_pseudolabel_row,
    ASPIREEnhanced,
    Feature,
)

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Quartile-based continuous prediction
# ─────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_feature_quartile(
    model: ASPIREEnhanced,
    repr_vec: torch.Tensor,
    feature: Feature,
    temperature: float = 1.0,
) -> Tuple[Any, float]:
    """Predict continuous features by sampling from quartile bins
    weighted by the MoG output, instead of directly from the MoG.
    
    For categorical features, falls back to vanilla prediction.
    
    Steps:
      1. Divide the feature's value_range into 4 equal quartile bins
      2. Use the MoG (pi, mu, sigma) to compute a probability mass
         for each quartile bin
      3. Sample a quartile proportional to its mass
      4. Sample uniformly within that quartile
    """
    device = next(model.parameters()).device

    if feature.dtype != "continuous":
        return predict_feature_vanilla(model, repr_vec, feature, temperature)

    # Get MoG parameters
    pi, mu, logvar = model.reg_head(repr_vec.unsqueeze(0))
    pi = pi.squeeze(0).cpu().numpy()       # [K]
    mu = mu.squeeze(0).cpu().numpy()       # [K]
    logvar = logvar.squeeze(0).cpu().numpy()  # [K]
    std = np.exp(0.5 * logvar)             # [K]

    # Value range
    if feature.value_range:
        vmin, vmax = feature.value_range
    else:
        vmin, vmax = 0.0, 1.0

    # Define 4 quartile bin edges in normalized [0,1] space
    bin_edges = [0.0, 0.25, 0.50, 0.75, 1.0]
    num_bins = 4

    # Compute probability mass in each bin from the MoG
    # P(bin) = sum_k pi_k * [Phi((edge_hi - mu_k)/std_k) - Phi((edge_lo - mu_k)/std_k)]
    from scipy.stats import norm as scipy_norm
    bin_probs = np.zeros(num_bins)
    for b in range(num_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        for k in range(len(pi)):
            s = max(std[k], 1e-6)
            cdf_hi = scipy_norm.cdf(hi, loc=mu[k], scale=s)
            cdf_lo = scipy_norm.cdf(lo, loc=mu[k], scale=s)
            bin_probs[b] += pi[k] * (cdf_hi - cdf_lo)

    # Apply temperature
    bin_probs = np.maximum(bin_probs, 1e-8)
    log_probs = np.log(bin_probs) / max(temperature, 1e-8)
    log_probs -= log_probs.max()
    bin_probs = np.exp(log_probs)
    bin_probs /= bin_probs.sum()

    # Sample a quartile bin
    chosen_bin = np.random.choice(num_bins, p=bin_probs)
    lo_norm = bin_edges[chosen_bin]
    hi_norm = bin_edges[chosen_bin + 1]

    # Sample uniformly within the chosen quartile (in normalized space)
    y_norm = np.random.uniform(lo_norm, hi_norm)

    # Denormalize to actual value range
    pred = y_norm * (vmax - vmin) + vmin
    conf = float(bin_probs[chosen_bin])

    return float(pred), conf


@torch.no_grad()
def autoregressive_pseudolabel_row_quartile(
    model: ASPIREEnhanced,
    features: List[Feature],
    original_values: List[Any],
    dataset_context: str,
    temperature: float = 1.0,
    cast_indices: Optional[Dict] = None,
    cast_k: int = 10,
    cast_alpha: float = 1.0,
    cast_metric: str = "cosine",
) -> Dict[str, Any]:
    """AR pseudolabelling using quartile sampling for continuous features,
    vanilla for categorical."""
    predicted_values = [None] * len(features)
    confidences = [0.0] * len(features)

    order = list(range(len(features)))
    random.shuffle(order)

    for step, f_idx in enumerate(order):
        feat = features[f_idx]

        obs_feats = [features[prev_idx] for prev_idx in order[:step]]
        obs_vals = [predicted_values[prev_idx] for prev_idx in order[:step]]

        try:
            repr_vec = compute_target_repr(model, feat, obs_feats, obs_vals, dataset_context)
        except Exception:
            repr_vec = None

        if repr_vec is None:
            predicted_values[f_idx] = original_values[f_idx]
            confidences[f_idx] = 0.0
            continue

        # Use quartile sampling for continuous, vanilla/cast for categorical
        if feat.dtype == "continuous":
            pred_val, conf = predict_feature_quartile(model, repr_vec, feat, temperature)
        elif cast_indices and f_idx in cast_indices:
            pred_val, conf = predict_feature_cast(
                model, repr_vec, feat, cast_indices.get(f_idx),
                temperature, cast_k, cast_alpha, cast_metric,
            )
        else:
            pred_val, conf = predict_feature_vanilla(model, repr_vec, feat, temperature)

        predicted_values[f_idx] = pred_val
        confidences[f_idx] = conf

    return {"predictions": predicted_values, "confidences": confidences}


# ─────────────────────────────────────────────────────────────────────
# Class-balanced categorical prediction
# ─────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_feature_balanced(
    model: ASPIREEnhanced,
    repr_vec: torch.Tensor,
    feature: Feature,
    class_freqs: Dict[str, float],
    temperature: float = 1.0,
) -> Tuple[Any, float]:
    """Predict categorical features with inverse-class-frequency reweighting.
    For continuous features, falls back to quartile sampling.
    
    Steps:
      1. Compute softmax(logits) as usual
      2. Multiply each class probability by 1/freq(class)
      3. Renormalize and sample
    This boosts rare classes and dampens the majority class.
    """
    device = next(model.parameters()).device

    if feature.dtype == "continuous":
        return predict_feature_quartile(model, repr_vec, feature, temperature)

    categories = list(feature.choices) if feature.choices else []
    if "[UNK]" not in categories:
        categories = categories + ["[UNK]"]

    label_vecs = []
    for cat in categories:
        emb = model.shared_text.encode_text(str(cat), is_context=False, device=device)
        label_vecs.append(emb)
    label_vecs = torch.stack(label_vecs, dim=0)
    label_embs = model.cls_head.category_proj(label_vecs)

    logits = model.cls_head.logits(repr_vec.unsqueeze(0), label_embs).squeeze(0)  # [num_classes]

    # Apply inverse class frequency as a logit bias (additive in log-space)
    # log(1/freq) is added to each class logit before softmax
    log_weights = torch.zeros_like(logits)
    for i, cat in enumerate(categories):
        if cat == "[UNK]":
            log_weights[i] = -1e9  # suppress UNK
        else:
            freq = class_freqs.get(cat, 1.0 / len(categories))
            log_weights[i] = math.log(1.0 / max(freq, 1e-6))

    balanced_logits = logits + log_weights
    balanced_probs = F.softmax(balanced_logits / max(1e-8, temperature), dim=-1)

    sampled_idx = torch.multinomial(balanced_probs, 1).item()
    pred_cat = categories[sampled_idx]
    conf = float(balanced_probs[sampled_idx].item())
    return pred_cat, conf


@torch.no_grad()
def autoregressive_pseudolabel_row_balanced(
    model: ASPIREEnhanced,
    features: List[Feature],
    original_values: List[Any],
    dataset_context: str,
    class_freq_map: Dict[int, Dict[str, float]],
    temperature: float = 1.0,
) -> Dict[str, Any]:
    """AR pseudolabelling with class-balanced categorical + quartile continuous."""
    predicted_values = [None] * len(features)
    confidences = [0.0] * len(features)

    order = list(range(len(features)))
    random.shuffle(order)

    for step, f_idx in enumerate(order):
        feat = features[f_idx]

        obs_feats = [features[prev_idx] for prev_idx in order[:step]]
        obs_vals = [predicted_values[prev_idx] for prev_idx in order[:step]]

        try:
            repr_vec = compute_target_repr(model, feat, obs_feats, obs_vals, dataset_context)
        except Exception:
            repr_vec = None

        if repr_vec is None:
            predicted_values[f_idx] = original_values[f_idx]
            confidences[f_idx] = 0.0
            continue

        if feat.dtype == "categorical" and f_idx in class_freq_map:
            pred_val, conf = predict_feature_balanced(
                model, repr_vec, feat, class_freq_map[f_idx], temperature
            )
        elif feat.dtype == "continuous":
            pred_val, conf = predict_feature_quartile(model, repr_vec, feat, temperature)
        else:
            pred_val, conf = predict_feature_vanilla(model, repr_vec, feat, temperature)

        predicted_values[f_idx] = pred_val
        confidences[f_idx] = conf

    return {"predictions": predicted_values, "confidences": confidences}


def compute_class_frequencies(df: pd.DataFrame, features: List[Feature]) -> Dict[int, Dict[str, float]]:
    """Compute per-feature class frequency distributions from the dataset."""
    freq_map = {}
    for i, feat in enumerate(features):
        if feat.dtype == "categorical":
            counts = df[feat.name].value_counts(normalize=True)
            freq_map[i] = {str(k): float(v) for k, v in counts.items()}
    return freq_map


def load_model(checkpoint_path, device_str="cuda:0"):
    """Load ASPIRE model from checkpoint."""
    logger.info(f"Loading ASPIRE from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')

    model = ASPIREEnhanced(
        model_dim=checkpoint.get('model_dim', 768),
        num_heads=checkpoint.get('num_heads', 8),
        num_inds=checkpoint.get('num_inds', 32),
        mask_prob=checkpoint.get('mask_prob', 0.40),
        max_targets=checkpoint.get('max_targets', 3),
        intra_layers=checkpoint.get('intra_layers', 2),
        inter_layers=checkpoint.get('inter_layers', 2),
        use_intra_set2set=checkpoint.get('use_intra_set2set', True),
        use_dataset_description=checkpoint.get('use_dataset_description', True),
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    logger.info(f"Model loaded on {device}")
    return model


def _extract_row_values(row, features):
    """Extract validated values from a row."""
    vals = []
    for feat in features:
        val = row.get(feat.name)
        if pd.isna(val):
            return None
        if feat.dtype == 'continuous':
            try:
                v = float(val)
                if not math.isfinite(v):
                    return None
                vals.append(v)
            except Exception:
                return None
        else:
            sval = str(val)
            if feat.choices and sval not in feat.choices:
                return None
            vals.append(sval)
    return vals


def run_pseudolabel(model, df, features, dataset_context, method, cast_indices=None, temperature=1.0):
    """Run pseudolabelling on all rows for a given method."""
    results = []

    for row_idx, row in df.iterrows():
        vals = _extract_row_values(row, features)
        if vals is None:
            continue

        result = autoregressive_pseudolabel_row(
            model, features, vals, dataset_context,
            method=method, cast_indices=cast_indices,
            temperature=temperature,
        )

        row_data = {"row_id": row_idx}
        for i, feat in enumerate(features):
            row_data[f"{feat.name}_original"] = vals[i]
            row_data[f"{feat.name}_pseudo"] = result["predictions"][i]
            row_data[f"{feat.name}_confidence"] = result["confidences"][i]
        results.append(row_data)

    return pd.DataFrame(results)


def run_pseudolabel_quartile(model, df, features, dataset_context, cast_indices=None, temperature=1.0):
    """Run pseudolabelling with quartile sampling for continuous features."""
    results = []

    for row_idx, row in df.iterrows():
        vals = _extract_row_values(row, features)
        if vals is None:
            continue

        result = autoregressive_pseudolabel_row_quartile(
            model, features, vals, dataset_context,
            temperature=temperature,
            cast_indices=cast_indices,
        )

        row_data = {"row_id": row_idx}
        for i, feat in enumerate(features):
            row_data[f"{feat.name}_original"] = vals[i]
            row_data[f"{feat.name}_pseudo"] = result["predictions"][i]
            row_data[f"{feat.name}_confidence"] = result["confidences"][i]
        results.append(row_data)

    return pd.DataFrame(results)


def run_pseudolabel_balanced(model, df, features, dataset_context, class_freq_map, temperature=1.0):
    """Run pseudolabelling with class-balanced categorical + quartile continuous."""
    results = []

    for row_idx, row in df.iterrows():
        vals = _extract_row_values(row, features)
        if vals is None:
            continue

        result = autoregressive_pseudolabel_row_balanced(
            model, features, vals, dataset_context,
            class_freq_map=class_freq_map,
            temperature=temperature,
        )

        row_data = {"row_id": row_idx}
        for i, feat in enumerate(features):
            row_data[f"{feat.name}_original"] = vals[i]
            row_data[f"{feat.name}_pseudo"] = result["predictions"][i]
            row_data[f"{feat.name}_confidence"] = result["confidences"][i]
        results.append(row_data)

    return pd.DataFrame(results)


def compare_results(df_original, features, vanilla_df, cast_df, quartile_df, balanced_df, num_display=10):
    """Print a side-by-side comparison of all 4 methods."""

    print("\n" + "=" * 130)
    print("COMPARISON: Original vs Vanilla vs CAST vs Quartile vs Balanced")
    print("=" * 130)

    for feat in features:
        print(f"\n{'─' * 110}")
        print(f"Feature: {feat.name}  (type: {feat.dtype})")
        print(f"{'─' * 110}")

        orig_col = f"{feat.name}_original"
        pseudo_col = f"{feat.name}_pseudo"
        conf_col = f"{feat.name}_confidence"

        rows_to_show = min(num_display, len(vanilla_df))
        n = len(vanilla_df)

        if feat.dtype == "continuous":
            print(f"  {'Row':>4s}  {'Original':>10s}  {'Vanilla':>10s}  {'CAST':>10s}  {'Quartile':>10s}  {'Balanced':>10s}  {'B.Err':>8s}")
            print("  " + "-" * 80)
            for i in range(rows_to_show):
                orig = float(vanilla_df.iloc[i][orig_col])
                vp = float(vanilla_df.iloc[i][pseudo_col])
                cp = float(cast_df.iloc[i][pseudo_col])
                qp = float(quartile_df.iloc[i][pseudo_col])
                bp = float(balanced_df.iloc[i][pseudo_col])
                print(f"  {i:>4d}  {orig:>10.4f}  {vp:>10.4f}  {cp:>10.4f}  {qp:>10.4f}  {bp:>10.4f}  {abs(bp-orig):>8.4f}")

            v_mae = np.mean([abs(float(vanilla_df.iloc[i][pseudo_col]) - float(vanilla_df.iloc[i][orig_col])) for i in range(n)])
            c_mae = np.mean([abs(float(cast_df.iloc[i][pseudo_col]) - float(cast_df.iloc[i][orig_col])) for i in range(n)])
            q_mae = np.mean([abs(float(quartile_df.iloc[i][pseudo_col]) - float(quartile_df.iloc[i][orig_col])) for i in range(n)])
            b_mae = np.mean([abs(float(balanced_df.iloc[i][pseudo_col]) - float(balanced_df.iloc[i][orig_col])) for i in range(n)])
            print(f"\n  Summary ({n} rows) — MAE:")
            print(f"    Vanilla:  {v_mae:.4f}")
            print(f"    CAST:     {c_mae:.4f}")
            print(f"    Quartile: {q_mae:.4f}")
            print(f"    Balanced: {b_mae:.4f}  (uses quartile for continuous)")
        else:
            print(f"  {'Row':>4s}  {'Original':>10s}  {'Vanilla':>10s}  {'V.ok':>5s}  {'CAST':>10s}  {'C.ok':>5s}  {'Balanced':>10s}  {'B.ok':>5s}")
            print("  " + "-" * 80)
            for i in range(rows_to_show):
                orig = str(vanilla_df.iloc[i][orig_col])
                vp = str(vanilla_df.iloc[i][pseudo_col])
                cp = str(cast_df.iloc[i][pseudo_col])
                bp = str(balanced_df.iloc[i][pseudo_col])
                print(f"  {i:>4d}  {orig:>10s}  {vp:>10s}  {'Y' if vp==orig else 'N':>5s}  {cp:>10s}  {'Y' if cp==orig else 'N':>5s}  {bp:>10s}  {'Y' if bp==orig else 'N':>5s}")

            v_acc = sum(1 for i in range(n) if str(vanilla_df.iloc[i][pseudo_col]) == str(vanilla_df.iloc[i][orig_col])) / n
            c_acc = sum(1 for i in range(n) if str(cast_df.iloc[i][pseudo_col]) == str(cast_df.iloc[i][orig_col])) / n
            b_acc = sum(1 for i in range(n) if str(balanced_df.iloc[i][pseudo_col]) == str(balanced_df.iloc[i][orig_col])) / n

            # Show class distribution for each method
            print(f"\n  Summary ({n} rows):")
            print(f"    Vanilla  — Accuracy: {v_acc:.2%}")
            print(f"    CAST     — Accuracy: {c_acc:.2%}")
            print(f"    Balanced — Accuracy: {b_acc:.2%}")

            # Class distribution comparison
            orig_dist = {}
            v_dist = {}
            c_dist = {}
            b_dist = {}
            for i in range(n):
                o = str(vanilla_df.iloc[i][orig_col])
                v = str(vanilla_df.iloc[i][pseudo_col])
                c = str(cast_df.iloc[i][pseudo_col])
                b = str(balanced_df.iloc[i][pseudo_col])
                orig_dist[o] = orig_dist.get(o, 0) + 1
                v_dist[v] = v_dist.get(v, 0) + 1
                c_dist[c] = c_dist.get(c, 0) + 1
                b_dist[b] = b_dist.get(b, 0) + 1

            all_classes = sorted(set(list(orig_dist.keys()) + list(v_dist.keys()) + list(c_dist.keys()) + list(b_dist.keys())))
            print(f"\n  Class Distribution:")
            print(f"    {'Class':>8s}  {'Original':>8s}  {'Vanilla':>8s}  {'CAST':>8s}  {'Balanced':>8s}")
            for cls in all_classes:
                print(f"    {cls:>8s}  {orig_dist.get(cls,0):>8d}  {v_dist.get(cls,0):>8d}  {c_dist.get(cls,0):>8d}  {b_dist.get(cls,0):>8d}")

    # Overall summary
    print(f"\n{'=' * 130}")
    print("OVERALL SUMMARY")
    print(f"{'=' * 130}")

    cat_feats = [f for f in features if f.dtype == "categorical"]
    cont_feats = [f for f in features if f.dtype == "continuous"]
    n = len(vanilla_df)

    if cat_feats:
        print("\nCategorical Features (Accuracy):")
        for feat in cat_feats:
            orig_col = f"{feat.name}_original"
            pseudo_col = f"{feat.name}_pseudo"
            v_acc = sum(1 for i in range(n) if str(vanilla_df.iloc[i][pseudo_col]) == str(vanilla_df.iloc[i][orig_col])) / n
            c_acc = sum(1 for i in range(n) if str(cast_df.iloc[i][pseudo_col]) == str(cast_df.iloc[i][orig_col])) / n
            q_acc = sum(1 for i in range(n) if str(quartile_df.iloc[i][pseudo_col]) == str(quartile_df.iloc[i][orig_col])) / n
            b_acc = sum(1 for i in range(n) if str(balanced_df.iloc[i][pseudo_col]) == str(balanced_df.iloc[i][orig_col])) / n
            print(f"  {feat.name:25s}  Vanilla: {v_acc:.2%}  |  CAST: {c_acc:.2%}  |  Quartile: {q_acc:.2%}  |  Balanced: {b_acc:.2%}")

    if cont_feats:
        print("\nContinuous Features (MAE):")
        for feat in cont_feats:
            orig_col = f"{feat.name}_original"
            pseudo_col = f"{feat.name}_pseudo"
            v_mae = np.mean([abs(float(vanilla_df.iloc[i][pseudo_col]) - float(vanilla_df.iloc[i][orig_col])) for i in range(n)])
            c_mae = np.mean([abs(float(cast_df.iloc[i][pseudo_col]) - float(cast_df.iloc[i][orig_col])) for i in range(n)])
            q_mae = np.mean([abs(float(quartile_df.iloc[i][pseudo_col]) - float(quartile_df.iloc[i][orig_col])) for i in range(n)])
            b_mae = np.mean([abs(float(balanced_df.iloc[i][pseudo_col]) - float(balanced_df.iloc[i][orig_col])) for i in range(n)])
            best = min(v_mae, c_mae, q_mae, b_mae)
            def tag(m): return " <--" if m == best else ""
            print(f"  {feat.name:25s}  Vanilla: {v_mae:.4f}{tag(v_mae)}  |  CAST: {c_mae:.4f}{tag(c_mae)}  |  Quartile: {q_mae:.4f}{tag(q_mae)}  |  Balanced: {b_mae:.4f}{tag(b_mae)}")

    print()


def main():
    # Configuration
    CHECKPOINT = "/playpen-nvme/scribble/shbhat/universal_machine/checkpoints/aspire_v2_icml_split1/best_model_calibrated.pt"
    DATASET_PATH = "/playpen-nvme/scribble/shbhat/universal_machine/data/synthetic-v0/QwQ-32B/QwQ-32B_fast_False_bn_llm_mmr_l0.5/synthetic_production_waste_reduction_2000.csv"
    NUM_ROWS = 20  # Small subset for quick comparison
    TEMPERATURE = 1.0
    DEVICE = "cuda:0"

    # Set seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # Load data
    logger.info(f"Loading dataset: {os.path.basename(DATASET_PATH)}")
    df = pd.read_csv(DATASET_PATH).head(NUM_ROWS)
    dataset_name = os.path.splitext(os.path.basename(DATASET_PATH))[0]
    dataset_context = f"Dataset: {dataset_name}"

    print(f"\nDataset: {dataset_name}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst 5 rows (original data):")
    print(df.head().to_string())

    # Infer features
    features = infer_features_from_csv(df, dataset_name)
    print(f"\nInferred {len(features)} features:")
    for f in features:
        print(f"  {f.name}: {f.dtype} {'choices=' + str(f.choices) if f.choices else ''} "
              f"{'range=' + str(f.value_range) if f.value_range else ''}")

    # Load model
    model = load_model(CHECKPOINT, DEVICE)

    # ── Run Vanilla ──
    logger.info("Running VANILLA pseudolabelling...")
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    vanilla_df = run_pseudolabel(model, df, features, dataset_context, method="vanilla", temperature=TEMPERATURE)
    logger.info(f"Vanilla: {len(vanilla_df)} rows processed")

    # ── Build CAST indices & run CAST ──
    logger.info("Building CAST indices...")
    df_full = pd.read_csv(DATASET_PATH).head(100)
    cast_indices = build_cast_indices(df_full, features, model, dataset_context, max_rows=100, max_per_class=30)

    logger.info("Running CAST pseudolabelling...")
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    cast_df = run_pseudolabel(model, df, features, dataset_context, method="cast",
                              cast_indices=cast_indices, temperature=TEMPERATURE)
    logger.info(f"CAST: {len(cast_df)} rows processed")

    # ── Run Quartile (continuous: quartile sampling, categorical: CAST) ──
    logger.info("Running QUARTILE pseudolabelling...")
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    quartile_df = run_pseudolabel_quartile(model, df, features, dataset_context,
                                           cast_indices=cast_indices, temperature=TEMPERATURE)
    logger.info(f"Quartile: {len(quartile_df)} rows processed")

    # ── Compute class frequencies from full dataset for balanced method ──
    logger.info("Computing class frequencies...")
    df_freq = pd.read_csv(DATASET_PATH)
    class_freq_map = compute_class_frequencies(df_freq, features)
    for f_idx, freqs in class_freq_map.items():
        logger.info(f"  {features[f_idx].name}: {freqs}")

    # ── Run Balanced (class-balanced categorical + quartile continuous) ──
    logger.info("Running BALANCED pseudolabelling...")
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    balanced_df = run_pseudolabel_balanced(model, df, features, dataset_context,
                                           class_freq_map=class_freq_map, temperature=TEMPERATURE)
    logger.info(f"Balanced: {len(balanced_df)} rows processed")

    # ── Compare all 4 ──
    compare_results(df, features, vanilla_df, cast_df, quartile_df, balanced_df, num_display=20)

    # Save results
    output_dir = "/playpen-nvme/scribble/shbhat/universal_machine/vanilla_vs_cast_comparison"
    os.makedirs(output_dir, exist_ok=True)
    vanilla_df.to_csv(os.path.join(output_dir, "vanilla_pseudolabels.csv"), index=False)
    cast_df.to_csv(os.path.join(output_dir, "cast_pseudolabels.csv"), index=False)
    quartile_df.to_csv(os.path.join(output_dir, "quartile_pseudolabels.csv"), index=False)
    balanced_df.to_csv(os.path.join(output_dir, "balanced_pseudolabels.csv"), index=False)
    df.to_csv(os.path.join(output_dir, "original_data.csv"), index=False)
    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
