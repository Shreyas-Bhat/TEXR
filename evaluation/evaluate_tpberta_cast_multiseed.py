"""
Multi-seed evaluation wrapper for TP-BERTa (CAST).
Runs evaluation across 5 seeds and computes mean ± std statistics.
"""
import argparse
import json
import os
import sys
import subprocess
import numpy as np
from collections import defaultdict
from pathlib import Path

SEEDS = [42, 43, 44, 45, 46]

EVAL_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluate_tpberta_cast.py")


def run_single_seed(seed, args, setting, gpu_id):
    """Run evaluation for a single seed and return the results file path."""
    seed_output_dir = os.path.join(args.output_dir, f"seed_{seed}")
    os.makedirs(seed_output_dir, exist_ok=True)

    cmd = [
        sys.executable, EVAL_SCRIPT,
        "--pretrain_dir", args.pretrain_dir,
        "--model_suffix", args.model_suffix,
        "--settings", setting,
        "--num_shots", str(args.num_shots),
        "--fewshot_epochs", str(args.fewshot_epochs),
        "--finetune_epochs", str(args.finetune_epochs),
        "--lr", str(args.lr),
        "--batch_size", str(args.batch_size),
        "--device", "cuda:0",
        "--seed", str(seed),
        "--output_dir", seed_output_dir,
    ]

    if hasattr(args, 'backbone_pt') and args.backbone_pt:
        cmd.extend(["--backbone_pt", args.backbone_pt])

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    log_path = os.path.join(seed_output_dir, f"{setting}_seed{seed}.log")
    print(f"  [seed={seed}] Launching on GPU {gpu_id}, log: {log_path}")

    with open(log_path, "w") as log_f:
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=env)

    return proc, seed_output_dir, log_path


def aggregate_results(output_dir, setting, seeds):
    """Load per-seed results and compute mean ± std."""
    per_dataset = defaultdict(lambda: {"accuracy": [], "f1_score": [], "auc": []})

    for seed in seeds:
        seed_dir = os.path.join(output_dir, f"seed_{seed}")
        results_file = os.path.join(seed_dir, f"tpberta_cast_{setting}_results.json")

        if not os.path.exists(results_file):
            print(f"  WARNING: missing {results_file}")
            continue

        with open(results_file) as f:
            results = json.load(f)

        for ds_name, result in results.items():
            if "error" in result:
                continue
            for metric in ["accuracy", "f1_score", "auc"]:
                val = result.get(metric)
                if val is not None:
                    per_dataset[ds_name][metric].append(val)

    # Compute stats
    summary = {}
    all_accs_mean, all_f1s_mean = [], []

    for ds_name in sorted(per_dataset.keys()):
        ds_stats = {}
        for metric in ["accuracy", "f1_score", "auc"]:
            vals = per_dataset[ds_name][metric]
            if vals:
                ds_stats[f"{metric}_mean"] = float(np.mean(vals))
                ds_stats[f"{metric}_std"] = float(np.std(vals))
                ds_stats[f"{metric}_values"] = vals
            else:
                ds_stats[f"{metric}_mean"] = None
                ds_stats[f"{metric}_std"] = None

        summary[ds_name] = ds_stats

        if ds_stats["accuracy_mean"] is not None:
            all_accs_mean.append(ds_stats["accuracy_mean"])
        if ds_stats["f1_score_mean"] is not None:
            all_f1s_mean.append(ds_stats["f1_score_mean"])

    # Overall mean
    summary["__overall__"] = {
        "accuracy_mean": float(np.mean(all_accs_mean)) if all_accs_mean else None,
        "accuracy_std": float(np.std(all_accs_mean)) if all_accs_mean else None,
        "f1_score_mean": float(np.mean(all_f1s_mean)) if all_f1s_mean else None,
        "f1_score_std": float(np.std(all_f1s_mean)) if all_f1s_mean else None,
        "n_datasets": len(all_accs_mean),
        "n_seeds": len(seeds),
    }

    return summary


def print_summary_table(summary, setting):
    """Print a nice summary table."""
    print(f"\n{'='*90}")
    print(f"  {setting.upper()} — Mean ± Std over {summary['__overall__']['n_seeds']} seeds")
    print(f"{'='*90}")
    print(f"  {'Dataset':<22s} {'Accuracy':>18s}  {'F1 Score':>18s}  {'AUC':>18s}")
    print(f"  {'-'*22} {'-'*18}  {'-'*18}  {'-'*18}")

    for ds_name, stats in summary.items():
        if ds_name == "__overall__":
            continue

        acc_str = f"{stats['accuracy_mean']:.4f}±{stats['accuracy_std']:.4f}" if stats['accuracy_mean'] is not None else "N/A"
        f1_str = f"{stats['f1_score_mean']:.4f}±{stats['f1_score_std']:.4f}" if stats['f1_score_mean'] is not None else "N/A"
        auc_str = f"{stats['auc_mean']:.4f}±{stats['auc_std']:.4f}" if stats['auc_mean'] is not None else "N/A"
        print(f"  {ds_name:<22s} {acc_str:>18s}  {f1_str:>18s}  {auc_str:>18s}")

    ov = summary["__overall__"]
    print(f"  {'-'*22} {'-'*18}  {'-'*18}  {'-'*18}")
    acc_str = f"{ov['accuracy_mean']:.4f}±{ov['accuracy_std']:.4f}" if ov['accuracy_mean'] else "N/A"
    f1_str = f"{ov['f1_score_mean']:.4f}±{ov['f1_score_std']:.4f}" if ov['f1_score_mean'] else "N/A"
    print(f"  {'MEAN':<22s} {acc_str:>18s}  {f1_str:>18s}")
    print(f"{'='*90}\n")


def main():
    parser = argparse.ArgumentParser(description="Multi-seed TP-BERTa CAST evaluation")
    parser.add_argument('--pretrain_dir', type=str, required=True)
    parser.add_argument('--model_suffix', type=str, default='pytorch_models/best')
    parser.add_argument('--settings', type=str, nargs='+', default=['few_shot', 'finetuning'],
                        choices=['few_shot', 'finetuning'])
    parser.add_argument('--num_shots', type=int, default=5)
    parser.add_argument('--fewshot_epochs', type=int, default=50)
    parser.add_argument('--finetune_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--gpu', type=int, default=0, help='GPU to use')
    parser.add_argument('--output_dir', type=str, default='./tpberta_cast_multiseed_results')
    parser.add_argument('--seeds', type=int, nargs='+', default=SEEDS)
    parser.add_argument('--backbone_pt', type=str, default=None,
                        help='Path to raw .pt backbone weights file')
    parser.add_argument('--parallel', action='store_true',
                        help='Run seeds in parallel (requires multiple GPUs via --gpus)')
    parser.add_argument('--gpus', type=int, nargs='+', default=None,
                        help='GPU IDs for parallel execution (one seed per GPU)')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for setting in args.settings:
        print(f"\n{'#'*60}")
        print(f"# Running {setting} across seeds: {args.seeds}")
        print(f"{'#'*60}")

        if args.parallel and args.gpus:
            # Launch seeds in parallel across GPUs
            procs = []
            gpu_cycle = args.gpus
            for i, seed in enumerate(args.seeds):
                gpu_id = gpu_cycle[i % len(gpu_cycle)]
                proc, seed_dir, log_path = run_single_seed(seed, args, setting, gpu_id)
                procs.append((seed, proc, log_path))

            # Wait for all to finish
            for seed, proc, log_path in procs:
                print(f"  [seed={seed}] Waiting...")
                proc.wait()
                rc = proc.returncode
                status = "OK" if rc == 0 else f"FAILED (rc={rc})"
                print(f"  [seed={seed}] {status}")
        else:
            # Sequential execution on single GPU
            for seed in args.seeds:
                proc, seed_dir, log_path = run_single_seed(seed, args, setting, args.gpu)
                print(f"  [seed={seed}] Running (sequential)...")
                proc.wait()
                rc = proc.returncode
                status = "OK" if rc == 0 else f"FAILED (rc={rc})"
                print(f"  [seed={seed}] {status}")

        # Aggregate results
        print(f"\nAggregating {setting} results...")
        summary = aggregate_results(args.output_dir, setting, args.seeds)
        print_summary_table(summary, setting)

        # Save aggregated results
        agg_path = os.path.join(args.output_dir, f"tpberta_cast_{setting}_aggregated.json")
        # Remove raw values list for cleaner output
        clean_summary = {}
        for ds_name, stats in summary.items():
            clean_summary[ds_name] = {k: v for k, v in stats.items() if not k.endswith("_values")}
        with open(agg_path, 'w') as f:
            json.dump(clean_summary, f, indent=2)
        print(f"Saved aggregated results to {agg_path}")

    print("\nAll done!")


if __name__ == "__main__":
    main()
