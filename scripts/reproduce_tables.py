#!/usr/bin/env python3
"""
Rebuild the summary tables reported in the preprint from the raw per-run logs,
and check them against the saved tables in results/tables/.

No GPU, model weights or images are needed: everything is computed from
results/raw/experiment_results.json (65 training runs) and
results/raw/cross_dataset_results.json (195 external evaluations).

Usage:
    python scripts/reproduce_tables.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "raw"
TABLES = ROOT / "results" / "tables"


def config_name(row):
    name = f"{row['model_name']}/{row['method']}"
    if row["method"] == "lora":
        name += f" r={int(row['lora_rank'])}"
    if not row["apply_clahe"]:
        name += " (no CLAHE)"
    return name


def load_internal():
    runs = json.loads((RAW / "experiment_results.json").read_text())
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "history"} for r in runs])
    df["config"] = df.apply(config_name, axis=1)
    return df


def load_external():
    runs = json.loads((RAW / "cross_dataset_results.json").read_text())
    df = pd.DataFrame(runs)
    df["config"] = df.apply(config_name, axis=1)
    return df


def internal_summary(df):
    g = df.groupby("config")
    out = pd.DataFrame({
        "n_folds": g.size(),
        "trainable_params": g["trainable_params"].first(),
        "pct_trainable": (g["trainable_params"].first() / g["total_params"].first() * 100).round(3),
        "mean_auroc": g["best_val_auroc"].mean(),
        "std_auroc": g["best_val_auroc"].std(),
        "peak_vram_gb": g["peak_vram_gb"].mean(),
        "mean_time_min": g["training_time_sec"].mean() / 60,
    })
    return out.sort_values("mean_auroc", ascending=False)


def external_summary(df):
    g = df.groupby(["config", "test_dataset"])
    out = pd.DataFrame({
        "n_folds": g.size(),
        "mean_auroc": g["auroc"].mean(),
        "std_auroc": g["auroc"].std(),
        "mean_pr_auc": g["pr_auc"].mean(),
        "mean_sens_at_95spec": g["sens_at_95spec"].mean(),
    }).reset_index()
    return out


def check_against_saved(internal, external):
    """Compare recomputed means with the tables saved at the time of the runs."""
    ok = True

    saved = []
    for _, s in pd.read_csv(TABLES / "internal_cv_summary.csv").iterrows():
        saved.append((s["model_name"], s["method"], s["lora_rank_fill"], True, s["mean_auroc"]))
    for _, s in pd.read_csv(TABLES / "results_summary.csv").iterrows():
        saved.append((s["model_name"], s["method"], s["lora_rank"], bool(s["apply_clahe"]), s["mean_auroc"]))
    for model_name, method, rank, clahe, mean_auroc in saved:
        cfg = config_name({"model_name": model_name, "method": method,
                           "lora_rank": rank, "apply_clahe": clahe})
        s = {"mean_auroc": mean_auroc}
        if cfg not in internal.index:
            continue
        diff = abs(internal.loc[cfg, "mean_auroc"] - s["mean_auroc"])
        if diff > 1e-6:
            ok = False
            print(f"  MISMATCH internal {cfg}: {internal.loc[cfg, 'mean_auroc']:.6f} vs saved {s['mean_auroc']:.6f}")

    saved_ext = pd.read_csv(TABLES / "cross_dataset_aggregated.csv")
    ext = external.copy()
    ext["key"] = ext["config"].str.replace(" (no CLAHE)", "", regex=False)
    n_checked = 0
    for _, s in saved_ext.iterrows():
        match = ext[(ext["test_dataset"] == s["test_dataset"]) &
                    (np.isclose(ext["mean_auroc"], s["mean_auroc"], atol=1e-6))]
        n_checked += 1
        if match.empty:
            ok = False
            print(f"  MISMATCH external {s['config']} on {s['test_dataset']}: saved {s['mean_auroc']:.6f} not reproduced")
    return ok, n_checked


def main():
    internal_runs = load_internal()
    external_runs = load_external()
    print(f"Loaded {len(internal_runs)} training runs and {len(external_runs)} external evaluations.\n")

    internal = internal_summary(internal_runs)
    external = external_summary(external_runs)

    pd.set_option("display.width", 160)
    print("Internal 5-fold cross-validation on Shenzhen (mean AUROC over folds)")
    print(internal.round(4).to_string(), "\n")

    pivot = external.pivot(index="config", columns="test_dataset", values="mean_auroc")
    pivot["mean_external"] = pivot.mean(axis=1)
    print("External evaluation, mean AUROC over the 5 fold models")
    print(pivot.sort_values("mean_external", ascending=False).round(4).to_string(), "\n")

    sens = external.pivot(index="config", columns="test_dataset", values="mean_sens_at_95spec")
    print("External sensitivity at 95% specificity (mean over the 5 fold models)")
    print(sens.round(3).to_string())
    hits = [(cfg, ds) for cfg in sens.index for ds in sens.columns if sens.loc[cfg, ds] >= 0.85]
    print(f"\nConfiguration/dataset pairs reaching sensitivity >= 0.85 at 95% specificity: {len(hits)}")
    for cfg, ds in hits:
        print(f"  - {cfg} on {ds}: {sens.loc[cfg, ds]:.3f}")
    print()

    out_dir = ROOT / "results" / "reproduced"
    out_dir.mkdir(exist_ok=True)
    internal.to_csv(out_dir / "internal_summary.csv")
    external.to_csv(out_dir / "external_summary.csv", index=False)
    print(f"Recomputed tables written to {out_dir.relative_to(ROOT)}/")

    ok, n = check_against_saved(internal, external)
    print(f"\nCheck against saved tables: {'all values match' if ok else 'MISMATCHES FOUND'} "
          f"({n} external rows checked)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
