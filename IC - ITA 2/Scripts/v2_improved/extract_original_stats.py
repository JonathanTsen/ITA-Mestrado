"""
extract_original_stats.py — Extract pre-normalization statistics from processed real datasets.

Reads processed datasets (already with X0-X4 and capped missing rates) and computes
X0 observed statistics BEFORE min-max normalization. These stats are needed to
provide domain context to the LLM context-aware extractor.

NOTE: Processed datasets have already gone through cap_missing_rate and impute_with_sample,
and min-max normalization has been applied. This script RECONSTRUCTS original values
from raw data when available, or computes stats from normalized versions as fallback.

Usage:
    cd "IC - ITA 2/Scripts/v2_improved"
    uv run python extract_original_stats.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
DATASET_DIR = os.path.join(os.path.dirname(SCRIPTS_DIR), "Dataset", "real_data")
PROCESSED_DIR = os.path.join(DATASET_DIR, "processado")
OUTPUT_PATH = os.path.join(PROCESSED_DIR, "stats_originais.json")


def compute_x0_stats(series: pd.Series) -> dict:
    """Compute comprehensive statistics for X0 observed values."""
    obs = series.dropna()
    if len(obs) < 2:
        return {}
    return {
        "X0_mean": round(float(obs.mean()), 6),
        "X0_std": round(float(obs.std()), 6),
        "X0_min": round(float(obs.min()), 6),
        "X0_max": round(float(obs.max()), 6),
        "X0_median": round(float(obs.median()), 6),
        "X0_skew": round(float(sp_stats.skew(obs)), 6),
        "X0_kurtosis": round(float(sp_stats.kurtosis(obs, fisher=True)), 6),
        "X0_p5": round(float(np.percentile(obs, 5)), 6),
        "X0_p25": round(float(np.percentile(obs, 25)), 6),
        "X0_p75": round(float(np.percentile(obs, 75)), 6),
        "X0_p95": round(float(np.percentile(obs, 95)), 6),
        "n_total": len(series),
        "n_missing": int(series.isna().sum()),
        "n_observed": int(obs.shape[0]),
        "missing_rate": round(float(series.isna().mean()), 4),
    }


def try_load_raw_stats(mechanism: str, name: str) -> dict | None:
    """Try to compute stats from raw (pre-normalization) data sources."""
    raw_dir = os.path.join(DATASET_DIR, mechanism)

    # Map processed names to raw files and column names
    RAW_MAP = {
        # MCAR
        "MCAR_oceanbuoys_humidity": ("oceanbuoys_tao.csv", "humidity"),
        "MCAR_oceanbuoys_airtemp": ("oceanbuoys_tao.csv", "air.temp"),
        # MAR
        "MAR_airquality_ozone": ("airquality.csv", "Ozone"),
        "MAR_mammographic_density": ("mammographic_mass_raw.csv", "Density"),
        # MNAR
        "MNAR_pima_insulin": ("pima_diabetes_raw.csv", "Insulin"),
        "MNAR_mroz_wages": ("mroz_wages.csv", "lwg"),
    }

    key = f"{mechanism}_{name}"
    if key not in RAW_MAP:
        return None

    raw_file, col_name = RAW_MAP[key]
    raw_path = os.path.join(raw_dir, raw_file)

    if not os.path.exists(raw_path):
        return None

    try:
        if "pima" in raw_file:
            df = pd.read_csv(
                raw_path, header=None,
                names=["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                       "Insulin", "BMI", "DiabetesPedigree", "Age", "Outcome"],
            )
            df["Insulin"] = df["Insulin"].replace(0, np.nan)
        elif "mroz" in raw_file:
            df = pd.read_csv(raw_path)
            df.loc[df["lfp"] == "no", "lwg"] = np.nan
        elif "mammographic" in raw_file:
            df = pd.read_csv(
                raw_path, header=None,
                names=["BIRADS", "Age", "Shape", "Margin", "Density", "Severity"],
                na_values="?",
            )
            for c in ["BIRADS", "Age", "Shape", "Margin", "Density"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df = pd.read_csv(raw_path)

        if col_name in df.columns:
            series = pd.to_numeric(df[col_name], errors="coerce")
            stats = compute_x0_stats(series)
            stats["source"] = "raw"
            return stats
    except Exception as e:
        print(f"  Warning: Could not load raw data for {key}: {e}")

    return None


def main():
    all_stats = {}

    for mechanism in ["MCAR", "MAR", "MNAR"]:
        mech_dir = os.path.join(PROCESSED_DIR, mechanism)
        if not os.path.exists(mech_dir):
            continue

        for fname in sorted(os.listdir(mech_dir)):
            if not fname.endswith(".txt"):
                continue

            # Extract dataset name (e.g., "pima_insulin" from "MNAR_pima_insulin.txt")
            dataset_key = fname.replace(".txt", "")
            name = dataset_key[len(mechanism) + 1:]  # Remove "MNAR_" prefix

            # Try raw data first
            raw_stats = try_load_raw_stats(mechanism, name)
            if raw_stats:
                print(f"  {dataset_key}: raw stats loaded")
                all_stats[dataset_key] = raw_stats
                continue

            # Fall back to processed (normalized) data
            fpath = os.path.join(mech_dir, fname)
            df = pd.read_csv(fpath, sep="\t")
            stats = compute_x0_stats(df["X0"])
            stats["source"] = "normalized"
            print(f"  {dataset_key}: normalized stats (no raw available)")
            all_stats[dataset_key] = stats

    # Save
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)

    print(f"\nSaved stats for {len(all_stats)} datasets to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
