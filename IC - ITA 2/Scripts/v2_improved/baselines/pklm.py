"""
PKLM — Teste não-paramétrico para MCAR baseado em Spohn et al. (2024).

Predicted KL divergence with Machine learning (PKLM):
  1. Treinar Random Forest para prever a mask de missing a partir de X observado
  2. Calcular KL divergence entre distribuições de probabilidade preditas
  3. Permutation test: repetir com mask embaralhada
  4. p-valor = proporção de KL_permuted >= KL_observed

PKLM é um teste binário (MCAR vs não-MCAR), não 3-way.

Uso como baseline:
    python -m baselines.pklm --data sintetico --experiment step05_pro
    python -m baselines.pklm --data real --experiment step05_pro

Uso como feature (chamado por extract_features.py):
    from baselines.pklm import pklm_test
    result = pklm_test(df, missing_col='X0', n_permutations=100)
"""
import argparse
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.paths import DATASET_PATHS, OUTPUT_BASE

warnings.filterwarnings("ignore")


# ==============================================================================
# PKLM TEST
# ==============================================================================
def pklm_test(df, missing_col="X0", n_permutations=100, n_estimators=100, random_state=42):
    """
    Teste PKLM (Spohn et al., 2024).

    Retorna:
        dict com:
        - pklm_statistic: KL divergence observada
        - pklm_pvalue: p-valor via permutação
        - rejects_mcar: bool (p < 0.05)
        - pklm_log_statistic: log(1 + statistic) para uso como feature
    """
    rng = np.random.RandomState(random_state)

    mask = df[missing_col].isna().astype(int).values
    n_missing = mask.sum()
    n_total = len(mask)

    # Casos degenerados
    if n_missing < 5 or (n_total - n_missing) < 5:
        return {
            "pklm_statistic": 0.0,
            "pklm_pvalue": 1.0,
            "rejects_mcar": False,
            "pklm_log_statistic": 0.0,
        }

    # Features: todas as colunas exceto a coluna com missing
    other_cols = [c for c in df.columns if c != missing_col]
    X = df[other_cols].values

    # Tratar NaN em X (imputação pela mediana)
    for col_idx in range(X.shape[1]):
        col_vals = X[:, col_idx]
        nan_mask = np.isnan(col_vals)
        if nan_mask.any():
            median_val = np.nanmedian(col_vals)
            col_vals[nan_mask] = median_val

    # KL divergence observada
    kl_observed = _compute_kl_divergence(X, mask, n_estimators, rng)

    # Permutation test
    kl_permuted = np.zeros(n_permutations)
    for i in range(n_permutations):
        mask_shuffled = mask.copy()
        rng.shuffle(mask_shuffled)
        kl_permuted[i] = _compute_kl_divergence(X, mask_shuffled, n_estimators, rng)

    # p-valor: proporção de KL_permuted >= KL_observed
    p_value = float(np.mean(kl_permuted >= kl_observed))

    return {
        "pklm_statistic": float(kl_observed),
        "pklm_pvalue": float(p_value),
        "rejects_mcar": p_value < 0.05,
        "pklm_log_statistic": float(np.log1p(kl_observed)),
    }


def _compute_kl_divergence(X, y, n_estimators, rng):
    """
    Treina RF para prever y a partir de X, depois calcula KL divergence
    entre distribuições de probabilidade preditas para cada classe.
    """
    min_class = min(np.sum(y == 0), np.sum(y == 1))

    if min_class < 2:
        return 0.0

    # Treinar RF com out-of-bag predictions
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=5,
        min_samples_leaf=10,
        oob_score=True,
        random_state=rng.randint(0, 2**31),
        n_jobs=-1,
    )
    clf.fit(X, y)

    # Usar OOB predictions para evitar overfitting
    if hasattr(clf, "oob_decision_function_"):
        proba = clf.oob_decision_function_
    else:
        # Fallback: CV predictions
        proba = _cv_predict_proba(X, y, n_estimators, rng)

    if proba is None or proba.shape[1] < 2:
        return 0.0

    # Probabilidades preditas para classe 1 (missing)
    p_miss = proba[y == 1, 1] if proba.shape[1] > 1 else proba[y == 1, 0]
    p_obs = proba[y == 0, 1] if proba.shape[1] > 1 else proba[y == 0, 0]

    if len(p_miss) == 0 or len(p_obs) == 0:
        return 0.0

    # KL divergence estimada via histogramas
    kl = _histogram_kl_divergence(p_miss, p_obs)

    return kl


def _cv_predict_proba(X, y, n_estimators, rng):
    """Fallback: predições via 3-fold CV."""
    min_class = min(np.sum(y == 0), np.sum(y == 1))
    n_folds = min(3, max(2, min_class))

    proba = np.zeros((len(y), 2))
    counts = np.zeros(len(y))

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=rng.randint(0, 2**31))

    for train_idx, test_idx in skf.split(X, y):
        if len(np.unique(y[test_idx])) < 2:
            continue
        clf = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=5,
            min_samples_leaf=10, random_state=rng.randint(0, 2**31), n_jobs=-1,
        )
        clf.fit(X[train_idx], y[train_idx])
        pred = clf.predict_proba(X[test_idx])
        proba[test_idx] += pred
        counts[test_idx] += 1

    valid = counts > 0
    if valid.sum() == 0:
        return None
    proba[valid] /= counts[valid, np.newaxis]
    return proba


def _histogram_kl_divergence(p, q, n_bins=20):
    """
    Calcula KL divergence simétrica (Jensen-Shannon) entre duas distribuições
    representadas por amostras, usando histogramas.
    """
    eps = 1e-10
    bins = np.linspace(0, 1, n_bins + 1)

    hist_p, _ = np.histogram(p, bins=bins, density=True)
    hist_q, _ = np.histogram(q, bins=bins, density=True)

    # Normalizar
    hist_p = hist_p / (hist_p.sum() + eps) + eps
    hist_q = hist_q / (hist_q.sum() + eps) + eps

    # KL simétrica (Jensen-Shannon divergence)
    m = 0.5 * (hist_p + hist_q)
    kl_pm = np.sum(hist_p * np.log(hist_p / m))
    kl_qm = np.sum(hist_q * np.log(hist_q / m))
    jsd = 0.5 * (kl_pm + kl_qm)

    return float(jsd)


# ==============================================================================
# BASELINE: Classificação por PKLM
# ==============================================================================
def pklm_classify_dataset(df, missing_col="X0", n_permutations=100):
    """
    Classifica dataset usando PKLM.
    PKLM é binário: MCAR vs não-MCAR.
    Para 3-way, combinamos com heurística simples baseada em correlação X1-mask.
    """
    result = pklm_test(df, missing_col=missing_col, n_permutations=n_permutations)

    if not result["rejects_mcar"]:
        prediction = "MCAR"
    else:
        # PKLM rejeita MCAR → é MAR ou MNAR
        # Heurística: usar correlação X1-mask para distinguir
        mask = df[missing_col].isna().astype(int).values
        X1 = df["X1"].values

        if np.std(X1) > 0 and np.std(mask) > 0:
            corr = abs(np.corrcoef(mask, X1)[0, 1])
        else:
            corr = 0.0

        # Se X1 correlaciona com mask → MAR, senão → MNAR
        if corr > 0.1:
            prediction = "MAR"
        else:
            prediction = "MNAR"

    return prediction, result


# ==============================================================================
# MAIN — Executar como baseline
# ==============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PKLM Baseline (Spohn et al., 2024)")
    parser.add_argument("--data", choices=["sintetico", "real"], required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--n-permutations", type=int, default=100)
    args = parser.parse_args()

    DATA_TYPE = args.data
    EXPERIMENT = args.experiment
    N_PERM = args.n_permutations
    OUT_DIR = os.path.join(OUTPUT_BASE, EXPERIMENT, DATA_TYPE, "baselines", "pklm")
    os.makedirs(OUT_DIR, exist_ok=True)

    print("=" * 70)
    print("🔬 PKLM BASELINE — Spohn et al. (2024)")
    print("=" * 70)
    print(f"📊 Dados: {DATA_TYPE}")
    print(f"🔄 Permutações: {N_PERM}")
    print(f"📂 Output: {OUT_DIR}")
    print("=" * 70)

    CLASS_MAP = {"MCAR": 0, "MAR": 1, "MNAR": 2}
    results = []

    for class_name in ["MCAR", "MAR", "MNAR"]:
        folder = DATASET_PATHS[DATA_TYPE][class_name]
        if not os.path.exists(folder):
            print(f"⚠️ Pasta não encontrada: {folder}")
            continue

        files = sorted([f for f in os.listdir(folder) if f.endswith(".txt")])
        print(f"\n📂 {class_name}: {len(files)} arquivos em {folder}")

        for fname in tqdm(files, desc=f"  {class_name}", leave=False):
            fpath = os.path.join(folder, fname)
            try:
                df = pd.read_csv(fpath, sep="\t")
                if "X0" not in df.columns:
                    df.columns = [f"X{i}" for i in range(df.shape[1])]

                prediction, pklm_result = pklm_classify_dataset(
                    df, missing_col="X0", n_permutations=N_PERM
                )

                results.append({
                    "file": fname,
                    "true_label": class_name,
                    "true_code": CLASS_MAP[class_name],
                    "pred_label": prediction,
                    "pred_code": CLASS_MAP[prediction],
                    "pklm_statistic": pklm_result["pklm_statistic"],
                    "pklm_pvalue": pklm_result["pklm_pvalue"],
                    "rejects_mcar": pklm_result["rejects_mcar"],
                    "pklm_log_statistic": pklm_result["pklm_log_statistic"],
                })
            except Exception as e:
                print(f"    ⚠️ Erro em {fname}: {e}")
                results.append({
                    "file": fname,
                    "true_label": class_name,
                    "true_code": CLASS_MAP[class_name],
                    "pred_label": "MCAR",
                    "pred_code": 0,
                    "pklm_statistic": 0.0,
                    "pklm_pvalue": 1.0,
                    "rejects_mcar": False,
                    "pklm_log_statistic": 0.0,
                })

    df_results = pd.DataFrame(results)

    # ==============================================================================
    # MÉTRICAS
    # ==============================================================================
    y_true = df_results["true_code"].values
    y_pred = df_results["pred_code"].values

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    report = classification_report(
        y_true, y_pred, target_names=["MCAR", "MAR", "MNAR"],
        output_dict=True, zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

    print(f"\n{'='*70}")
    print(f"📊 RESULTADOS PKLM BASELINE — {DATA_TYPE.upper()}")
    print(f"{'='*70}")
    print(f"   Accuracy: {acc:.4f}")
    print(f"   F1 Macro: {f1_macro:.4f}")
    print(f"\n   Per class:")
    for cls in ["MCAR", "MAR", "MNAR"]:
        r = report[cls]
        print(f"     {cls}: P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1-score']:.3f} (n={int(r['support'])})")

    print(f"\n   Confusion Matrix:")
    print(f"          Pred MCAR  MAR  MNAR")
    for i, cls in enumerate(["MCAR", "MAR", "MNAR"]):
        print(f"   {cls:5s}  {cm[i][0]:5d} {cm[i][1]:4d} {cm[i][2]:5d}")

    # ==============================================================================
    # TESTE MCAR BINÁRIO (poder e tipo I)
    # ==============================================================================
    print(f"\n{'='*70}")
    print(f"📊 PKLM COMO TESTE BINÁRIO (MCAR vs não-MCAR)")
    print(f"{'='*70}")

    df_results["is_mcar_true"] = df_results["true_label"] == "MCAR"
    df_results["is_mcar_pred"] = ~df_results["rejects_mcar"]

    # Tipo I: taxa de rejeição em MCAR verdadeiro (falso positivo)
    mcar_data = df_results[df_results["true_label"] == "MCAR"]
    type_i_rate = mcar_data["rejects_mcar"].mean()
    print(f"   Taxa Tipo I (falso positivo MCAR): {type_i_rate:.1%} (meta: <10%)")

    # Poder: taxa de rejeição em não-MCAR (MAR + MNAR)
    non_mcar_data = df_results[df_results["true_label"] != "MCAR"]
    power = non_mcar_data["rejects_mcar"].mean()
    print(f"   Poder (rejeição em MAR+MNAR): {power:.1%} (meta: >80%)")

    # Por classe
    for cls in ["MAR", "MNAR"]:
        sub = df_results[df_results["true_label"] == cls]
        rej_rate = sub["rejects_mcar"].mean()
        print(f"     {cls}: taxa de rejeição = {rej_rate:.1%}")

    # PKLM statistic stats
    print(f"\n   PKLM statistic (média por classe real):")
    for cls in ["MCAR", "MAR", "MNAR"]:
        sub = df_results[df_results["true_label"] == cls]
        print(
            f"     {cls}: stat={sub['pklm_statistic'].mean():.4f} "
            f"(std={sub['pklm_statistic'].std():.4f}) "
            f"p_mean={sub['pklm_pvalue'].mean():.3f}"
        )

    # ==============================================================================
    # SALVAR RESULTADOS
    # ==============================================================================
    df_results.to_csv(os.path.join(OUT_DIR, "predictions.csv"), index=False)

    metrics_rows = []
    for cls in ["MCAR", "MAR", "MNAR"]:
        r = report[cls]
        metrics_rows.append({
            "modelo": "PKLM",
            "classe": cls,
            "precision": r["precision"],
            "recall": r["recall"],
            "f1": r["f1-score"],
            "support": int(r["support"]),
        })
    pd.DataFrame(metrics_rows).to_csv(os.path.join(OUT_DIR, "metrics_per_class.csv"), index=False)

    print(f"\n   Distribuição predições: {dict(df_results['pred_label'].value_counts())}")
    print(f"   Distribuição real:      {dict(df_results['true_label'].value_counts())}")

    # Gráfico: Confusion Matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    labels = ["MCAR", "MAR", "MNAR"]
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(f"PKLM Baseline — {DATA_TYPE.upper()}\nAcc={acc:.1%}, F1_macro={f1_macro:.3f}")
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels)
    ax.set_yticks(range(3))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    plt.colorbar(im, ax=ax)
    thresh = cm.max() / 2.0
    for i in range(3):
        for j in range(3):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "confusion_matrix.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Gráfico: PKLM statistic distributions por classe
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # PKLM statistic
    for cls, color in zip(["MCAR", "MAR", "MNAR"], ["#3498db", "#2ecc71", "#e74c3c"]):
        vals = df_results[df_results["true_label"] == cls]["pklm_statistic"]
        axes[0].hist(vals, alpha=0.5, label=cls, color=color, bins=20)
    axes[0].set_title("PKLM Statistic (KL Divergence)")
    axes[0].legend()
    axes[0].set_xlabel("JSD")

    # PKLM p-value
    for cls, color in zip(["MCAR", "MAR", "MNAR"], ["#3498db", "#2ecc71", "#e74c3c"]):
        vals = df_results[df_results["true_label"] == cls]["pklm_pvalue"]
        axes[1].hist(vals, alpha=0.5, label=cls, color=color, bins=20)
    axes[1].axvline(x=0.05, color="red", linestyle="--", label="α=0.05")
    axes[1].set_title("PKLM p-value")
    axes[1].legend()
    axes[1].set_xlabel("p-value")

    plt.suptitle(f"PKLM Distributions — {DATA_TYPE.upper()}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "pklm_distributions.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Training summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "method": "PKLM",
        "reference": "Spohn et al. (2024)",
        "data_type": DATA_TYPE,
        "experiment": EXPERIMENT,
        "n_datasets": len(df_results),
        "n_permutations": N_PERM,
        "classifier": "RandomForestClassifier",
        "accuracy_3way": float(acc),
        "f1_macro_3way": float(f1_macro),
        "binary_test": {
            "type_i_rate": float(type_i_rate),
            "power": float(power),
            "power_mar": float(df_results[df_results["true_label"] == "MAR"]["rejects_mcar"].mean()),
            "power_mnar": float(df_results[df_results["true_label"] == "MNAR"]["rejects_mcar"].mean()),
        },
        "per_class": {
            cls: {
                "precision": float(report[cls]["precision"]),
                "recall": float(report[cls]["recall"]),
                "f1": float(report[cls]["f1-score"]),
                "support": int(report[cls]["support"]),
            }
            for cls in ["MCAR", "MAR", "MNAR"]
        },
        "confusion_matrix": cm.tolist(),
        "pklm_stats_by_class": {
            cls: {
                "mean_statistic": float(df_results[df_results["true_label"] == cls]["pklm_statistic"].mean()),
                "std_statistic": float(df_results[df_results["true_label"] == cls]["pklm_statistic"].std()),
                "mean_pvalue": float(df_results[df_results["true_label"] == cls]["pklm_pvalue"].mean()),
            }
            for cls in ["MCAR", "MAR", "MNAR"]
        },
    }
    with open(os.path.join(OUT_DIR, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Salvos em: {OUT_DIR}")
    print(f"{'='*70}")
