"""
Forensic Ablation — Steps 2 & 4 do docs/forensic_analysis_context_aware.md.

Responde duas perguntas sem consumir API do LLM:
  Step 2: "Sem a feature llm_ctx_domain_prior (que vaza o rótulo via metadata),
          quanto da acurácia permanece?"
  Step 4: "Leave-One-Dataset-Out — o modelo generaliza para um domínio novo?"

Reusa os CSVs já extraídos em
  Output/v2_improved/{experiment}/real/ml_com_llm/{llm_model}/

Cenários (3):
  - C_full       : 31 features (baseline de 78.2%)
  - C_no_prior   : 30 features (remove llm_ctx_domain_prior)
  - C_only_prior : 1 feature  (apenas llm_ctx_domain_prior)

CV (2):
  - GroupKFold(5)       : replica a avaliação original
  - LeaveOneGroupOut    : 23 folds, gold-standard de generalização

Modelos (7): mesma grade de train_model.py (regime n>=100).

Saídas em Output/v2_improved/{experiment}/real/forensic/:
  - forensic_summary.csv
  - lodo_per_dataset.csv
  - forensic_deltas.csv
  - forensic_heatmap.png
  - lodo_per_dataset.png
  - training_summary.json

Uso:
  uv run python forensic_ablation.py --experiment ctx_aware \\
      --llm-model gemini-3-flash-preview
"""

import argparse
import json
import os
import sys
import warnings
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.paths import get_output_dir

warnings.filterwarnings("ignore")


# ======================================================================
# CONFIG
# ======================================================================
SEED = 42
N_BOOTSTRAP = 1000
CLASS_NAMES = {0: "MCAR", 1: "MAR", 2: "MNAR"}
DOMAIN_PRIOR_COL = "llm_ctx_domain_prior"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Forensic Ablation (Steps 2 & 4)")
    p.add_argument("--experiment", default="ctx_aware", help="Nome do experimento com os CSVs já extraídos")
    p.add_argument(
        "--llm-model", default="gemini-3-flash-preview", help="Modelo LLM usado na extração (subdiretório ml_com_llm/)"
    )
    p.add_argument("--data", default="real", choices=["real", "sintetico"], help="Tipo de dado (default: real)")
    p.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP, help="Iterações de bootstrap para CI 95%%")
    return p.parse_args()


# ======================================================================
# MODELOS (replica regime n>=100 de train_model.py)
# ======================================================================
def get_models(has_llm_features: bool, n_features: int) -> dict:
    """Retorna os 7 modelos com scaler+PCA para modelos sensíveis.

    PCA só é aplicado quando há pelo menos 2 features (evita n_components
    inválido no cenário C_only_prior).
    """
    use_pca = has_llm_features and n_features >= 2
    pca_step = ("pca", PCA(n_components=0.95, random_state=SEED)) if use_pca else None

    def _pipe(clf, enable_pca: bool = True) -> Pipeline:
        steps = [("scaler", StandardScaler())]
        if enable_pca and pca_step is not None:
            steps.append(pca_step)
        steps.append(("clf", clf))
        return Pipeline(steps)

    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=400,
            random_state=SEED,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=300,
            random_state=SEED,
        ),
        "LogisticRegression": _pipe(
            LogisticRegression(max_iter=3000, random_state=SEED),
            enable_pca=False,
        ),
        "SVM_RBF": _pipe(
            SVC(kernel="rbf", C=3, random_state=SEED, probability=False),
        ),
        "KNN": _pipe(KNeighborsClassifier(n_neighbors=5)),
        "MLP": _pipe(
            MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=2000, random_state=SEED),
        ),
        "NaiveBayes": _pipe(GaussianNB(), enable_pca=False),
    }


def apply_smote(X_train: pd.DataFrame, y_train: pd.Series):
    """SMOTE defensivo; retorna X, y originais se SMOTE falhar."""
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        return X_train, y_train

    counts = pd.Series(y_train).value_counts()
    min_count = int(counts.min())
    if min_count < 2 or len(counts) < 2:
        return X_train, y_train

    k = min(3, min_count - 1)
    try:
        return SMOTE(random_state=SEED, k_neighbors=k).fit_resample(X_train, y_train)
    except Exception:
        return X_train, y_train


# ======================================================================
# CV COM COLETA DE PREDIÇÕES
# ======================================================================
def run_cv(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    cv_splitter,
    has_llm_features: bool,
) -> dict:
    """Executa cross-validation coletando predições por fold.

    Retorna dict com:
      - 'predictions': lista de dicts {modelo, fold, test_group, y_true, y_pred}
      - 'fold_groups': lista de tuplas (fold_idx, [test_groups])
      - 'skipped_folds': contador por modelo (folds com <2 classes no treino)
    """
    n_features = X.shape[1]
    predictions = []
    skipped = {name: 0 for name in get_models(has_llm_features, n_features)}
    fold_groups = []

    splits = list(cv_splitter.split(X, y, groups))

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        X_tr_raw = X.iloc[train_idx]
        y_tr_raw = y.iloc[train_idx]
        X_te = X.iloc[test_idx]
        y_te = y.iloc[test_idx]
        groups_te = groups.iloc[test_idx]
        test_group_list = sorted(groups_te.unique().tolist())
        fold_groups.append((fold_idx, test_group_list))

        # Requer pelo menos 2 classes no treino
        if y_tr_raw.nunique() < 2:
            for name in skipped:
                skipped[name] += 1
            continue

        X_tr, y_tr = apply_smote(X_tr_raw, y_tr_raw)

        models = get_models(has_llm_features, n_features)
        for name, model in models.items():
            try:
                model.fit(X_tr, y_tr)
                y_pred = model.predict(X_te)
            except Exception as e:
                print(f"  ?? fold {fold_idx} {name} falhou: {e}")
                skipped[name] += 1
                continue

            for i, global_idx in enumerate(X_te.index):
                predictions.append(
                    {
                        "modelo": name,
                        "fold": fold_idx,
                        "test_group": groups_te.iloc[i],
                        "sample_idx": int(global_idx),
                        "y_true": int(y_te.iloc[i]),
                        "y_pred": int(y_pred[i]),
                    }
                )

    return {
        "predictions": predictions,
        "fold_groups": fold_groups,
        "skipped_folds": skipped,
    }


# ======================================================================
# AGREGAÇÕES
# ======================================================================
def summarize_predictions(preds: list[dict], n_bootstrap: int) -> pd.DataFrame:
    """Agrega predições por modelo: accuracy + f1_macro + bootstrap CI 95%."""
    df = pd.DataFrame(preds)
    if df.empty:
        return pd.DataFrame()

    rng = np.random.RandomState(SEED)
    rows = []
    for name, g in df.groupby("modelo"):
        y_true = g["y_true"].to_numpy()
        y_pred = g["y_pred"].to_numpy()
        acc = accuracy_score(y_true, y_pred)
        f1m = f1_score(y_true, y_pred, average="macro", zero_division=0)

        # Bootstrap sobre predições concatenadas (nível sample)
        boot_accs = []
        boot_f1s = []
        n = len(y_true)
        for _ in range(n_bootstrap):
            idx = rng.choice(n, n, replace=True)
            boot_accs.append(accuracy_score(y_true[idx], y_pred[idx]))
            boot_f1s.append(f1_score(y_true[idx], y_pred[idx], average="macro", zero_division=0))

        rows.append(
            {
                "modelo": name,
                "n_preds": int(n),
                "accuracy": float(acc),
                "accuracy_ci_lo": float(np.percentile(boot_accs, 2.5)),
                "accuracy_ci_hi": float(np.percentile(boot_accs, 97.5)),
                "f1_macro": float(f1m),
                "f1_macro_ci_lo": float(np.percentile(boot_f1s, 2.5)),
                "f1_macro_ci_hi": float(np.percentile(boot_f1s, 97.5)),
            }
        )
    return pd.DataFrame(rows)


def per_dataset_accuracy(preds: list[dict]) -> pd.DataFrame:
    """Accuracy por dataset (grupo) × modelo — usado no LODO."""
    df = pd.DataFrame(preds)
    if df.empty:
        return pd.DataFrame()

    rows = []
    for (name, group), g in df.groupby(["modelo", "test_group"]):
        y_true = g["y_true"].to_numpy()
        y_pred = g["y_pred"].to_numpy()
        # Classe verdadeira do grupo é constante por dataset (todos bootstraps
        # do mesmo dataset têm o mesmo rótulo)
        true_class_id = int(y_true[0])
        rows.append(
            {
                "modelo": name,
                "test_group": group,
                "true_class": CLASS_NAMES[true_class_id],
                "n_samples": int(len(y_true)),
                "accuracy": float(accuracy_score(y_true, y_pred)),
            }
        )
    return pd.DataFrame(rows)


# ======================================================================
# PLOTS
# ======================================================================
def plot_heatmap(df_summary: pd.DataFrame, out_path: str, title: str) -> None:
    """Heatmap cenario × modelo, facetado por CV."""
    cvs = sorted(df_summary["cv"].unique())
    scenarios = ["C_full", "C_no_prior", "C_only_prior"]
    scenarios = [s for s in scenarios if s in df_summary["cenario"].unique()]
    models = sorted(df_summary["modelo"].unique())

    fig, axes = plt.subplots(1, len(cvs), figsize=(7 * len(cvs), 5), squeeze=False)
    for ax, cv in zip(axes[0], cvs, strict=False):
        sub = df_summary[df_summary["cv"] == cv]
        mat = np.zeros((len(scenarios), len(models)))
        for i, s in enumerate(scenarios):
            for j, m in enumerate(models):
                row = sub[(sub["cenario"] == s) & (sub["modelo"] == m)]
                mat[i, j] = row["accuracy"].iloc[0] if not row.empty else np.nan

        im = ax.imshow(mat, cmap="RdYlGn", aspect="auto", vmin=0.2, vmax=0.95)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(scenarios)))
        ax.set_yticklabels(scenarios, fontsize=9)
        ax.set_title(f"{title} — {cv}")
        plt.colorbar(im, ax=ax, label="Accuracy")
        for i in range(len(scenarios)):
            for j in range(len(models)):
                v = mat[i, j]
                if not np.isnan(v):
                    ax.text(
                        j, i, f"{v:.1%}", ha="center", va="center", fontsize=7, color="white" if v < 0.55 else "black"
                    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_lodo_per_dataset(df_lodo: pd.DataFrame, out_path: str, best_model: str) -> None:
    """Acurácia LODO por dataset no melhor modelo, barras coloridas por cenário."""
    sub = df_lodo[df_lodo["modelo"] == best_model]
    scenarios = ["C_full", "C_no_prior", "C_only_prior"]
    scenarios = [s for s in scenarios if s in sub["cenario"].unique()]
    datasets = sorted(sub["test_group"].unique())

    x = np.arange(len(datasets))
    width = 0.8 / max(len(scenarios), 1)
    colors = {"C_full": "#2ecc71", "C_no_prior": "#e67e22", "C_only_prior": "#9b59b6"}

    fig, ax = plt.subplots(figsize=(max(12, len(datasets) * 0.55), 6))
    for i, s in enumerate(scenarios):
        rows = sub[sub["cenario"] == s].set_index("test_group").reindex(datasets)
        ax.bar(
            x + i * width,
            rows["accuracy"].fillna(0).to_numpy(),
            width,
            label=s,
            color=colors.get(s),
            edgecolor="black",
            linewidth=0.3,
        )

    ax.axhline(y=0.333, color="gray", linestyle="--", alpha=0.5, label="Acaso (33.3%)")
    ax.set_xticks(x + width * (len(scenarios) - 1) / 2)
    ax.set_xticklabels(datasets, rotation=75, ha="right", fontsize=7)
    ax.set_ylabel("Accuracy no fold LODO")
    ax.set_ylim([0, 1.05])
    ax.set_title(f"LODO por dataset ({best_model})")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


# ======================================================================
# MAIN
# ======================================================================
def main() -> None:
    args = parse_args()

    input_dir = get_output_dir(args.data, args.llm_model, args.experiment)
    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(input_dir)),  # .../real/
        "forensic",
    )
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("?? FORENSIC ABLATION — Steps 2 (no-prior) + 4 (LODO)")
    print("=" * 70)
    print(f"?? Input : {input_dir}")
    print(f"?? Output: {out_dir}")

    X_path = os.path.join(input_dir, "X_features.csv")
    y_path = os.path.join(input_dir, "y_labels.csv")
    g_path = os.path.join(input_dir, "groups.csv")

    for p in (X_path, y_path, g_path):
        if not os.path.exists(p):
            print(f"? Arquivo ausente: {p}")
            sys.exit(1)

    X_all = pd.read_csv(X_path)
    y = pd.read_csv(y_path).squeeze("columns")
    groups = pd.read_csv(g_path).squeeze("columns")

    if DOMAIN_PRIOR_COL not in X_all.columns:
        print(f"? Coluna {DOMAIN_PRIOR_COL!r} não encontrada em {X_path}")
        print(f"   Colunas disponíveis: {list(X_all.columns)}")
        sys.exit(1)

    print(f"?? Samples : {len(X_all)}")
    print(f"?? Features: {X_all.shape[1]}")
    print(f"?? Groups  : {groups.nunique()}")
    print(f"?? Classes : {dict(y.value_counts().sort_index())}")

    # ------------------------------------------------------------------
    # DEFINIR CENÁRIOS
    # ------------------------------------------------------------------
    scenarios = {
        "C_full": {
            "X": X_all.copy(),
            "desc": f"Todas as {X_all.shape[1]} features (baseline)",
        },
        "C_no_prior": {
            "X": X_all.drop(columns=[DOMAIN_PRIOR_COL]),
            "desc": f"Sem {DOMAIN_PRIOR_COL} ({X_all.shape[1] - 1} features)",
        },
        "C_only_prior": {
            "X": X_all[[DOMAIN_PRIOR_COL]].copy(),
            "desc": f"Apenas {DOMAIN_PRIOR_COL} (1 feature)",
        },
    }
    for name, cfg in scenarios.items():
        print(f"   {name}: {cfg['X'].shape[1]} feats — {cfg['desc']}")

    # ------------------------------------------------------------------
    # DEFINIR ESTRATÉGIAS DE CV
    # ------------------------------------------------------------------
    n_groups_total = groups.nunique()
    cv_strategies = {
        "GroupKFold-5": GroupKFold(n_splits=min(5, n_groups_total)),
        "LODO": LeaveOneGroupOut(),
    }
    print(f"\n?? CV: GroupKFold-{min(5, n_groups_total)} + LODO ({n_groups_total} folds)")

    # ------------------------------------------------------------------
    # RODAR TUDO
    # ------------------------------------------------------------------
    summary_rows = []
    lodo_rows = []
    skipped_log = {}
    total_iters = len(scenarios) * len(cv_strategies)

    with tqdm(total=total_iters, desc="Cenários×CV") as pbar:
        for cv_name, cv_splitter in cv_strategies.items():
            for s_name, s_cfg in scenarios.items():
                X_s = s_cfg["X"]
                has_llm = any(c.startswith("llm_") or c.startswith("emb_") for c in X_s.columns)
                result = run_cv(X_s, y, groups, cv_splitter, has_llm)

                # Agregados por modelo
                summary = summarize_predictions(
                    result["predictions"],
                    n_bootstrap=args.n_bootstrap,
                )
                for _, row in summary.iterrows():
                    r = row.to_dict()
                    r["cenario"] = s_name
                    r["cv"] = cv_name
                    r["n_features"] = X_s.shape[1]
                    summary_rows.append(r)

                # Per-dataset só para LODO (cada fold = 1 dataset)
                if cv_name == "LODO":
                    dp = per_dataset_accuracy(result["predictions"])
                    for _, row in dp.iterrows():
                        r = row.to_dict()
                        r["cenario"] = s_name
                        lodo_rows.append(r)

                skipped_log[f"{cv_name}|{s_name}"] = result["skipped_folds"]
                pbar.update(1)

    df_summary = pd.DataFrame(summary_rows)
    df_lodo = pd.DataFrame(lodo_rows)

    # ------------------------------------------------------------------
    # DELTAS ENTRE CENÁRIOS E ENTRE CVs
    # ------------------------------------------------------------------
    delta_rows = []
    pivot = df_summary.pivot_table(
        index=["cv", "modelo"],
        columns="cenario",
        values="accuracy",
    )

    for (cv, modelo), row in pivot.iterrows():
        d = {"cv": cv, "modelo": modelo}
        if "C_full" in row and "C_no_prior" in row:
            d["delta_full_minus_no_prior"] = float(row["C_full"] - row["C_no_prior"])
        if "C_full" in row and "C_only_prior" in row:
            d["delta_full_minus_only_prior"] = float(row["C_full"] - row["C_only_prior"])
        delta_rows.append(d)

    df_deltas_scn = pd.DataFrame(delta_rows)

    # Delta entre CVs (GroupKFold - LODO) para cada (modelo, cenario)
    cv_pivot = df_summary.pivot_table(
        index=["cenario", "modelo"],
        columns="cv",
        values="accuracy",
    )
    cv_delta_rows = []
    for (cenario, modelo), row in cv_pivot.iterrows():
        d = {"cenario": cenario, "modelo": modelo}
        if "GroupKFold-5" in row and "LODO" in row:
            d["delta_groupkfold_minus_lodo"] = float(row["GroupKFold-5"] - row["LODO"])
            d["acc_groupkfold"] = float(row["GroupKFold-5"])
            d["acc_lodo"] = float(row["LODO"])
        cv_delta_rows.append(d)
    df_deltas_cv = pd.DataFrame(cv_delta_rows)

    # ------------------------------------------------------------------
    # SALVAR
    # ------------------------------------------------------------------
    df_summary.sort_values(["cv", "cenario", "accuracy"], ascending=[True, True, False]).to_csv(
        os.path.join(out_dir, "forensic_summary.csv"), index=False
    )

    if not df_lodo.empty:
        df_lodo.sort_values(["cenario", "modelo", "test_group"]).to_csv(
            os.path.join(out_dir, "lodo_per_dataset.csv"), index=False
        )

    df_deltas_scn.to_csv(
        os.path.join(out_dir, "forensic_deltas_cenarios.csv"),
        index=False,
    )
    df_deltas_cv.to_csv(
        os.path.join(out_dir, "forensic_deltas_cv.csv"),
        index=False,
    )

    # ------------------------------------------------------------------
    # PLOTS
    # ------------------------------------------------------------------
    plot_heatmap(
        df_summary,
        out_path=os.path.join(out_dir, "forensic_heatmap.png"),
        title="Accuracy (cenário × modelo)",
    )

    if not df_lodo.empty:
        # Melhor modelo = maior accuracy média sobre LODO no cenário C_full
        c_full_lodo = df_summary[(df_summary["cenario"] == "C_full") & (df_summary["cv"] == "LODO")]
        if not c_full_lodo.empty:
            best_model = c_full_lodo.sort_values("accuracy", ascending=False).iloc[0]["modelo"]
        else:
            best_model = df_summary.sort_values("accuracy", ascending=False).iloc[0]["modelo"]

        plot_lodo_per_dataset(
            df_lodo,
            out_path=os.path.join(out_dir, "lodo_per_dataset.png"),
            best_model=best_model,
        )

    # ------------------------------------------------------------------
    # SUMMARY JSON
    # ------------------------------------------------------------------
    training_summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "experiment": args.experiment,
        "llm_model": args.llm_model,
        "data_type": args.data,
        "n_samples": int(len(X_all)),
        "n_features_full": int(X_all.shape[1]),
        "n_groups": int(n_groups_total),
        "scenarios": {s: {"n_features": int(cfg["X"].shape[1]), "desc": cfg["desc"]} for s, cfg in scenarios.items()},
        "cv_strategies": list(cv_strategies.keys()),
        "n_bootstrap": int(args.n_bootstrap),
        "seed": SEED,
        "skipped_folds": skipped_log,
        "input_dir": input_dir,
        "output_dir": out_dir,
    }
    with open(os.path.join(out_dir, "training_summary.json"), "w") as f:
        json.dump(training_summary, f, indent=2, ensure_ascii=False)

    # ------------------------------------------------------------------
    # PRINT FINAL
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("?? RESULTADOS — Accuracy média por cenário × CV (7 modelos)")
    print(f"{'=' * 70}")
    for cv_name in cv_strategies:
        print(f"\n? {cv_name}")
        for s_name in scenarios:
            sub = df_summary[(df_summary["cv"] == cv_name) & (df_summary["cenario"] == s_name)]
            if sub.empty:
                continue
            mean_acc = sub["accuracy"].mean()
            max_acc = sub["accuracy"].max()
            best = sub.loc[sub["accuracy"].idxmax(), "modelo"]
            print(
                f"  {s_name:15s} (n_feats={sub['n_features'].iloc[0]:3d}): "
                f"mean={mean_acc:.3f}  max={max_acc:.3f} ({best})"
            )

    print(f"\n{'=' * 70}")
    print("?? HIPÓTESES DO FORENSIC DOC — confronto com números observados")
    print(f"{'=' * 70}")

    def _get_max(cv: str, cen: str) -> float | None:
        sub = df_summary[(df_summary["cv"] == cv) & (df_summary["cenario"] == cen)]
        return float(sub["accuracy"].max()) if not sub.empty else None

    gk_full = _get_max("GroupKFold-5", "C_full")
    gk_nop = _get_max("GroupKFold-5", "C_no_prior")
    lodo_full = _get_max("LODO", "C_full")
    lodo_nop = _get_max("LODO", "C_no_prior")
    lodo_only = _get_max("LODO", "C_only_prior")

    if gk_full is not None and gk_nop is not None:
        drop = gk_full - gk_nop
        print(
            f"  H1 GroupKFold-5: C_full - C_no_prior = {drop:+.3f} "
            f"({gk_full:.3f} vs {gk_nop:.3f}) "
            f"{'? vazamento confirmado' if drop >= 0.20 else '?? vazamento moderado/ausente'}"
        )
    if lodo_full is not None:
        print(
            f"  H2 LODO C_full : {lodo_full:.3f} "
            f"{'? generaliza' if lodo_full >= 0.65 else '?? generalização fraca'}"
        )
    if lodo_nop is not None:
        print(
            f"  H3 LODO C_no_prior : {lodo_nop:.3f} "
            f"{'?? colapso sem metadata' if lodo_nop < 0.45 else '? segura sem prior'}"
        )
    if lodo_only is not None:
        print(
            f"  H4 LODO C_only_prior: {lodo_only:.3f} "
            f"{'?? LLM sozinho já resolve' if lodo_only >= 0.80 else '?? prior não basta'}"
        )

    print(f"\n?? Artefatos salvos em: {out_dir}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
