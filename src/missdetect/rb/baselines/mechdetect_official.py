"""MechDetect conforme o código oficial dos autores (Jung, Chandler, Jäger e Biessmann, arXiv:2512.04138).

Fonte: https://github.com/calgo-lab/MechDetect, commit 43c0f0031de65beb372b9f35ac150b9c2095c5f1 (licença MIT).

- Regra de decisão: `src/accuracy_analysis.ipynb`, função `decide_error_mech`. São dois testes de
  Mann-Whitney unilaterais sobre os AUCs da validação cruzada, cada um ao nível alpha/2 (Bonferroni):
  complete > shuffled (se não rejeita: MCAR) e complete > excluded (rejeita: MNAR; senão: MAR).
- AUCs: `src/auc_roc_scores/testing.py`, função `train_and_evaluate_models`, com
  `HistGradientBoostingClassifier(max_iter=200)` e `cross_val_score(cv=10, scoring=make_scorer(roc_auc_score))`.
  O scorer padrão chama `predict()`, de modo que o AUC é calculado sobre rótulos 0/1, não sobre probabilidades.
- Cubo de AUCs publicado: `src/auc_roc_scores/data/auc_roc_tensors_dict.npz`, com tensores no formato
  (taxa, mecanismo, coluna, clean/dirty, caso, iteração da CV).

A tarefa "complete" usa os valores da coluna-alvo também nas linhas em que ela foi removida. No regime
"clean", isso é informação de oráculo; no regime "dirty", a própria coluna com NaN revela a máscara.
Por isso este módulo serve à reprodução e ao painel oráculo, nunca à comparação observed-only (passo 39).

Uso:
    uv run python -m missdetect.rb.baselines.mechdetect_official --cubo <auc_roc_tensors_dict.npz> [--saida <csv>]
    uv run python -m missdetect.rb.baselines.mechdetect_official --cubo <npz> --recomputado <results/exp> --taxa 0.5
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OrdinalEncoder

MECHANISMS = ("MCAR", "MAR", "MNAR")  # ordem do eixo "mecanismo" do cubo oficial (ECAR, EAR, ENAR)
ERROR_RATES = (0.1, 0.25, 0.5, 0.75, 0.9)  # ordem do eixo "taxa" do cubo oficial
REGIMES = ("clean", "dirty")
CASE_COMPLETE, CASE_SHUFFLED, CASE_EXCLUDED, CASE_ONLY_COLUMN = 0, 1, 2, 3
DEFAULT_ALPHA = 0.05
DEFAULT_SEED = 1234  # semente usada por `run_all_experiments.sh`
DEFAULT_CV_FOLDS = 10


def mechdetect_pvalues(
    auc_complete: np.ndarray, auc_shuffled: np.ndarray, auc_excluded: np.ndarray
) -> tuple[float, float]:
    """p-valores dos dois testes de Mann-Whitney unilaterais (complete > shuffled; complete > excluded)."""
    p_shuffled = stats.mannwhitneyu(auc_complete, auc_shuffled, alternative="greater").pvalue
    p_excluded = stats.mannwhitneyu(auc_complete, auc_excluded, alternative="greater").pvalue
    return float(p_shuffled), float(p_excluded)


def mechdetect_decide(
    auc_complete: np.ndarray,
    auc_shuffled: np.ndarray,
    auc_excluded: np.ndarray,
    alpha: float = DEFAULT_ALPHA,
) -> str:
    """Decide o mecanismo com a regra do notebook oficial (dois testes, Bonferroni alpha/2)."""
    p_shuffled, p_excluded = mechdetect_pvalues(auc_complete, auc_shuffled, auc_excluded)
    if p_shuffled < alpha / 2:
        return "MNAR" if p_excluded < alpha / 2 else "MAR"
    return "MCAR"


def mechdetect_aucs(
    data: pd.DataFrame,
    mask: pd.Series,
    column: str,
    seed: int = DEFAULT_SEED,
    cv_folds: int = DEFAULT_CV_FOLDS,
    model: BaseEstimator | None = None,
    n_jobs: int = 1,
) -> np.ndarray:
    """AUCs das quatro tarefas, como em `train_and_evaluate_models`. Retorna um array (4, cv_folds).

    `data` é a tabela que o modelo vê: completa no regime "clean", com NaN no regime "dirty".
    `mask` é a máscara de ausência da coluna `column` (1 = ausente).
    """
    estimator = clone(model) if model is not None else HistGradientBoostingClassifier(max_iter=200)
    scorer = make_scorer(roc_auc_score)
    frame = data.copy()
    categorical = frame.select_dtypes(include=["object"]).columns
    if len(categorical):
        encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        frame[categorical] = encoder.fit_transform(frame[categorical].astype(str))

    def cv_auc(features: pd.DataFrame, target: pd.Series) -> np.ndarray:
        return cross_val_score(estimator, features, y=target, cv=cv_folds, scoring=scorer, n_jobs=n_jobs)

    complete = cv_auc(frame, mask)
    shuffled = cv_auc(frame, mask.sample(frac=1, random_state=seed))
    excluded = cv_auc(frame.drop(columns=[column]), mask)
    only_column = cv_auc(pd.DataFrame(frame[column]), mask)
    return np.array([complete, shuffled, excluded, only_column])


def load_cube(path: Path) -> dict[str, np.ndarray]:
    """Lê o cubo oficial. Os tensores são numéricos, então `allow_pickle` fica desligado."""
    with np.load(path, allow_pickle=False) as npz:
        return {key: npz[key] for key in npz.files}


def decide_cube(
    cube: dict[str, np.ndarray], alpha: float = DEFAULT_ALPHA, rates: tuple[float, ...] = ERROR_RATES
) -> pd.DataFrame:
    """Aplica a regra a todas as instâncias do cubo. Uma linha por (dataset, taxa, mecanismo, coluna, regime).

    `rates` rotula o primeiro eixo dos tensores; use `(taxa,)` com tensores `arr[None]` de uma reexecução."""
    rows = []
    for dataset_id in sorted(cube, key=int):
        tensor = cube[dataset_id]
        n_rates, n_mechs, n_cols, n_regimes = tensor.shape[:4]
        for i in range(n_rates):
            for j in range(n_mechs):
                for k in range(n_cols):
                    for r in range(n_regimes):
                        aucs = tensor[i, j, k, r]
                        decision = mechdetect_decide(
                            aucs[CASE_COMPLETE], aucs[CASE_SHUFFLED], aucs[CASE_EXCLUDED], alpha
                        )
                        rows.append(
                            {
                                "dataset_id": dataset_id,
                                "error_rate": rates[i],
                                "mechanism": MECHANISMS[j],
                                "column": k,
                                "regime": REGIMES[r],
                                "decision": decision,
                                "correct": decision == MECHANISMS[j],
                            }
                        )
    return pd.DataFrame(rows)


def accuracy_as_in_notebook(decisions: pd.DataFrame, regime: str, error_rates: tuple[float, ...]) -> float:
    """Acurácia agregada como no notebook: média por dataset sobre as colunas, depois média sobre datasets,
    mecanismos e taxas."""
    subset = decisions[(decisions["regime"] == regime) & decisions["error_rate"].isin(error_rates)]
    per_dataset = subset.groupby(["error_rate", "mechanism", "dataset_id"])["correct"].mean()
    return float(per_dataset.mean())


def summarize(decisions: pd.DataFrame) -> dict:
    """Números que o notebook imprime, mais as acurácias por regime, taxa e mecanismo."""
    summary = {
        "clean_rate_0.5": accuracy_as_in_notebook(decisions, "clean", (0.5,)),
        "dirty_all_rates": accuracy_as_in_notebook(decisions, "dirty", ERROR_RATES),
        "clean_all_rates": accuracy_as_in_notebook(decisions, "clean", ERROR_RATES),
        "n_datasets": int(decisions["dataset_id"].nunique()),
        "n_instances": len(decisions),
    }
    for regime in REGIMES:
        for rate in ERROR_RATES:
            subset = decisions[(decisions["regime"] == regime) & (decisions["error_rate"] == rate)]
            per_mech = subset.groupby(["mechanism", "dataset_id"])["correct"].mean().groupby("mechanism").mean()
            for mechanism in MECHANISMS:
                summary[f"{regime}_rate_{rate}_{mechanism}"] = float(per_mech[mechanism])
    return summary


def load_recomputed(results_dir: Path, rate: float) -> dict[str, np.ndarray]:
    """Lê as saídas de `testing.py` (`<results_dir>/<dataset_id>/<rate>/results.npy`), cada uma no formato
    (mecanismo, coluna, clean/dirty, caso, iteração da CV). Só entram execuções marcadas com FINISHED."""
    recomputed = {}
    for run_dir in sorted(results_dir.iterdir()):
        rate_dir = run_dir / str(rate)
        if (rate_dir / "FINISHED").exists():
            recomputed[run_dir.name] = np.load(rate_dir / "results.npy", allow_pickle=False)
    return recomputed


def compare_with_cube(
    recomputed: dict[str, np.ndarray], cube: dict[str, np.ndarray], rate: float, alpha: float = DEFAULT_ALPHA
) -> pd.DataFrame:
    """Compara AUCs e decisões recalculados com o cubo publicado, na mesma taxa. Uma linha por dataset e regime."""
    rate_index = ERROR_RATES.index(rate)
    rows = []
    for dataset_id, ours in recomputed.items():
        published = cube[dataset_id][rate_index]
        if ours.shape != published.shape:
            raise ValueError(f"{dataset_id}: formato {ours.shape} difere do cubo {published.shape}")
        for r, regime in enumerate(REGIMES):
            diff = np.abs(ours[:, :, r, :3] - published[:, :, r, :3])  # casos complete, shuffled, excluded
            same_decision = [
                mechdetect_decide(*ours[j, k, r, :3], alpha) == mechdetect_decide(*published[j, k, r, :3], alpha)
                for j in range(ours.shape[0])
                for k in range(ours.shape[1])
            ]
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "regime": regime,
                    "n_columns": ours.shape[1],
                    "max_abs_diff_auc": float(diff.max()),
                    "mean_abs_diff_auc": float(diff.mean()),
                    "max_abs_diff_case_mean": float(np.abs(diff.mean(axis=-1)).max()),
                    "decision_agreement": float(np.mean(same_decision)),
                }
            )
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cubo", type=Path, required=True, help="caminho do auc_roc_tensors_dict.npz oficial")
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--saida", type=Path, help="CSV com uma decisão por instância (opcional)")
    parser.add_argument(
        "--recomputado", type=Path, help="diretório results/<experimento> de uma reexecução de testing.py"
    )
    parser.add_argument("--taxa", type=float, default=0.5, help="taxa da reexecução comparada ao cubo")
    args = parser.parse_args()

    if args.recomputado:
        comparison = compare_with_cube(load_recomputed(args.recomputado, args.taxa), load_cube(args.cubo), args.taxa)
        print(comparison.to_string(index=False))
        return 0

    decisions = decide_cube(load_cube(args.cubo), args.alpha)
    if args.saida:
        if args.saida.exists():
            raise SystemExit(f"{args.saida} já existe; não sobrescrevo resultados.")
        decisions.to_csv(args.saida, index=False)
    print(json.dumps(summarize(decisions), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
