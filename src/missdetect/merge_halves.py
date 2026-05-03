"""Consolida step1_v2_neutral_part1 + part2 em step1_v2_neutral.

Após rodar a Metade 1 (hoje) e a Metade 2 (amanhã) em diretórios separados,
este script concatena os CSVs e re-imputa colunas LLM com mediana global.

Uso:
    uv run python merge_halves.py
"""

import os
import sys

import pandas as pd
from sklearn.impute import SimpleImputer

BASE = os.path.dirname(os.path.abspath(__file__))
OUTPUT_BASE = os.path.normpath(os.path.join(BASE, "..", "..", "Output", "v2_improved"))
MODEL_DIR = "real/ml_com_llm/gemini-3-pro-preview"

P1 = os.path.join(OUTPUT_BASE, "step1_v2_neutral_part1", MODEL_DIR)
P2 = os.path.join(OUTPUT_BASE, "step1_v2_neutral_part2", MODEL_DIR)
OUT = os.path.join(OUTPUT_BASE, "step1_v2_neutral", MODEL_DIR)


def main() -> None:
    for label, path in [("Part1", P1), ("Part2", P2)]:
        if not os.path.exists(os.path.join(path, "X_features.csv")):
            print(f"❌ {label} não encontrado: {path}")
            sys.exit(1)

    os.makedirs(OUT, exist_ok=True)

    X = pd.concat(
        [pd.read_csv(os.path.join(P1, "X_features.csv")), pd.read_csv(os.path.join(P2, "X_features.csv"))],
        ignore_index=True,
    )
    y = pd.concat(
        [pd.read_csv(os.path.join(P1, "y_labels.csv")), pd.read_csv(os.path.join(P2, "y_labels.csv"))],
        ignore_index=True,
    )
    g = pd.concat(
        [pd.read_csv(os.path.join(P1, "groups.csv")), pd.read_csv(os.path.join(P2, "groups.csv"))],
        ignore_index=True,
    )

    # Re-impute LLM/embedding cols com mediana global do conjunto consolidado
    llm_cols = [c for c in X.columns if c.startswith("llm_") or c.startswith("emb_")]
    if llm_cols:
        imputer = SimpleImputer(strategy="median")
        X[llm_cols] = imputer.fit_transform(X[llm_cols])

    X.to_csv(os.path.join(OUT, "X_features.csv"), index=False)
    y.to_csv(os.path.join(OUT, "y_labels.csv"), index=False)
    g.to_csv(os.path.join(OUT, "groups.csv"), index=False)

    print("✅ Merge concluído")
    print(f"   Linhas: {len(X)}")
    print(f"   Grupos únicos: {g['group'].nunique()}")
    print(f"   Distribuição de classes:\n{y['label'].value_counts().sort_index()}")
    print(f"   Output: {OUT}")


if __name__ == "__main__":
    main()
