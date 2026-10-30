"""Referência do experimento do contexto (30/10/2026): o classificador do ensaio (modelo final congelado, treinado nas
8 famílias reais de desenvolvimento) aplicado aos 144 casos das bases inventadas, que ele também nunca viu.
Atenção: o classificador vê a instância inteira (1.000 linhas); o LLM vê 200.
Uso: .venv/bin/python classificador.py <run sintético> <final_model> <saída>
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from missdetect.rb import features as feat
from missdetect.rb.config import load_config
from missdetect.rb.evaluate import metrics
from missdetect.rb.llm.cache import CLASSES
from missdetect.rb.llm.run import build_panel
from missdetect.rb.pipeline import load_final
from missdetect.rb.runs import read_observed

run, model_dir, out = (Path(a) for a in sys.argv[1:4])
config = load_config(Path("configs/rb/contexto-sintetico.yaml"))
panel = build_panel(run, config)
panel = panel[panel.replicate == 0].reset_index(drop=True)
model, meta = load_final(model_dir)
rows = []
for r in panel.itertuples():
    with threadpool_limits(1):
        d = feat.compute(read_observed(run, r.sample_id))
        p = model.predict_proba(np.array([[d[c] for c in meta["feature_columns"]]], dtype=float))[0]
    rows.append(
        {
            "sample_id": r.sample_id,
            "family_id": r.family_id,
            "y_true": CLASSES.index(r.mechanism),
            "y_pred": int(np.argmax(p)),
            **{f"p_{c}": p[i] for i, c in enumerate(CLASSES)},
        }
    )
pred = pd.DataFrame(rows)
m = metrics(pred.y_true.to_numpy(), pred[[f"p_{c}" for c in CLASSES]].to_numpy(), pred.y_pred.to_numpy())
pred.to_parquet(out / "predicoes_classificador.parquet", index=False)
text = (
    f"Classificador (modelo final do ensaio, {meta['model_sha256'][:12]}), nos 144 casos inventados: "
    f"macro-recall {m['macro_recall']:.3f}; recall MCAR {m['recall_MCAR']:.3f}, MAR {m['recall_MAR']:.3f}, "
    f"MNAR {m['recall_MNAR']:.3f}; log-loss {m['log_loss']:.3f}"
)
(out / "classificador.txt").write_text(text + "\n")
print(text)
print(pd.crosstab(pred.y_true.map(dict(enumerate(CLASSES))), pred.y_pred.map(dict(enumerate(CLASSES)))).to_string())
