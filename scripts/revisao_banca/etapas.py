"""Execução em duas etapas (emenda P25; decisão do autor de 03/11/2026), sorteada e congelada antes da tag `prereg-v1`.

A produção roda em duas metades. A segunda roda sempre, qualquer que seja o resultado da primeira; nenhuma regra muda
por causa dos números da primeira, que o autor pode ver. A análise confirmatória final é a do run completo
(`inject-v1.yaml`, inalterado), sobre todas as famílias. Os runs das etapas são parciais e descritivos.

Sorteio estratificado, com semente fixa:
- benchmark: das 50 famílias efetivas congeladas (`data_rb/freeze_v1_addendum.json`), 15 das 30 do painel de LLM e 10
  das 20 restantes vão para a etapa 1 (25 famílias); as outras 25, para a etapa 2;
- H7: 15 das 30 bases inventadas na etapa 1; as outras 15, na etapa 2.

As configurações das etapas são cópias das congeladas, com outra tabela-fonte e o mesmo `name` e `root_seed`. Assim,
os ids, as instâncias, as perguntas e o caderno de respostas (`_llm_cache/<name>`) são os mesmos do run completo, e
as perguntas da etapa 1 não são pagas de novo nele.

Uso: uv run --frozen python scripts/revisao_banca/etapas.py
"""

import hashlib
import json
from pathlib import Path

import pandas as pd

from missdetect.rb.ids import child_rng
from missdetect.rb.llm.panel import panel_families

ROOT_SEED = 20261101  # a do inject-v1.yaml
H7_ROOT_SEED = 20261103  # a do contexto-h7.yaml
FROZEN = Path("data_rb/freeze_v1_addendum.json")
SOURCES = Path("data_rb/sources_local.csv")
H7_SOURCES = Path("data_rb/contexto_h7/sources.csv")
OUT = Path("data_rb/etapas")
CONFIGS = {"inject-v1": Path("configs/rb/inject-v1.yaml"), "contexto-h7": Path("configs/rb/contexto-h7.yaml")}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sortear() -> dict:
    effective = sorted(json.loads(FROZEN.read_text(encoding="utf-8"))["effective_families"])
    panel = panel_families(effective, ROOT_SEED)
    rest = [f for f in effective if f not in panel]
    rng = child_rng(ROOT_SEED, "etapas")
    stage1 = sorted(int(f) for f in [*rng.choice(panel, 15, replace=False), *rng.choice(rest, 10, replace=False)])
    h7 = sorted(int(f) for f in pd.read_csv(H7_SOURCES).family_id)
    h7_stage1 = sorted(int(f) for f in child_rng(H7_ROOT_SEED, "etapas").choice(h7, 15, replace=False))
    return {
        "benchmark": {"etapa1": stage1, "etapa2": [f for f in effective if f not in stage1], "painel_completo": panel},
        "h7": {"etapa1": h7_stage1, "etapa2": [f for f in h7 if f not in h7_stage1]},
    }


def _config(source: Path, table: Path, stage: int) -> str:
    lines = source.read_text(encoding="utf-8").splitlines()
    header = [
        f"# Etapa {stage} de 2 (emenda P25, 03/11/2026; scripts/revisao_banca/etapas.py): cópia de {source.name} com só",
        f"# as famílias da etapa {stage}. Mesmo name e root_seed: mesmos ids, instâncias, perguntas e caderno de respostas",
        "# do run completo. Run parcial e descritivo; a análise confirmatória final é a do run completo.",
    ]
    body = [line for line in lines if not line.startswith("#")]
    table_lines = [i for i, line in enumerate(body) if line.strip().startswith("table:")]
    assert len(table_lines) == 1, source
    body[table_lines[0]] = f"  table: {table}"
    return "\n".join(header + body) + "\n"


def gravar() -> None:
    split = sortear()
    OUT.mkdir(parents=True, exist_ok=True)
    sources, h7 = pd.read_csv(SOURCES), pd.read_csv(H7_SOURCES)
    files = {}
    for stage in (1, 2):
        for key, table, name in (("benchmark", sources, "inject-v1"), ("h7", h7, "contexto-h7")):
            families = split[key][f"etapa{stage}"]
            path = OUT / f"{name}-etapa{stage}-sources.csv"
            table[table.family_id.isin(families)].to_csv(path, index=False)
            config = Path(f"configs/rb/{name}-etapa{stage}.yaml")
            config.write_text(_config(CONFIGS[name], path, stage), encoding="utf-8")
            files[str(path)] = _sha256(path)
            files[str(config)] = _sha256(config)
    record = {
        "emenda": "P25", "data": "2026-11-03", "decisao": "autor, 03/11/2026: duas metades sorteadas; a 2ª roda sempre",
        "sementes": {"benchmark": [ROOT_SEED, "etapas"], "h7": [H7_ROOT_SEED, "etapas"]},
        "fontes": {str(p): _sha256(p) for p in (FROZEN, SOURCES, H7_SOURCES, *CONFIGS.values())},
        **split, "arquivos": files,
    }  # fmt: skip
    (OUT / "etapas.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: record[k] for k in ("benchmark", "h7")}, ensure_ascii=False))


if __name__ == "__main__":
    gravar()
