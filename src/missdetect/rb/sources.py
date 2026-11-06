"""Tabelas-fonte completas e famílias de dados (passo 31 do plano).

Fonte dos dados: as 101 tabelas OpenML avaliadas pelo MechDetect. Elas podem vir da API do OpenML ou, de forma
equivalente, dos CSV distribuídos em `calgo-lab/MechDetect/src/auc_roc_scores/datasets.tar.gz`. A fonte usada fica
registrada em `source_backend`.

Filtros pré-registrados (aplicados após remover a coluna-alvo do benchmark e as colunas de identificador):
- nenhum valor ausente;
- pelo menos 500 linhas;
- pelo menos 3 colunas numéricas não constantes (categóricas ficam fora do escopo e são descartadas).

Famílias (versões ou derivações da mesma base original):
- candidatos: pares com sobreposição de nomes de colunas |A∩B|/min(|A|,|B|) >= 0,6 e >= 3 colunas em comum;
- fusão automática: contenção média de valores >= 0,9 nas colunas em comum (os valores da tabela menor estão
  contidos nos da maior);
- decisões manuais: `family_overrides.csv` (id_a, id_b, action ∈ {merge, split}, reason), versionado;
- agrupamento transitivo (union-find); `family_id` = menor ID OpenML do grupo.

Representante de cada família: maior nº de colunas numéricas não constantes; depois maior nº de linhas; empate → menor
ID. Famílias de desenvolvimento: sorteadas com semente fixa entre as famílias com representante incluído.

Uso:
    uv run python -m missdetect.rb.sources --csv-dir <mech-detect-datasets> --overrides data_rb/family_overrides.csv \
        --saida data_rb/sources_local.csv
"""

import argparse
import hashlib
import re
import tarfile
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from missdetect.rb.ids import child_rng

MECHDETECT_IDS = (
    1046, 1049, 1050, 1063, 1067, 1068, 1461, 1464, 1480, 1494, 151, 1510, 23, 23517, 251, 31, 310, 37, 40701,
    40975, 40994, 4135, 42493, 44, 44025, 44026, 44054, 44055, 44056, 44059, 44061, 44062, 44063, 44064, 44066,
    44089, 44090, 44091, 44120, 44122, 44123, 44124, 44125, 44126, 44127, 44130, 44131, 44132, 44133, 44134, 44136,
    44137, 44138, 44139, 44140, 44141, 44142, 44144, 44145, 44147, 44148, 44156, 44157, 44158, 44160, 44162, 44956,
    44957, 44958, 44959, 44960, 44962, 44963, 44964, 44966, 44967, 44969, 44970, 44971, 44972, 44974, 44976, 44977,
    44978, 44979, 44980, 44981, 44983, 44984, 44987, 44989, 44990, 44993, 44994, 45012, 4534, 45402, 54, 6, 725, 823,
)  # fmt: skip
TARGET_COLUMN_LOCAL = "target"  # nome da coluna-alvo nos CSV do MechDetect (removida, como em testing.py)
MIN_ROWS = 500
MIN_NUMERIC = 3
NAME_OVERLAP_CANDIDATE = 0.6
MIN_SHARED_COLUMNS = 3
VALUE_CONTAINMENT_MERGE = 0.9
DEFAULT_ROOT_SEED = 20261021
DEFAULT_N_DEV_FAMILIES = 8


@dataclass
class TableProfile:
    openml_id: int
    source_backend: str
    sha256_raw: str
    n_rows: int
    n_columns: int
    n_missing: int
    id_like_columns: str
    n_numeric_nonconstant: int
    numeric_columns: str
    column_set_hash: str
    content_fingerprint: str
    included: bool
    exclusion_reason: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def ensure_source_csvs(csv_dir: Path, tarball: Path, sources: pd.DataFrame) -> int:
    """Garante os CSV-fonte em `csv_dir` (retomada, plano 38c): se faltar algum, reextrai o arquivo dos autores do
    MechDetect (`tarball`, que contém a pasta `csv_dir.name`) no diretório pai, com o filtro seguro do `tarfile`.
    Depois confere o SHA-256 de cada CSV contra `sources.sha256_raw`. Devolve quantos arquivos foram extraídos."""
    expected = {f"{int(i)}.csv": h for i, h in zip(sources.openml_id, sources.sha256_raw, strict=True)}
    missing = [name for name in expected if not (csv_dir / name).exists()]
    extracted = 0
    if missing:
        csv_dir.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tarball, "r:gz") as tar:
            members = [m for m in tar.getmembers() if Path(m.name).parts[:1] == (csv_dir.name,)]
            tar.extractall(csv_dir.parent, members=members, filter="data")
        extracted = len(missing)
    bad = [name for name, h in expected.items() if not (csv_dir / name).exists() or sha256_file(csv_dir / name) != h]
    if bad:
        raise ValueError(f"CSV-fonte ausentes ou com SHA-256 diferente do registrado: {bad[:5]}")
    return extracted


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def read_local_csv(path: Path) -> pd.DataFrame:
    """Lê um CSV do MechDetect: descarta a coluna de índice sem nome e a coluna-alvo do benchmark."""
    frame = pd.read_csv(path)
    unnamed = [c for c in frame.columns if c == "" or str(c).startswith("Unnamed")]
    return frame.drop(columns=[*unnamed, TARGET_COLUMN_LOCAL], errors="ignore")


def id_like(frame: pd.DataFrame) -> list[str]:
    """Colunas de identificador: nome com "id" como palavra, ou valores inteiros únicos em todas as linhas."""
    columns = []
    for col in frame.columns:
        name_hit = re.search(r"(^|[^a-z])id([^a-z]|$)", str(col).lower()) is not None
        values = frame[col]
        unique_integer = values.is_unique and pd.api.types.is_integer_dtype(values)
        if name_hit or unique_integer:
            columns.append(str(col))
    return columns


def profile_table(openml_id: int, frame: pd.DataFrame, backend: str, sha256_raw: str) -> TableProfile:
    ids = id_like(frame)
    work = frame.drop(columns=ids)
    numeric = [c for c in work.columns if pd.api.types.is_numeric_dtype(work[c]) and work[c].nunique(dropna=True) > 1]
    names = sorted(normalize_name(c) for c in work.columns)
    column_set_hash = hashlib.sha256("|".join(names).encode()).hexdigest()[:16]
    means = [f"{work[c].mean():.4g}" for c in sorted(numeric)]
    fingerprint = hashlib.sha256(f"{names}|{len(work)}|{means}".encode()).hexdigest()[:16]
    n_missing = int(work.isna().sum().sum())
    reasons = []
    if n_missing:
        reasons.append("has_missing")
    if len(work) < MIN_ROWS:
        reasons.append("n_rows<500")
    if len(numeric) < MIN_NUMERIC:
        reasons.append("numeric<3")
    return TableProfile(
        openml_id=openml_id,
        source_backend=backend,
        sha256_raw=sha256_raw,
        n_rows=len(work),
        n_columns=work.shape[1],
        n_missing=n_missing,
        id_like_columns=";".join(ids),
        n_numeric_nonconstant=len(numeric),
        numeric_columns=";".join(map(str, numeric)),
        column_set_hash=column_set_hash,
        content_fingerprint=fingerprint,
        included=not reasons,
        exclusion_reason=";".join(reasons),
    )


def _value_set(series: pd.Series) -> set:
    values = series.dropna()
    if pd.api.types.is_numeric_dtype(values):
        return set(np.round(values.astype(float).to_numpy(), 6))
    return set(values.astype(str))


def candidate_pairs(frames: dict[int, pd.DataFrame]) -> pd.DataFrame:
    """Pares com nomes de colunas sobrepostos e a contenção de valores nas colunas em comum."""
    normalized = {i: f.set_axis([normalize_name(c) for c in f.columns], axis=1) for i, f in frames.items()}
    rows = []
    for a, b in combinations(sorted(normalized), 2):
        fa, fb = normalized[a], normalized[b]
        shared = [c for c in fa.columns if c in set(fb.columns)]
        if len(shared) < MIN_SHARED_COLUMNS:
            continue
        overlap = len(shared) / min(fa.shape[1], fb.shape[1])
        if overlap < NAME_OVERLAP_CANDIDATE:
            continue
        small, big = (fa, fb) if len(fa) <= len(fb) else (fb, fa)
        containment = []
        for col in shared:
            vs, vb = _value_set(small[col]), _value_set(big[col])
            containment.append(len(vs & vb) / max(1, len(vs)))
        rows.append(
            {
                "id_a": a,
                "id_b": b,
                "name_overlap": round(overlap, 3),
                "shared_columns": len(shared),
                "mean_value_containment": round(float(np.mean(containment)), 3),
                "min_value_containment": round(float(np.min(containment)), 3),
                "auto_merge": float(np.mean(containment)) >= VALUE_CONTAINMENT_MERGE,
            }
        )
    return pd.DataFrame(rows)


def assign_families(ids: list[int], pairs: pd.DataFrame, overrides: pd.DataFrame) -> dict[int, int]:
    """Union-find sobre as fusões automáticas e manuais; `split` em overrides anula uma fusão automática."""
    splits = {
        tuple(sorted((int(r.id_a), int(r.id_b)))) for r in overrides.itertuples(index=False) if r.action == "split"
    }
    merges = [
        (int(r.id_a), int(r.id_b))
        for r in pairs.itertuples(index=False)
        if r.auto_merge and tuple(sorted((int(r.id_a), int(r.id_b)))) not in splits
    ]
    merges += [(int(r.id_a), int(r.id_b)) for r in overrides.itertuples(index=False) if r.action == "merge"]
    parent = {i: i for i in ids}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in merges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return {i: find(i) for i in ids}


def choose_representatives(table: pd.DataFrame) -> pd.Series:
    """Marca o representante de cada família entre as tabelas incluídas."""
    eligible = table[table.included].sort_values(
        ["family_id", "n_numeric_nonconstant", "n_rows", "openml_id"], ascending=[True, False, False, True]
    )
    representatives = set(eligible.groupby("family_id").head(1).openml_id)
    return table.openml_id.isin(representatives)


def assign_roles(table: pd.DataFrame, n_dev: int, root_seed: int) -> pd.Series:
    """Sorteia as famílias de desenvolvimento entre as que têm representante; as demais são confirmatórias."""
    families = np.sort(table.loc[table.representative, "family_id"].unique())
    dev = set(child_rng(root_seed, "development_families").choice(families, size=n_dev, replace=False).tolist())
    role = np.where(table.family_id.isin(dev), "dev", "confirm")
    return pd.Series(np.where(table.representative, role, "not_used"), index=table.index)


def build_from_local(
    csv_dir: Path, overrides: pd.DataFrame, n_dev: int = DEFAULT_N_DEV_FAMILIES, root_seed: int = DEFAULT_ROOT_SEED
) -> tuple[pd.DataFrame, pd.DataFrame]:
    profiles, frames = [], {}
    for openml_id in MECHDETECT_IDS:
        path = csv_dir / f"{openml_id}.csv"
        frame = read_local_csv(path)
        profiles.append(asdict(profile_table(openml_id, frame, "mechdetect_csv", sha256_file(path))))
        frames[openml_id] = frame
    table = pd.DataFrame(profiles)
    pairs = candidate_pairs(frames)
    families = assign_families(list(MECHDETECT_IDS), pairs, overrides)
    table["family_id"] = table.openml_id.map(families)
    table["representative"] = choose_representatives(table)
    table["role"] = assign_roles(table, n_dev, root_seed)
    return table, pairs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv-dir", type=Path, required=True)
    parser.add_argument("--overrides", type=Path, required=True)
    parser.add_argument("--saida", type=Path, required=True, help="CSV de perfis; os pares vão para <saida>.pairs.csv")
    parser.add_argument("--n-dev", type=int, default=DEFAULT_N_DEV_FAMILIES)
    parser.add_argument("--seed", type=int, default=DEFAULT_ROOT_SEED)
    args = parser.parse_args()
    pairs_path = args.saida.with_suffix(".pairs.csv")
    for path in (args.saida, pairs_path):
        if path.exists():
            raise SystemExit(f"{path} já existe; não sobrescrevo.")
    overrides = pd.read_csv(args.overrides, comment="#")
    table, pairs = build_from_local(args.csv_dir, overrides, args.n_dev, args.seed)
    args.saida.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.saida, index=False)
    pairs.to_csv(pairs_path, index=False)
    used = table[table.representative]
    print(
        f"tabelas: {len(table)}; incluídas: {int(table.included.sum())}; famílias com representante: {len(used)} "
        f"(desenvolvimento: {int((used.role == 'dev').sum())}; confirmatórias: {int((used.role == 'confirm').sum())})"
    )
    print(table.groupby("exclusion_reason").size().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
