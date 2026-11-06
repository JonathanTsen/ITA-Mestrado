"""Obtém de novo, pela fonte original, as tabelas do estudo de caso com ausência natural (passo 45; decisão D3).

Sem espelhos não verificados e sem TLS desligado. Os valores não são alterados: a única conversão é ler o código de
ausência que a própria fonte usa ("?" no UCI e no ARFF do OpenML; NA no R). Saídas:
- `data_rb/case_study_raw/<fonte>/`: arquivos baixados (fora do git);
- `data_rb/case_study_tables/<fonte>.csv`: tabela padronizada que o `case-study` lê (fora do git);
- `data_rb/case_study_manifest.csv` (versionado): URL, data, SHA-256 dos brutos e da tabela, checagem e licença.

Uso (na raiz do repositório): uv run --frozen python scripts/revisao_banca/passo45/obter_fontes.py
"""

import csv
import hashlib
import io
import subprocess
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data_rb" / "case_study_raw"
TABLES = ROOT / "data_rb" / "case_study_tables"
MANIFEST = ROOT / "data_rb" / "case_study_manifest.csv"
R_SCRIPT = Path(__file__).with_name("exportar_r.R")

OPENML = {  # fonte: ID do OpenML (a licença vem da descrição do próprio OpenML)
    "openml_55_hepatitis": 55,
    "openml_38_sick": 38,
    "openml_57_hypothyroid": 57,
    "openml_6332_cylinder_bands": 6332,
    "openml_25_colic": 25,
    "openml_42972_chronic_kidney": 42972,
    "openml_37_diabetes": 37,
    "openml_40945_titanic": 40945,
    "openml_15_breast_w": 15,
}
R_CRAN = {  # fonte: (pacote, objeto, licença do pacote)
    "mice_boys": ("mice", "boys", "GPL-2 | GPL-3 (pacote mice)"),
    "mice_brandsma": ("mice", "brandsma", "GPL-2 | GPL-3 (pacote mice)"),
    "naniar_oceanbuoys": ("naniar", "oceanbuoys", "MIT (pacote naniar); dados NOAA TAO, domínio público"),
    "wooldridge_mroz": ("wooldridge", "mroz", "GPL-3 (pacote wooldridge); dados de Mroz (1987), PSID"),
}
NHANES = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2017/DataFiles/{}.xpt"
UCI_MAMMOGRAPHIC = "https://archive.ics.uci.edu/static/public/161/mammographic+mass.zip"
SUPPORT2 = "https://hbiostat.org/data/repo/support2csv.zip"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        with urllib.request.urlopen(url, timeout=120) as response:
            dest.write_bytes(response.read())
    return dest


def save_table(source: str, frame: pd.DataFrame) -> Path:
    TABLES.mkdir(parents=True, exist_ok=True)
    path = TABLES / f"{source}.csv"
    frame.to_csv(path, index=False)
    return path


def openml_sources() -> list[dict]:
    rows = []
    for source, data_id in OPENML.items():
        home = RAW / source
        bunch = fetch_openml(data_id=data_id, as_frame=True, parser="liac-arff", data_home=str(home))
        frame = bunch.frame
        raw_files = sorted(p for p in home.rglob("*") if p.is_file())
        path = save_table(source, frame)
        details = bunch.details
        rows.append({
            "source": source, "origin": f"OpenML {data_id} ({details['name']} v{details['version']})",
            "url": details.get("url", f"https://www.openml.org/d/{data_id}"),
            "raw_files": ";".join(f"{p.relative_to(RAW)}:{sha256(p)}" for p in raw_files),
            "checksum_check": f"md5 do OpenML conferido pelo sklearn ({details.get('md5_checksum', '')})",
            "licence": details.get("licence", ""), "version": details["version"], "table": path.name,
        })  # fmt: skip
    return rows


def r_sources() -> list[dict]:
    rows = []
    for source, (package, obj, licence) in R_CRAN.items():
        out = RAW / source
        info = (
            subprocess.run(
                ["Rscript", str(R_SCRIPT), "cran", package, obj, str(out)], capture_output=True, text=True, check=True
            )
            .stdout.strip()
            .split("\t")
        )
        tarball = out / info[2]
        frame = pd.read_csv(out / f"{obj}.csv")
        path = save_table(source, frame)
        rows.append({
            "source": source, "origin": f"CRAN {package} {info[1]}, objeto {obj}",
            "url": f"https://cloud.r-project.org/src/contrib/{info[2]}",
            "raw_files": f"{tarball.relative_to(RAW)}:{sha256(tarball)}", "checksum_check": "SHA-256 do tarball do CRAN",
            "licence": licence, "version": info[1], "table": path.name,
        })  # fmt: skip
    out = RAW / "r_airquality"
    info = (
        subprocess.run(
            ["Rscript", str(R_SCRIPT), "base", "airquality", str(out)], capture_output=True, text=True, check=True
        )
        .stdout.strip()
        .split("\t")
    )
    path = save_table("r_airquality", pd.read_csv(out / "airquality.csv"))
    rows.append({
        "source": "r_airquality", "origin": f"R base, datasets::airquality ({info[1]})", "url": "R base (pacote datasets)",
        "raw_files": f"airquality.csv:{sha256(out / 'airquality.csv')}", "checksum_check": "SHA-256 do CSV exportado",
        "licence": "GPL-2 | GPL-3 (R); dados de Chambers et al. (1983)", "version": info[1], "table": path.name,
    })  # fmt: skip
    return rows


def nhanes_sources() -> list[dict]:
    out = RAW / "nhanes_2017"
    files = {name: download(NHANES.format(name), out / f"{name}.xpt") for name in ("DEMO_J", "PBCD_J", "COT_J")}
    demo = pd.read_sas(files["DEMO_J"], format="xport")[["SEQN", "RIAGENDR", "RIDAGEYR"]]
    rows = []
    for name, source in (("PBCD_J", "nhanes_2017_pbcd"), ("COT_J", "nhanes_2017_cot")):
        lab = pd.read_sas(files[name], format="xport")
        path = save_table(source, lab.merge(demo, on="SEQN", how="left"))
        rows.append({
            "source": source, "origin": f"NHANES 2017–2018, {name} + DEMO_J (CDC/NCHS)",
            "url": NHANES.format(name), "raw_files": ";".join(f"{p.relative_to(RAW)}:{sha256(p)}" for p in
                                                              (files[name], files["DEMO_J"])),
            "checksum_check": "SHA-256 dos XPT do CDC", "licence": "domínio público (governo dos EUA)",
            "version": "2017–2018 (ciclo J)", "table": path.name,
        })  # fmt: skip
    return rows


def zip_sources() -> list[dict]:
    rows = []
    raw = download(UCI_MAMMOGRAPHIC, RAW / "uci_161_mammographic" / "mammographic+mass.zip")
    with zipfile.ZipFile(raw) as z:
        data = z.read("mammographic_masses.data").decode()
    names = ["BI_RADS", "Age", "Shape", "Margin", "Density", "Severity"]
    frame = pd.read_csv(io.StringIO(data), names=names, na_values="?")
    path = save_table("uci_161_mammographic", frame)
    rows.append({
        "source": "uci_161_mammographic", "origin": "UCI Machine Learning Repository, dataset 161 (Mammographic Mass)",
        "url": UCI_MAMMOGRAPHIC, "raw_files": f"{raw.relative_to(RAW)}:{sha256(raw)}",
        "checksum_check": "SHA-256 do zip do UCI", "licence": "CC BY 4.0", "version": "UCI 161", "table": path.name,
    })  # fmt: skip
    raw = download(SUPPORT2, RAW / "hbiostat_support2" / "support2csv.zip")
    with zipfile.ZipFile(raw) as z:
        member = next(n for n in z.namelist() if n.endswith(".csv"))
        frame = pd.read_csv(io.BytesIO(z.read(member)))
    path = save_table("hbiostat_support2", frame)
    rows.append({
        "source": "hbiostat_support2", "origin": "SUPPORT2, Vanderbilt Department of Biostatistics (F. Harrell)",
        "url": SUPPORT2, "raw_files": f"{raw.relative_to(RAW)}:{sha256(raw)}",
        "checksum_check": "SHA-256 do zip do repositório", "licence": "ver fonte (citar Knaus et al. 1995)",
        "version": member, "table": path.name,
    })  # fmt: skip
    return rows


def main() -> int:
    retrieved = datetime.now(UTC).isoformat(timespec="seconds")
    rows = [*openml_sources(), *r_sources(), *nhanes_sources(), *zip_sources()]
    for row in rows:
        table = TABLES / row["table"]
        frame = pd.read_csv(table)
        row |= {
            "retrieved_utc": retrieved,
            "n_rows": len(frame),
            "n_cols": frame.shape[1],
            "table_sha256": sha256(table),
        }
    with MANIFEST.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} fontes → {MANIFEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
