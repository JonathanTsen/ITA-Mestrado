"""Descrições completas do OpenML das tabelas-fonte, congeladas para o braço A3 do LLM (30/10/2026).

O A3 (semântico, exploratório; passo 42 §2) recebe os nomes reais das colunas e uma descrição neutra da base. As
descrições do OpenML podem mudar, então o texto cru é gravado com data e SHA-256; a neutralização é feita pelo código
(`missdetect.rb.llm.render.neutral_description`), sempre a partir deste arquivo.

Uso: python3 -I scripts/revisao_banca/llm/openml_descricoes.py data_rb/sources_local.csv data_rb/openml_descriptions_<data>.csv
"""

import csv
import hashlib
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

sources, out = sys.argv[1], sys.argv[2]
with open(sources, newline="") as fh:
    ids = sorted({int(r["openml_id"]) for r in csv.DictReader(fh)})


def fetch(i: int) -> dict:
    err = ""
    for _attempt in range(3):
        try:
            with urllib.request.urlopen(f"https://www.openml.org/api/v1/json/data/{i}", timeout=60) as r:
                d = json.load(r)["data_set_description"]
            break
        except Exception as exc:  # rede: tenta de novo e registra a falha
            err = repr(exc)
            time.sleep(2)
    else:
        return {"openml_id": i, "status": f"failed: {err}", "name": "", "version": "", "description": "",
                "description_sha256": "", "fetched_at": datetime.now(UTC).isoformat(timespec="seconds")}
    text = d.get("description") or ""
    return {"openml_id": i, "status": "ok", "name": d.get("name", ""), "version": d.get("version", ""),
            "description": text, "description_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "fetched_at": datetime.now(UTC).isoformat(timespec="seconds")}


with ThreadPoolExecutor(max_workers=8) as pool:  # a API levou ≈ 26 s por pedido em 30/10/2026
    rows = sorted(pool.map(fetch, ids), key=lambda r: r["openml_id"])
with open(out, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(sum(r["status"] == "ok" for r in rows), "de", len(rows))
