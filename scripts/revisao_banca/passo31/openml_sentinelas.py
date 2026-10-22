"""Procura, nas descrições do OpenML, menções a valores que codificam 'não medido' (passo 31 / regra dos sentinelas).

Uso: python3 -I scripts/revisao_banca/passo31/openml_sentinelas.py data_rb/sources_local.csv
"""

import csv
import json
import re
import sys
import urllib.request

PATTERN = re.compile(
    r"(missing|not (?:measured|available|recorded)|unknown|zero[s]? (?:mean|indicat|denot|represent|code)|"
    r"coded as|encoded as|placeholder|sentinel|-999|9999|n/?a\b)",
    re.IGNORECASE,
)
with open(sys.argv[1], newline="") as fh:
    src = [r for r in csv.DictReader(fh) if r["representative"] == "True"]
for r in src:
    i = r["openml_id"]
    d = json.load(urllib.request.urlopen(f"https://www.openml.org/api/v1/json/data/{i}", timeout=30))[
        "data_set_description"
    ]
    text = d.get("description") or ""
    hits = sorted({m.group(0).lower() for m in PATTERN.finditer(text)})
    if hits:
        snippets = [text[max(0, m.start() - 80) : m.end() + 80].replace("\n", " ") for m in PATTERN.finditer(text)][:3]
        print(f"{r['role']} {i} {d['name']}: {hits}")
        for s in snippets:
            print("    …", s, "…")
