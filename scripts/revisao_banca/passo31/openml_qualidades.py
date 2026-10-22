"""Qualidades do OpenML (nº de instâncias, atributos e ausências) para conferir os CSV-fonte (passo 31).

Uso: python3 -I scripts/revisao_banca/passo31/openml_qualidades.py <ids separados por vírgula> <saída.csv>
"""

import csv
import json
import sys
import time
import urllib.request

ids = [int(x) for x in sys.argv[1].split(",")]
rows = []
for i in ids:
    q = {}
    for _attempt in range(3):
        try:
            with urllib.request.urlopen(f"https://www.openml.org/api/v1/json/data/qualities/{i}", timeout=30) as r:
                q = {x["name"]: x["value"] for x in json.load(r)["data_qualities"]["quality"]}
            break
        except Exception:
            time.sleep(2)
    rows.append(
        {
            "openml_id": i,
            "n_instances": q.get("NumberOfInstances"),
            "n_features": q.get("NumberOfFeatures"),
            "n_missing": q.get("NumberOfMissingValues"),
            "n_numeric": q.get("NumberOfNumericFeatures"),
        }
    )
with open(sys.argv[2], "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
