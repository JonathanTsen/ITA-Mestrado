"""Metadados do OpenML para as 101 tabelas do MechDetect (passo 31): só a descrição, sem baixar dados.

Uso: python3 -I scripts/revisao_banca/passo31/openml_metadados.py <ids separados por vírgula> <saída.csv>
"""

import csv
import json
import sys
import time
import urllib.request

ids = [int(x) for x in sys.argv[1].split(",")]
out = sys.argv[2]
rows = []
for i in ids:
    for _attempt in range(3):
        try:
            with urllib.request.urlopen(f"https://www.openml.org/api/v1/json/data/{i}", timeout=30) as r:
                d = json.load(r)["data_set_description"]
            break
        except Exception as exc:
            err = repr(exc)
            time.sleep(2)
    else:
        rows.append({"openml_id": i, "status": f"failed: {err}"})
        continue
    rows.append(
        {
            "openml_id": i,
            "status": "ok",
            "name": d.get("name"),
            "version": d.get("version"),
            "licence": d.get("licence"),
            "original_data_url": d.get("original_data_url", ""),
            "collection_date": d.get("collection_date", ""),
            "creator": json.dumps(d.get("creator", "")),
            "default_target": d.get("default_target_attribute", ""),
            "md5_checksum": d.get("md5_checksum", ""),
            "upload_date": d.get("upload_date", ""),
            "format": d.get("format", ""),
            "description_head": (d.get("description") or "")[:300].replace("\n", " "),
        }
    )
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=sorted({k for r in rows for k in r}))
    w.writeheader()
    w.writerows(rows)
print(sum(r["status"] == "ok" for r in rows), "de", len(rows))
