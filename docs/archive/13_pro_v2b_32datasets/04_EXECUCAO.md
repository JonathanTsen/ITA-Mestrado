# Detalhes de execução — Gemini Pro v2b

## Estratégia de execução em duas metades

O Gemini Pro é mais lento que o Flash para extração de features context-aware.
Para viabilizar a execução sem timeout, os 32 datasets foram divididos em duas
metades de 16 datasets cada.

### Part 1 (16 datasets)

**Arquivo de filtro**: `src/missdetect/metadata/datasets_v2b_part1.txt`

```
MCAR_boys_hc, MCAR_boys_hgt, MCAR_brandsma_apr
MAR_airquality_ozone, MAR_breastcancer_barenuclei, MAR_cylinderbands_bladepressure,
MAR_cylinderbands_esavoltage, MAR_hypothyroid_t4u, MAR_mammographic_density,
MAR_oceanbuoys_airtemp
MNAR_adult_capitalgain, MNAR_colic_refluxph, MNAR_hepatitis_protime,
MNAR_kidney_pot, MNAR_kidney_sod, MNAR_mroz_wages
```

**Comando**:
```bash
uv run python -m missdetect.extract_features \
  --model gemini-3-pro-preview \
  --data real \
  --experiment step12_pro_v2b_part1 \
  --llm-approach context \
  --metadata-variant neutral \
  --datasets-include src/missdetect/metadata/datasets_v2b_part1.txt
```

**Resultado**: 791 bootstraps, 16 datasets, sem NaNs, sem erros.

Distribuição por dataset:
- 50 bootstraps: maioria dos datasets
- 48 bootstraps: MCAR_boys_hgt, MAR_hypothyroid_t4u (filtro missing_rate >= 1%)
- 45 bootstraps: MAR_breastcancer_barenuclei

### Part 2 (16 datasets)

**Arquivo de filtro**: `src/missdetect/metadata/datasets_v2b_part2.txt`

```
MCAR_brandsma_lpr, MCAR_hepatitis_albumin, MCAR_hepatitis_alkphosphate
MAR_oceanbuoys_humidity, MAR_sick_t3, MAR_sick_tsh, MAR_support2_pafi,
MAR_titanic_age, MAR_titanic_age_v2
MNAR_nhanes_cadmium, MNAR_nhanes_cotinine, MNAR_nhanes_mercury,
MNAR_pima_insulin, MNAR_pima_skinthickness, MNAR_support2_albumin,
MNAR_support2_bilirubin
```

**Comando**: mesmo da part1, trocando `part1` por `part2`.

**Resultado**: 799 bootstraps, 16 datasets, sem NaNs, sem erros.

Distribuição: MCAR 150 | MAR 299 | MNAR 350

### Merge

O script `src/missdetect/merge_halves.py` concatena os CSVs de part1 + part2
e re-imputa colunas LLM com mediana global do conjunto consolidado.

```bash
uv run python src/missdetect/merge_halves.py
```

**Resultado**: 1590 linhas, 32 datasets, 34 features.

O `merge_halves.py` foi atualizado para apontar para os diretórios corretos:
- `P1 = results/step12_pro_v2b_part1/real/ml_com_llm/gemini-3-pro-preview`
- `P2 = results/step12_pro_v2b_part2/real/ml_com_llm/gemini-3-pro-preview`
- `OUT = results/step12_pro_v2b_32datasets/real/ml_com_llm/gemini-3-pro-preview`

### Treinamento

```bash
uv run python -m missdetect.train_model \
  --model gemini-3-pro-preview \
  --data real \
  --experiment step12_pro_v2b_32datasets
```

Tempo de treinamento: ~30 segundos (7 modelos + CV).

## Notas de reprodutibilidade

- O `.env` com `GEMINI_API_KEY` deve estar em `src/missdetect/.env`
- A execução do Pro usa a mesma infraestrutura corrigida na Fase 12
  (ver `docs/archive/12_flash_v2b_32datasets/04_INFRAESTRUTURA.md`)
- Os bootstraps em `data/real/processed_chunks/` devem estar pré-gerados
- O checkpoint system do `extract_features.py` permite retomar execuções
  interrompidas automaticamente

## Tempos de execução

| Etapa         | Part1       | Part2       |
|---------------|-------------|-------------|
| Extração      | ~45 min     | ~50 min     |
| Merge         | < 1 seg     | —           |
| Treinamento   | ~30 seg     | —           |

Ritmo médio: ~5-8 segundos por bootstrap (LLM context-aware sem chamada de API,
usa features CAAFE-MNAR pré-computadas + scoring local).

## Arquivos gerados

```
results/step12_pro_v2b_part1/real/ml_com_llm/gemini-3-pro-preview/
  X_features.csv, y_labels.csv, groups.csv

results/step12_pro_v2b_part2/real/ml_com_llm/gemini-3-pro-preview/
  X_features.csv, y_labels.csv, groups.csv

results/step12_pro_v2b_32datasets/real/ml_com_llm/gemini-3-pro-preview/
  X_features.csv, y_labels.csv, groups.csv
  relatorio.txt, resultados.png, precisao_por_classe.png
  predictions.csv, metrics_per_class.csv, feature_importance.csv
  cv_scores.csv, confusion_matrices.json, hyperparameters.json
  feature_selection_log.json, training_summary.json

src/missdetect/metadata/
  datasets_v2b_part1.txt (novo)
  datasets_v2b_part2.txt (novo)
```
