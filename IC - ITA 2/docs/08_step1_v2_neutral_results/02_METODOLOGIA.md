# Metodologia — Step 1 V2 Neutral

**Data:** 2026-04-25

---

## 1. Pipeline geral

```
┌──────────────────┐    ┌─────────────────┐    ┌──────────────────┐
│ 29 datasets reais│ ──>│ Split balanceado│──> │ 2 runs de extração│
│ (1421 bootstraps)│    │ (15 + 14 grupos)│    │ Pro + neutral    │
└──────────────────┘    └─────────────────┘    └────────┬─────────┘
                                                        │
                              ┌─────────────────────────┘
                              v
                    ┌──────────────────┐
                    │  merge_halves.py │
                    │  (concat + reimpute)│
                    └────────┬─────────┘
                              v
                    ┌──────────────────┐
                    │  train_model.py  │
                    │  (7 modelos × CV)│
                    └────────┬─────────┘
                              v
                    ┌──────────────────┐
                    │ Métricas finais  │
                    │ relatorio.txt    │
                    │ predictions.csv  │
                    └──────────────────┘
```

## 2. Inventário do benchmark

### 2.1 Total: 29 datasets reais, 1.421 bootstraps

```
MCAR (9 datasets, 421 bootstraps):
  - autompg_horsepower (37)        - hepatitis_albumin (50)
  - breastcancer_barenuclei (45)   - hepatitis_alkphosphate (50)
  - creditapproval_a14 (39)        - hypothyroid_t4u (50)
  - cylinderbands_bladepressure (50)
  - cylinderbands_esavoltage (50)
  - echomonths_epss (50)

MAR (11 datasets, 550 bootstraps):
  - airquality_ozone (50)          - oceanbuoys_humidity (50)
  - colic_resprate (50)            - sick_t3 (50)
  - hearth_chol (50)               - sick_tsh (50)
  - kidney_hemo (50)               - titanic_age (50)
  - mammographic_density (50)      - titanic_age_v2 (50)
  - oceanbuoys_airtemp (50)

MNAR (9 datasets, 450 bootstraps):
  - adult_capitalgain (50)         - kidney_sod (50)
  - colic_refluxph (50)            - mroz_wages (50)
  - cylinderbands_varnishpct (50)  - pima_insulin (50)
  - hepatitis_protime (50)         - pima_skinthickness (50)
  - kidney_pot (50)
```

Os 6 datasets MCAR com bootstrap < 50 (autompg, breastcancer, creditapproval) refletem limitações dos dados originais (poucas linhas para gerar 50 bootstraps mantendo independência amostral).

### 2.2 Particionamento balanceado em halves

A divisão foi pensada para preservar diversidade de mecanismos em cada metade, permitindo teste isolado de cada metade se necessário, sem concentrar casos canônicos em apenas uma das partes.

**Metade 1 (15 datasets, 721 bootstraps) — `data/datasets_part1.txt`:**

| MCAR (5) | MAR (5) | MNAR (5) |
|----------|---------|----------|
| autompg_horsepower | airquality_ozone | adult_capitalgain |
| breastcancer_barenuclei | colic_resprate | colic_refluxph |
| creditapproval_a14 | hearth_chol | cylinderbands_varnishpct |
| cylinderbands_bladepressure | kidney_hemo | hepatitis_protime |
| cylinderbands_esavoltage | mammographic_density | kidney_pot |

**Metade 2 (14 datasets, 700 bootstraps) — `data/datasets_part2.txt`:**

| MCAR (4) | MAR (6) | MNAR (4) |
|----------|---------|----------|
| echomonths_epss | oceanbuoys_airtemp | kidney_sod |
| hepatitis_albumin | oceanbuoys_humidity | mroz_wages |
| hepatitis_alkphosphate | sick_t3 | pima_insulin |
| hypothyroid_t4u | sick_tsh | pima_skinthickness |
|  | titanic_age |  |
|  | titanic_age_v2 |  |

A Metade 2 contém os datasets canônicos mais conhecidos (`pima_insulin` para MNAR de seleção; `mroz_wages` para MNAR de truncamento; `oceanbuoys_*` para MAR fortemente correlacionado), enquanto a Metade 1 tem datasets de domínio mais homogêneo (predominância clínica e industrial).

## 3. Configuração de extração

### 3.1 Comandos exatos

**Metade 1:**

```bash
cd "IC - ITA 2/Scripts/v2_improved"
uv run python extract_features.py \
    --model gemini-3-pro-preview \
    --data real \
    --llm-approach context \
    --metadata-variant neutral \
    --datasets-include data/datasets_part1.txt \
    --experiment step1_v2_neutral_part1 \
    --workers 10
```

**Metade 2:** mesmo comando trocando `part1` → `part2`.

### 3.2 Modificação no `extract_features.py`

Foi adicionada uma flag CLI `--datasets-include FILE.txt` (linhas 67-80 e 230-247), que:
- Lê um arquivo texto com nomes de parent-datasets, um por linha (com `#` para comentários)
- Filtra a coleta de tasks para incluir apenas arquivos cujo prefixo (após remover `_bootNNN.txt`) bata com a lista
- Reporta no log: `📋 Filtro --datasets-include: {N} parent-datasets`
- Mostra contagem incluídos/excluídos

A filtragem usa a mesma regex (`r"_boot\d+\.txt$"`) que o cálculo de grupos (linha 377), garantindo consistência entre filtro e GroupKFold.

### 3.3 Hiperparâmetros LLM

- **Modelo:** `gemini-3-pro-preview`
- **Temperatura:** 0.1 (default do `LLMContextAwareExtractor`)
- **Workers paralelos:** 10 (rate-limit gentle para Pro + 2 chamadas/arquivo)
- **Total de chamadas LLM:** 2.842 (1.421 arquivos × 2 chamadas DAG+classification)
- **Cache:** in-memory por (stats, filename, data_type), invalidado a cada novo processo

## 4. Proteções contra vazamento de dados

A investigação de 2026-04-20 (`docs/99_tecnicos/investigacao_vazamento_self_consistency.md`) mapeou 6 canais de vazamento (A-F). O experimento atual confirma todos fechados:

| Canal | Mecanismo de vazamento | Status no `step1_v2_neutral` | Evidência |
|-------|------------------------|:----------------------------:|-----------|
| **A** | Mecanismo nomeado em `missing_context` (real) | ✅ Fechado | Metadata neutral usa "missingness is to be determined" |
| **B** | `expected_statistics` no metadata sintético | ✅ Fechado | Aplica-se apenas a sintéticos; não usado em `--data real` |
| **C** | Bootstrap leakage (mesmo dataset em train+test) | ✅ Fechado | GroupShuffleSplit + GroupKFold confirmam 0 grupos compartilhados (log linha "✅ Sem leakage: 0 grupos compartilhados") |
| **D** | Feature names codificando mecanismo | ✅ Fechado | LLM features são `llm_ctx_*` neutras |
| **E** | Synthetic metadata leaking tipologia | ✅ Fechado | Não aplicável (apenas reais) |
| **F** | `missing_context` revelando rótulo (real) | ✅ Fechado | `--metadata-variant neutral` carrega `real_datasets_metadata_neutral.json` |

**Verificação no log de extração:** ambas as runs mostram:
```
📖 Metadata variant: neutral (real_datasets_metadata_neutral.json)
```

**Verificação no log de treino:**
```
✅ Sem leakage: 0 grupos compartilhados
   Distribuição train: {0: 326, 1: 400, 2: 300}  (1026 amostras, 21 grupos)
   Distribuição test:  {0: 95,  1: 150, 2: 150}  (395 amostras, 8 grupos)
```

## 5. Configuração de treino

### 5.1 Split

- **GroupShuffleSplit** (n_splits=1, test_size=0.25, random_state=42)
- 21 grupos no treino, 8 grupos no teste
- Teste: `MAR_airquality_ozone`, `MAR_sick_tsh`, `MAR_titanic_age`, `MCAR_breastcancer_barenuclei`, `MCAR_echomonths_epss`, `MNAR_colic_refluxph`, `MNAR_cylinderbands_varnishpct`, `MNAR_pima_insulin`

A escolha de `pima_insulin` no holdout teste é particularmente informativa: é o caso canônico onde o modelo deveria acertar. Sua presença no teste (não no treino) testa a generalização.

### 5.2 Class balancing

- **SMOTE** com k=3 aplicado ao treino (não ao teste): {0: 326, 1: 400, 2: 300} → {0: 400, 1: 400, 2: 400}
- Total pós-SMOTE: 1.200 amostras de treino

### 5.3 Cross-Validation

- **Group 5-Fold** (estratégia padrão por dataset de origem)
- Cada fold contém grupos completos (todas as bootstraps de um dataset ficam no mesmo fold)

### 5.4 Modelos avaliados (7)

| Modelo | Hyperparâmetros (regime n>=100) |
|--------|----------------------------------|
| RandomForest | n_estimators=200, max_depth=10 |
| GradientBoosting | n_estimators=100, learning_rate=0.1 |
| LogisticRegression | C=1.0, max_iter=2000 |
| SVM_RBF | C=1.0, gamma='scale' (PCA pré-aplicado) |
| KNN | n_neighbors=7 (PCA pré-aplicado) |
| MLP | hidden_layer_sizes=(64,32), max_iter=500 (PCA pré-aplicado) |
| NaiveBayes | GaussianNB (default) |

PCA é aplicado em SVM/KNN/MLP quando há features LLM, conforme padrão do pipeline desde V3+.

## 6. Custos e tempos observados

| Métrica | Metade 1 | Metade 2 | Total |
|---------|:--------:|:--------:|:-----:|
| Bootstraps processados | 721 | 700 | 1.421 |
| Tempo (wall-clock) | 47:12 | 46:24 | 1h33min |
| Iter média | 3.93s | 3.98s | ~3.95s |
| Erros (Traceback/429/Exception) | 0 | 0 | 0 |
| Custo API estimado | ~$15-18 | ~$15-18 | **~$30-36** |

A estimativa de custo assume:
- Pro pricing: $1.25/M tokens input, $5/M output
- Avg input por chamada: ~3.5K tokens (DAG: 2K; classification: 5K)
- Avg output por chamada: ~700 tokens
- 2 chamadas por bootstrap × 1.421 bootstraps = 2.842 chamadas
- Total: ~10M input tokens + ~2M output tokens ≈ $12.5 + $10 = **$22.5 estimado mínimo**, podendo subir para $36 com overhead de retry/cache miss

## 7. Reproducibilidade

Todos os artefatos para reprodução:

| Artefato | Localização |
|----------|-------------|
| Listas de parent-datasets | `Scripts/v2_improved/data/datasets_part{1,2}.txt` |
| Script de merge | `Scripts/v2_improved/merge_halves.py` |
| Modificação CLI | `Scripts/v2_improved/extract_features.py` (linhas 67-80, 230-247) |
| Metadata neutral | `Scripts/v2_improved/data/real_datasets_metadata_neutral.json` |
| Outputs Metade 1 | `Output/v2_improved/step1_v2_neutral_part1/` |
| Outputs Metade 2 | `Output/v2_improved/step1_v2_neutral_part2/` |
| Outputs consolidados | `Output/v2_improved/step1_v2_neutral/` |
| Plano original | `~/.claude/plans/na-verdade-veja-se-glowing-candle.md` |

**Seed determinística:** `random_state=42` em todas as etapas (split, SMOTE, classifiers).
