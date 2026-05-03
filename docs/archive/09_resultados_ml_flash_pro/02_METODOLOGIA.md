# Metodologia da Comparação ML × Flash × Pro

**Data:** 2026-04-25

---

## 1. Objetivo do experimento

Isolar o **ganho marginal** de adicionar features extraídas via LLM (`llm_ctx_*`, 9 features) sobre um pipeline de classificação ML que já contém 25 features estatísticas, mantendo absolutamente fixos:

- Benchmark (29 datasets reais, 1.421 bootstraps)
- Conjunto de features estatísticas (idêntico nas 3 configurações)
- Estratégia de split (GroupShuffleSplit 75/25 + Group 5-Fold CV)
- SMOTE (k=3, random_state=42)
- Hiperparâmetros dos 7 classificadores
- Random seed (=42 em tudo)

A única variável livre é o **conjunto de features adicionadas**:
- ML-only: 0 features LLM (apenas as 25 estatísticas)
- Flash + ML: 9 features LLM extraídas com gemini-3-flash-preview
- Pro + ML: 9 features LLM extraídas com gemini-3-pro-preview + Step 1 prompt

## 2. Origem dos dados de cada configuração

### 2.1 ML-only

**Não houve nova extração.** Aproveitou-se o `X_features.csv` produzido por `step1_v2_neutral` (que já tinha as 34 features, sendo 25 estatísticas + 9 LLM Pro), e **filtrou-se** apenas as 25 colunas estatísticas:

```python
stat_cols = [c for c in X_full.columns if not c.startswith('llm_')]
X_ml_only = X_full[stat_cols]
```

A filtragem é determinística e perfeitamente reproduzível. Como as features estatísticas são calculadas **sem nenhuma dependência das features LLM** (ver `extract_features.py` linhas 110-200, onde features estatísticas/CAAFE/MechDetect são computadas em isolamento e antes das chamadas LLM), os valores das 25 colunas estatísticas em `step1_v2_neutral` são **idênticos** ao que seria produzido em uma extração ML-pura.

**Caveat menor:** a imputação por mediana das colunas `llm_*` em `step1_v2_neutral` foi feita usando o conjunto completo de amostras com LLM presentes. Isso não afeta as 25 colunas estatísticas (que já estão imputadas com `fillna(0)` separadamente — linha 352 do `extract_features.py`). Logo, a filtragem é equivalente a um experimento controlado.

### 2.2 Flash + ML

**Origem:** experimento `step10_flash_ca_neutral` (executado em 2026-04-21).

Configuração:
- Modelo: `gemini-3-flash-preview`
- Extrator: `context_aware` com prompt **original** (sem Step 1)
- Metadata: `neutral`
- Mesmas 29 datasets, mesmos 1.421 bootstraps

⚠️ **Caveat importante:** Flash usou prompt **original**, enquanto Pro usou Step 1 prompt. A comparação Flash vs Pro **não é controlada para o prompt** — mistura efeito do modelo + efeito do prompt. Para isolar puramente o modelo seria necessário rerodar Flash com Step 1 prompt (estimativa: ~$2-4 adicional).

### 2.3 Pro + ML

**Origem:** experimento `step1_v2_neutral` (executado em 2026-04-25, ver `08_step1_v2_neutral_results/`).

Configuração:
- Modelo: `gemini-3-pro-preview`
- Extrator: `context_aware` com prompt **Step 1** (3 exemplos canônicos + tipologia MNAR + anti-MAR-bias)
- Metadata: `neutral`
- Mesmas 29 datasets, mesmos 1.421 bootstraps

## 3. Comando equivalente para reproduzir cada configuração

### 3.1 ML-only

Não requer chamadas LLM. Pode ser reproduzido em duas formas equivalentes:

**Opção A: pipeline oficial (`train_model.py --model none`)**

```bash
cd "IC - ITA 2/Scripts/v2_improved"
# Re-extração apenas com features estatísticas
uv run python extract_features.py \
    --model none --data real --metadata-variant neutral \
    --experiment baseline_ml_only_29
# Treinamento
uv run python train_model.py \
    --model none --data real --experiment baseline_ml_only_29
```

Nota: `--model none` instrui `extract_features.py` a pular todas as chamadas LLM. Custo: $0. Tempo: ~30s (apenas pandas/sklearn).

**Opção B: filtrar X_features.csv existente (usado neste documento)**

```python
# Carrega step1_v2_neutral X_features e filtra LLM cols
X = pd.read_csv("Output/v2_improved/step1_v2_neutral/.../X_features.csv")
X_ml = X[[c for c in X.columns if not c.startswith('llm_')]]
# Treinar com sklearn diretamente
```

A Opção B foi usada porque é instantânea e idêntica em resultados.

### 3.2 Flash + ML

Já está em `Output/v2_improved/step10_flash_ca_neutral/`. Para reproduzir do zero (custo ~$2-4):

```bash
uv run python extract_features.py \
    --model gemini-3-flash-preview --data real \
    --llm-approach context --metadata-variant neutral \
    --experiment step10_flash_ca_neutral_rerun
uv run python train_model.py --model gemini-3-flash-preview --data real \
    --experiment step10_flash_ca_neutral_rerun
```

### 3.3 Pro + ML

Já está em `Output/v2_improved/step1_v2_neutral/`. Para reproduzir (custo ~$30-36, ~1h33min):

```bash
# Metade 1
uv run python extract_features.py --model gemini-3-pro-preview --data real \
    --llm-approach context --metadata-variant neutral \
    --datasets-include data/datasets_part1.txt \
    --experiment step1_v2_neutral_part1 --workers 10
# Metade 2
uv run python extract_features.py --model gemini-3-pro-preview --data real \
    --llm-approach context --metadata-variant neutral \
    --datasets-include data/datasets_part2.txt \
    --experiment step1_v2_neutral_part2 --workers 10
# Merge + treino
uv run python merge_halves.py
uv run python train_model.py --model gemini-3-pro-preview --data real \
    --experiment step1_v2_neutral
```

## 4. Variáveis controladas e variáveis livres

| Variável | ML-only | Flash | Pro | Controlada? |
|----------|:-------:|:-----:|:---:|:-----------:|
| Benchmark (29 datasets) | igual | igual | igual | ✅ |
| Bootstraps (1421) | igual | igual | igual | ✅ |
| Features estatísticas (25) | iguais | iguais | iguais | ✅ |
| GroupShuffleSplit (seed=42) | igual | igual | igual | ✅ |
| Train/test groups | iguais | iguais | iguais | ✅ |
| SMOTE (k=3) | igual | igual | igual | ✅ |
| 7 classificadores | iguais | iguais | iguais | ✅ |
| Hiperparâmetros | iguais | iguais | iguais | ✅ |
| **Modelo LLM** | nenhum | Flash | Pro | ❌ (variável) |
| **Prompt LLM** | n/a | original | Step 1 | ❌ (variável adicional) |
| Metadata variant | n/a | neutral | neutral | ✅ (entre Flash/Pro) |

⚠️ **Limitação principal:** Flash usa prompt original, Pro usa Step 1. A comparação Flash vs Pro **mistura efeito do modelo e do prompt**. Para isolar:
- "Pro com prompt original" não foi rodado (estimativa: ~$30 adicional)
- "Flash com Step 1 prompt" não foi rodado (estimativa: ~$2-4 adicional)

Para o objetivo principal (decidir se LLM agrega valor), a comparação ML vs Pro e ML vs Flash já é controlada e suficiente.

## 5. Métricas reportadas

### 5.1 Holdout

- Split: GroupShuffleSplit 1 split, test_size=0.25, random_state=42
- 21 grupos no train (1.026 amostras), 8 grupos no test (395 amostras)
- Aplica-se SMOTE no train (resultando em 1.200 amostras pós-balanceamento)
- Reporta-se acurácia simples no test set

### 5.2 Cross-Validation

- Estratégia: Group 5-Fold (`GroupKFold(n_splits=5)`)
- Aplica-se SMOTE em cada train fold (não no test fold)
- Reporta-se acurácia média ± desvio-padrão entre os 5 folds

### 5.3 Por-classe (precision, recall, F1)

Apenas reportadas no holdout (linhas individuais por classe MCAR/MAR/MNAR), conforme `relatorio.txt` de cada experimento.

## 6. Auditoria de leakage

Todas as 3 configurações respeitam:
- ✅ **Canal C** (bootstrap leakage): GroupShuffleSplit + GroupKFold com `groups.csv` (parent-dataset)
- ✅ **Canal F** (`missing_context` revela rótulo): `--metadata-variant neutral` em Flash/Pro
- ✅ **Canais A, B, D, E**: fechados conforme audit em `08_step1_v2_neutral_results/02_METODOLOGIA.md`

## 7. Estatística e significância

**Não foram realizados testes de significância estatística** (McNemar, bootstrap-t, etc.) entre as 3 configurações. Razões:

1. Single seed: variância de procedimento não medida
2. Mesma amostra: testes pareados são possíveis mas requereriam acesso às predições per-amostra das 3 configurações simultaneamente
3. Custo: implementar testes formais ($0) requer trabalho de engenharia adicional

**Recomendação para futura iteração:** rodar McNemar entre Pro vs ML usando as `predictions.csv` de cada experimento, para confirmar significância da diferença +1.86pp CV.

## 8. Reprodutibilidade

| Artefato | Caminho |
|----------|---------|
| ML-only X | `Output/v2_improved/step1_v2_neutral/.../X_features.csv` (filtrar `llm_*`) |
| Flash X | `Output/v2_improved/step10_flash_ca_neutral/.../X_features.csv` |
| Pro X | `Output/v2_improved/step1_v2_neutral/.../X_features.csv` |
| Listas de datasets | `Scripts/v2_improved/data/datasets_part{1,2}.txt` |
| Script de extração | `Scripts/v2_improved/extract_features.py` |
| Script de treino | `Scripts/v2_improved/train_model.py` |
| Script de merge | `Scripts/v2_improved/merge_halves.py` |

Todos versionados em git (branch: `last-test-check` no momento desta documentação).
