# STEP 05-B: LOGO Cross-Validation

**Status: PENDENTE**
**Estimativa: 1 dia**
**Dependência: Steps 05-A, 04-B, 06, 07 (rodar com todos os métodos)**

---

## Motivação

GroupKFold com k=5 agrupa vários datasets no mesmo fold. LOGO (Leave-One-Group-Out) é mais rigoroso: cada fold exclui TODOS os bootstraps de 1 dataset inteiro.

**Problema atual:** GroupKFold tem CV variance de ~30% em dados reais. LOGO dá uma estimativa mais honesta de "como o modelo performa em um dataset completamente novo".

---

## Implementação

**Arquivo a modificar:** `v2_improved/train_model.py`

### Lógica

```python
from sklearn.model_selection import LeaveOneGroupOut

logo = LeaveOneGroupOut()
# groups = nome do dataset de origem de cada amostra

# Com 23 datasets reais → 23 folds
# Cada fold: treina com 22 datasets, testa com 1
for train_idx, test_idx in logo.split(X, y, groups):
    model.fit(X[train_idx], y[train_idx])
    score = model.score(X[test_idx], y[test_idx])
```

### O que rodar

Rodar LOGO para as variantes principais do paper (V1-V6 do STEP05A + baselines):

**Variantes hierárquicas (STEP05A):**
1. V1: Direto 3-way baseline (21 features)
2. V2: Hierárquico puro (21 feat, ambos níveis)
3. V4: **Hierárquico + LLM no Nível 2** (21 stat + 8 LLM no N2) ← foco principal
4. V6: Hierárquico + LLM em ambos (29 feat, ambos níveis) ← controle

**Baselines externos (STEP06/07):**
5. MechDetect original (STEP 06)
6. PKLM + heurísticas (STEP 07)

### Comparação com GroupKFold

| Método | GroupKFold (k=5) | LOGO (k=23) | Delta |
|--------|:----------------:|:-----------:|:-----:|
| Baseline | ~41% ± 30% (step03_final, real) | ? ± ? | ? |
| + CAAFE | ? | ? | ? |
| + LLM | ? | ? | ? |
| Hierárquico | ? | ? | ? |
| MechDetect | ? | ? | ? |

**Interpretação:**
- Se LOGO ≈ GroupKFold → sem leakage residual, estimativas confiáveis
- Se LOGO << GroupKFold → havia leakage entre grupos, GroupKFold superestimava
- Se LOGO >> GroupKFold → GroupKFold era pessimista (possível com k pequeno)

---

## Testes de Validação

### Teste 1: LOGO funciona
23 folds executam sem erro, cada fold tem amostras de exatamente 1 dataset no teste.

### Teste 2: Variance LOGO < GroupKFold
Com 23 folds (vs 5), a variância deve ser menor (mais folds = estimativa mais estável).

### Teste 3: Sem leakage
Confirmar que nenhum bootstrap do dataset de teste aparece no treino.

---

## Critério de Conclusão

- [ ] LOGO implementada em train_model.py
- [ ] Rodada para todos os métodos do benchmark
- [ ] Comparação LOGO vs GroupKFold documentada
- [ ] Interpretação de leakage residual
