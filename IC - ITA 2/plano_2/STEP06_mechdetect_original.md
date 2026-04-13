# STEP 06: Reimplementar MechDetect Fiel

**Status: PENDENTE**
**Estimativa: 1-2 dias**
**Papel no paper: Baseline de comparação (não é o foco principal)**

---

## Motivação

MechDetect (Jung et al., 2024) é o **estado da arte** em classificação automática de mecanismos de missing data, reportando 89% de accuracy em 101 datasets sintéticos. Nosso pipeline já usa 6 features **inspiradas** no MechDetect, mas não é uma reimplementação fiel.

Para o paper, MechDetect serve como **baseline de comparação** contra nossa abordagem hierárquica + LLM. Não é o foco principal, mas precisamos de uma comparação justa.

**Referência:** `Artigos_Relevantes/08_Classificacao_Mecanismos_Missing_Data/MechDetect_Detecting_Data_Dependent_Errors_2024.pdf`

---

## Diferenças Atuais vs Original

| Aspecto | MechDetect Original | Nossa Implementação (features/mechdetect.py) |
|---------|--------------------|--------------------------------------------|
| Classificador base | HistGradientBoosting | LogisticRegression (escolha de velocidade) |
| CV para AUC | 10-fold | 3-fold (adaptativo) |
| Regra de decisão | Thresholds nos AUCs | Features alimentam outro classificador |
| Features extraídas | 3 AUCs + deltas + MWU | Mesmo, mas com classificador diferente |
| Classificação final | Regra baseada em thresholds | ML 7 classificadores |

---

## Implementação

**Arquivo a criar:** `v2_improved/baselines/mechdetect_original.py`

### Passo 1: Reimplementar extração de features como no paper

```python
# Usar HistGradientBoosting (como no paper)
from sklearn.ensemble import HistGradientBoostingClassifier

# 3 tarefas com 10-fold CV:
# Complete: X_all → mask (todas as features predizem missingness)
# Shuffled: X_all → mask_shuffled (labels permutados - baseline)  
# Excluded: X_without_X0 → mask (sem a variável com missing)
```

### Passo 2: Implementar regra de decisão do MechDetect

A regra original usa thresholds nos AUCs para classificar:
- Se AUC_complete ≈ AUC_shuffled (ambos ~0.5) → **MCAR** (nada prediz missing)
- Se AUC_complete >> AUC_shuffled E AUC_excluded ≈ AUC_complete → **MAR** (X1-X4 predizem missing)
- Se AUC_complete >> AUC_shuffled E AUC_excluded < AUC_complete → **MNAR** (X0 contribui para predição)

**Importante:** Ler o paper cuidadosamente para extrair os thresholds exatos. O paper pode usar regras heurísticas OU um classificador treinado nos AUCs — implementar ambos:
- **MechDetect-Rules:** Regras de threshold como descrito no paper (fiel ao original)
- **MechDetect-ML:** Usar os mesmos 3 AUCs + deltas como features em nossos 7 classificadores (variante nossa)

### Passo 3: Rodar nos nossos datasets

- 1200 sintéticos (12 variantes, 100 cada)
- 23 reais (com bootstraps)
- Salvar: AUC_complete, AUC_shuffled, AUC_excluded, deltas, predição final

### Passo 4: Comparação head-to-head

| Método | Acc Sint | Acc Real | MCAR R | MAR R | MNAR R |
|--------|:--------:|:--------:|:------:|:-----:|:------:|
| MechDetect original | ? | ? | ? | ? | ? |
| Direto 3-way baseline (21 feat) | 77% | 43% | ? | ? | ? |
| **Hier. + LLM no N2 (proposto)** | **?** | **?** | **?** | **?** | **?** |
| Hier. puro (sem LLM) | ? | ? | ? | ? | ? |

---

## O que esperamos encontrar

1. **MechDetect original ≈ 89% em sintético** (reproduzindo o paper) — mas nossos dados são diferentes (12 variantes vs 101 do paper)
2. **MechDetect < nosso pipeline em real** — porque o paper não testou em dados reais e thresholds fixos podem não generalizar
3. **Nosso pipeline + features do MechDetect original pode melhorar** — combinar o melhor dos dois mundos

---

## Testes de Validação

### Teste 1: Reprodução do MechDetect
Accuracy em sintético deve ser > 80% (mesmo com datasets diferentes dos do paper original). Se < 70%, a reimplementação tem erro.

### Teste 2: AUCs fazem sentido
- MCAR: AUC_complete ≈ 0.5 (nada prediz missing)
- MAR: AUC_complete >> 0.5, AUC_excluded ≈ AUC_complete
- MNAR: AUC_complete >> 0.5, AUC_excluded < AUC_complete

### Teste 3: Comparação justa
Usar mesmos datasets, mesma partição treino/teste, mesma validação cruzada para todos os métodos.

---

## Critério de Conclusão

- [ ] MechDetect reimplementado fielmente (HistGBT, 10-fold)
- [ ] Regra de decisão por thresholds implementada
- [ ] Rodado em 1200 sintéticos + 23 reais
- [ ] Accuracy reportada e comparada com paper original
- [ ] Tabela head-to-head completa
- [ ] Features do MechDetect original testadas como input no nosso pipeline
