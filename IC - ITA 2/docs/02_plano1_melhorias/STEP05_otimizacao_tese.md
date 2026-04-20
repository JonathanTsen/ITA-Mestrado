# STEP 05: Otimizacao Final + Documentacao para Tese

**Fase 4E — Refinar, validar, documentar**

---

## Parte A: Classificacao Hierarquica

### Motivacao

Classificar 3 classes diretamente e dificil porque MCAR, MAR e MNAR tem sinais de natureza diferente:
- MCAR = ausencia de sinal (missing nao correlaciona com nada)
- MAR = sinal em X1-X4 (missing correlaciona com variaveis observadas)
- MNAR = sinal em X0 (missing correlaciona com o proprio valor faltante)

Um modelo unico precisa aprender todos esses sinais simultaneamente.

### Logica

**Nivel 1: MCAR vs NAO-MCAR**
- Features mais relevantes: little_proxy_score, mask_entropy, mechdetect_delta_complete_shuffled, X0_missing_rate
- Se AUC_complete ≈ AUC_shuffled → MCAR (missing nao depende de nada)
- Classificador binario (mais facil que 3 classes)

**Nivel 2: MAR vs MNAR (so para amostras classificadas como NAO-MCAR)**
- Features mais relevantes: mechdetect_delta_complete_excluded, X0_ks_obs_vs_imputed, X0_tail_missing_ratio, X0_censoring_score, log_pval_X1_mask
- Se AUC_excluded ≈ AUC_complete → MNAR (excluir X0 nao reduz poder preditivo do modelo → depende de X0)
- Se AUC_excluded < AUC_complete → MAR (X1-X4 contem a informacao)

### Implementacao

Criar novo modo em `train_model.py` (flag `--hierarchical`) que:
1. Treina classificador binario MCAR vs {MAR, MNAR}
2. Para amostras classificadas como nao-MCAR, treina segundo classificador MAR vs MNAR
3. Combina predicoes
4. Compara accuracy com classificacao direta de 3 classes

### Quando usar

Usar classificacao hierarquica SE a classificacao direta tiver recall MNAR < 30%. Se a direta ja funciona bem (>60%), a hierarquica pode nao trazer ganho.

---

## Parte B: Leave-One-Group-Out Cross-Validation (LOGO)

### Motivacao

GroupKFold com k=5 agrupa varios datasets no mesmo fold. LOGO e mais rigoroso: cada fold exclui TODOS os bootstraps de 1 dataset inteiro.

### Logica

Com 10 datasets por mecanismo (meta do STEP03):
- 30 datasets totais → 30 folds LOGO
- Cada fold treina com 29 datasets, testa com 1
- Estimativa honesta: "como o modelo performa em um dataset COMPLETAMENTE novo?"

### Quando usar

- Requer >= 8 datasets por classe (STEP03 precisa estar completo)
- Complementa GroupKFold, nao substitui — reportar ambos na tese
- Se LOGO accuracy >> GroupKFold accuracy → GroupKFold tinha algum leakage residual

---

## Parte C: Documentacao para Tese

### Resultados a incluir

**Tabela principal — Comparacao entre fases:**

| Metrica | v1 (pre-fases) | Fases 1-2 | Fase 3 (sem leakage) | Fase 4 (proposta) |
|---------|:-:|:-:|:-:|:-:|
| Melhor accuracy baseline | 90.9%* | 98.7%* | 60.7% | ? |
| MNAR recall | ? | ? | 0% | ? |
| CV variancia | 40% | 1.5%* | 55-74% | ? |
| LLM delta | -18.2% | +1.3% | -10.7% | ? |
| Data leakage | Sim | Sim | Nao | Nao |

*Valores inflados por data leakage

**Tabela de ablacao (do STEP04):**
Accuracy por configuracao de features x modelo

**Feature importance:**
Top 15 features com importancia, separadas por tipo (invariante, discriminativa, MechDetect, LLM)

**Confusion matrices:**
Antes vs depois, mostrando melhoria no recall MNAR

**Limitacao teorica:**
Secao sobre Focused vs Diffuse MNAR, com dados empiricos de quais datasets sao cada tipo

### Graficos a gerar

1. Barras: accuracy por modelo, comparando fases (v1, v2, v3, v4)
2. Heatmap: confusion matrix do melhor modelo antes e depois
3. Barras empilhadas: feature importance por tipo (stat, disc, mechdetect, llm)
4. Box plot: CV scores por fold (mostrando reducao de variancia)
5. Scatter: accuracy sintetico vs real (mostrando gap)

### Narrativa da tese

A historia dos dados deve ser:

1. **Dados sinteticos funcionam** — LLM adiciona +3% accuracy com features de segunda ordem
2. **Dados reais com avalicao ingênua parecem funcionar** — 98.7% baseline (mas leakage)
3. **Correcao do leakage revela o problema real** — 60.7%, MNAR indetectavel
4. **Diagnostico** — features sao fingerprints, poucos datasets, rotulos errados
5. **Solucao: features invariantes + MechDetect + mais dados** — (resultados da Fase 4)
6. **LLM contribui via [CAAFE/embeddings/nenhum]** — (resultado do STEP04)
7. **Limitacoes** — Diffuse MNAR teoricamente indetectavel, N datasets ainda pequeno

---

## Testes de Validacao

### Teste 1: Classificacao hierarquica vs direta
Comparar accuracy e recall MNAR. Hierarquica deve ter recall MNAR >= direta. Se nao, a direta e suficiente.

### Teste 2: LOGO vs GroupKFold
Comparar accuracy media e variancia. LOGO deve ter variancia menor (mais folds) mas accuracy similar (se nao houver leakage residual).

### Teste 3: Consistencia sintetico-real
As features mais importantes devem ser similares entre sintetico e real. Se no sintetico mechdetect_delta e #1 e no real X0_q75 volta a ser #1, as features ainda nao sao invariantes.

### Teste 4: Reproducao
Rodar pipeline completo 3x com seeds diferentes (42, 123, 456). Accuracy nao deve variar mais que 5% entre runs.

### Teste 5: Metas atingidas
Verificar criterios de sucesso finais:

| Metrica | Meta Minima | Resultado |
|---------|:-----------:|:---------:|
| Accuracy (melhor modelo, real) | > 70% | ? |
| Recall MNAR (real) | > 40% | ? |
| CV variancia (real) | < 20% | ? |
| LLM delta vs baseline | >= 0% | ? |
| Features invariantes importancia | > 30% | ? |

### Teste 6: Nao ha data leakage
Confirmar que GroupShuffleSplit nao tem overlap entre grupos treino/teste. Confirmar que LOGO exclui completamente cada dataset. Verificar via `training_summary.json`.

---

## Criterio de Conclusao

- [ ] Classificacao hierarquica testada e comparada com direta
- [ ] LOGO CV implementada e comparada com GroupKFold
- [ ] Todos os graficos e tabelas para tese gerados
- [ ] Narrativa da tese mapeada com dados reais
- [ ] Limitacao MNAR focused/diffuse documentada
- [ ] Metas de sucesso avaliadas honestamente
- [ ] Pipeline completo reprodutivel end-to-end
