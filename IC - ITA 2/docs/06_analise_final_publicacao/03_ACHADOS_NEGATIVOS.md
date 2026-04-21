# Achados Negativos Documentados

**Data:** 2026-04-19

Resultados negativos bem documentados têm valor científico. Eles previnem que outros pesquisadores repitam os mesmos caminhos sem sucesso e informam o design de abordagens futuras.

---

## 1. LLM Features de Segunda Ordem (v2) — PIORAM o Desempenho

### O que foi testado
8 features extraídas via LLM analisando estatísticas do dataset (approach `--llm-approach v2`): confidências por classe, consistency, pattern clarity, anomaly score, etc.

### Resultado
- **Dados sintéticos:** -20 a -26pp de accuracy ao adicionar features LLM v2
- **Dados reais:** V4 (Hierárquico + LLM no Level 2) → MNAR recall cai de **40% → 6%**

### Por que falhou

**Poder discriminativo inexistente.** Cohen's d entre MAR e MNAR para todas as 8 features:

| Feature | Cohen's d (MAR vs MNAR) | KW p-value | Veredicto |
|---------|:-----------------------:|:----------:|:---------:|
| llm_mar_conf | 0.390 | <0.0001 | Fraco |
| llm_mcar_conf | -0.260 | <0.0001 | Muito fraco |
| llm_mnar_conf | -0.204 | 0.004 | Muito fraco |
| llm_dist_shift | -0.184 | 0.090 | Não significativo |
| llm_evidence_consistency | -0.158 | 0.630 | Não significativo |
| llm_mcar_vs_mnar | 0.122 | 0.210 | Não significativo |
| llm_pattern_clarity | 0.050 | 0.527 | Não significativo |
| llm_anomaly | -0.003 | 0.101 | Não significativo |

**Distribuições idênticas entre classes.** `llm_mnar_conf` — a feature mais relevante:
- MCAR: mediana = **0.40**, IQR [0.30, 0.50]
- MAR: mediana = **0.40**, IQR [0.30, 0.49]
- MNAR: mediana = **0.40**, IQR [0.30, 0.54]

A LLM retorna o mesmo valor para todas as classes. Não discrimina.

**Alta multicolinearidade.** 6 pares com |r| > 0.5:
- `llm_evidence_consistency` × `llm_pattern_clarity`: r = 0.813
- `llm_mcar_conf` × `llm_mcar_vs_mnar`: r = -0.811

8 features nominais, mas ~3 dimensões de informação independente, todas com baixo poder discriminativo.

**Mecanismo de falha:**
1. LLM features adicionam 8 dimensões de ruído ao Level 2
2. `llm_mar_conf` é marginalmente mais alta para MAR (d=0.39) → modelo aprende viés
3. GradientBoosting classifica **94% como MAR** no Level 2 → MNAR recall = 6%
4. Modelos menos sensíveis a ruído (NaiveBayes, KNN) sofrem menos mas ainda são afetados

### Contraste com CAAFE

| Feature | Cohen's d (MAR vs MNAR) | KW p-value | Veredicto |
|---------|:-----------------------:|:----------:|:---------:|
| caafe_tail_asymmetry | **-0.840** | <0.0001 | **Forte** |
| caafe_cond_entropy_X0_mask | 0.388 | <0.0001 | Moderado |
| caafe_kurtosis_excess | -0.290 | <0.0001 | Moderado |

Features determinísticas (CAAFE) têm poder discriminativo **2-4x maior** que features LLM.

### Implicação
LLMs tendem a regredir para confidências médias quando incertos — comportamento prejudicial para classes difíceis como MNAR. Features especializadas e determinísticas superam LLM para este tipo de problema.

---

## 2. LLM Judge MNAR — Sem Melhoria

### O que foi testado
4 features via LLM atuando como "juiz" binário MCAR vs MNAR (approach `--llm-approach judge`): probabilidade MNAR, censoring, anomalia distribucional, structured pattern.

### Resultado
Sem melhoria significativa. Cohen's d < 0.4 para todas as 4 features.

### Por que falhou
O LLM não consegue distinguir MNAR de MCAR apenas analisando estatísticas resumo. A informação que define MNAR (dependência de X0 em si mesmo) é inerentemente não-observável a partir de resumos estatísticos de X1-X4.

---

## 3. Embeddings — Sem Melhoria

### O que foi testado
10 features via sentence-transformers local (approach `--llm-approach embeddings`): embeddings de descrições textuais dos padrões de missing data.

### Resultado
Nenhuma melhoria sobre baseline. Embeddings capturam semântica textual, não padrões estatísticos de missingness.

### Por que falhou
As descrições textuais dos padrões de missing são genéricas ("dados faltantes em X0") e não codificam informação discriminativa sobre o mecanismo.

---

## 4. CAAFE via LLM (Code Generation) — Resultado Misto

### O que foi testado
LLM gerando código Python para criar features (inspirado em CAAFE, NeurIPS 2023).

### Resultado
+1pp de melhoria apenas. As 4 features CAAFE implementadas manualmente (+7.1pp) superam amplamente a geração automática via LLM.

### Lição
Features determinísticas escritas com conhecimento de domínio > features geradas automaticamente por LLM para este problema específico.

---

## 5. Classificação Hierárquica com LLM (V4) — Piora MNAR

### O que foi testado
V4: Hierárquico (Level 1: MCAR vs {MAR,MNAR}, Level 2: MAR vs MNAR com LLM features)

### Resultado
| Variante | Accuracy | MNAR Recall | MNAR F1 |
|----------|:--------:|:-----------:|:-------:|
| V3 (Hier+CAAFE, sem LLM) | **50.5%** | **40.0%** | **0.488** |
| V4 (Hier+CAAFE+LLM) | 44.4% | **6.0%** | 0.396 |

### Mecanismo de falha
Mesmo que na seção 1: LLM features no Level 2 introduzem ruído que enviesam o classificador para MAR, destruindo a capacidade de detectar MNAR.

---

## 6. Data Leakage no Pipeline Original — Accuracy Inflada

### O que aconteceu
Pipeline original (v1) reportava **90.9% accuracy** em dados reais. Após auditoria:

| Fase | Accuracy (Real) | Data Leakage |
|------|:---------------:|:------------:|
| Pipeline original (v1) | 90.9% | **SIM** |
| Fases 1-2 | 98.7% | **SIM** |
| Após correção | 40.5% | NÃO |

### Causas identificadas
1. **Bootstrap leak:** Amostras do mesmo dataset em treino E teste (GroupShuffleSplit)
2. **Features fingerprint:** `X0_mean`, quantiles identificavam o dataset, não o mecanismo
3. **Features assumiam distribuição uniforme:** Quebravam em dados reais (exponencial, bimodal)

### Correção
- GroupKFold por dataset de origem
- Remoção de features fingerprint
- Features invariantes a distribuição (CAAFE, discriminativas)

### Lição
Sempre validar com leave-one-group-out quando amostras vêm de fontes agrupadas.

---

## 7. Labels Inconsistentes em Benchmarks — 57%

### O que foi descoberto
Validação estatística (Little's MCAR test + correlação point-biserial + KS test) revelou que **57% dos labels de mecanismo** nos 23 datasets reais são inconsistentes com os testes estatísticos.

### Exemplo
- **Oceanbuoys:** Rotulado como MCAR, mas Little's test p=0.000 e correlação mask-Xi = 0.333 → **MAR**

### Implicação
O teto teórico de accuracy neste benchmark é ~60-65%, não 100%. Com 57% de label noise, nenhum classificador pode ultrapassar muito este limite.

---

## Valor Científico dos Achados Negativos

Estes resultados negativos são valiosos porque:

1. **Contrariam a tendência "LLM para tudo"** — evidência empírica de que LLMs como feature extractors estatísticos NÃO funcionam para classificação de mecanismos de missing data
2. **Identificam o mecanismo de falha** — regressão à média nas confidências do LLM
3. **Direcionam pesquisa futura** — o valor do LLM está no domain reasoning, não na análise de dados
4. **Documentam armadilhas** — data leakage via bootstrap, features fingerprint, label noise
