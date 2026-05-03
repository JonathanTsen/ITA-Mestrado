# Investigação: V4 MNAR Recall = 6%

**Data:** 2026-04-18
**Contexto:** V4 (Hier+CAAFE+LLM no N2) tem accuracy 44.4% mas MNAR recall de apenas 6%, vs V3 (Hier+CAAFE N2) com 50.5% accuracy e 40% MNAR recall.

---

## Diagnóstico: LLM features não discriminam MAR vs MNAR

### Análise estatística: poder discriminativo MAR vs MNAR

| Feature | Cohen's d | KW p-value | Média MAR | Média MNAR | Veredicto |
|---------|:---------:|:----------:|:---------:|:----------:|:---------:|
| **llm_mar_conf** | **0.390** | **<0.0001** | 0.425 | 0.334 | Fraco |
| llm_mcar_conf | -0.260 | <0.0001 | 0.189 | 0.243 | Muito fraco |
| llm_mnar_conf | -0.204 | 0.004 | 0.386 | 0.423 | Muito fraco |
| llm_dist_shift | -0.184 | 0.090 | 0.281 | 0.324 | NS |
| llm_evidence_consistency | -0.158 | 0.630 | 0.300 | 0.338 | NS |
| llm_mcar_vs_mnar | 0.122 | 0.210 | 0.688 | 0.661 | NS |
| llm_pattern_clarity | 0.050 | 0.527 | 0.434 | 0.424 | NS |
| llm_anomaly | -0.003 | 0.101 | 0.768 | 0.769 | NS |

**Contraste com CAAFE features:**

| Feature | Cohen's d | KW p-value | Média MAR | Média MNAR | Veredicto |
|---------|:---------:|:----------:|:---------:|:----------:|:---------:|
| **caafe_tail_asymmetry** | **-0.840** | **<0.0001** | 0.059 | 0.280 | **Forte** |
| **caafe_kurtosis_excess** | **-0.290** | **<0.0001** | 5.788 | 10.931 | Moderado |
| **caafe_cond_entropy_X0_mask** | **0.388** | **<0.0001** | 0.235 | 0.183 | Moderado |

### Causa raiz: LLM retorna valores quase idênticos para todas as classes

`llm_mnar_conf` (a feature chave para identificar MNAR):
- MCAR: mediana = 0.40, IQR [0.30, 0.50]
- MAR: mediana = 0.40, IQR [0.30, 0.49]
- MNAR: mediana = 0.40, IQR [0.30, 0.54]

**A LLM retorna ~0.40 de confiança MNAR para TODAS as classes.** Não discrimina.

`llm_mcar_vs_mnar` (feature de raciocínio MCAR vs MNAR):
- MCAR: mediana = 0.70
- MAR: mediana = 0.75
- MNAR: mediana = 0.75

**Não discrimina** — MNAR deveria ter valor mais alto, mas é quase igual a MAR.

### Multicolinearidade excessiva entre LLM features

Pares com |r| > 0.5:
- `llm_mcar_conf` × `llm_mcar_vs_mnar`: r = -0.811
- `llm_evidence_consistency` × `llm_pattern_clarity`: r = 0.813
- `llm_mcar_conf` × `llm_mar_conf`: r = -0.677
- `llm_nnar_conf` × `llm_mcar_vs_mnar`: r = 0.637
- `llm_evidence_consistency` × `llm_anomaly`: r = -0.575
- `llm_anomaly` × `llm_pattern_clarity`: r = -0.604
- `llm_mar_conf` × `llm_mnar_conf`: r = -0.513

**6 de 8 features são redundantes entre si.** Isso adiciona ruído sem informação nova.

---

## Mecanismo de falha no V4

1. **LLM features adicionam 8 dimensões de ruído** ao Level 2 (que já tem 25 features úteis de stat+CAAFE)
2. O modelo (GradientBoosting) tenta usar as LLM features e **aprende correlações espúrias**
3. Em particular, `llm_mar_conf` é marginalmente mais alta para MAR — o modelo usa isso
4. Resultado: o modelo **classifica quase tudo como MAR** no Level 2, matando o MNAR recall
5. V3 (sem LLM) não tem esse problema porque as 4 CAAFE features são genuinamente discriminativas

### Evidência: viés sistemático para MAR

| Modelo | V4 MAR Recall | V4 MNAR Recall | Viés |
|--------|:------------:|:--------------:|:----:|
| RandomForest | — | **2%** | Extremo MAR |
| GradientBoosting | 55.3% | **6%** | Forte MAR |
| LogisticRegression | — | **4%** | Extremo MAR |
| SVM_RBF | — | **8%** | Forte MAR |
| NaiveBayes | — | **30%** | Menor (menos sensível a ruído) |
| KNN | — | **26%** | Menor |
| MLP | — | **18%** | Moderado |

NaiveBayes e KNN sofrem menos porque são menos propensos a overfitting em features ruidosas.

---

## Conclusão para o paper

1. **LLM v2 features não ajudam no Level 2** — não discriminam MAR vs MNAR em dados reais
2. **CAAFE features são superiores** — `caafe_tail_asymmetry` (d=0.84) captura a assimetria distribucional que é marca registrada do MNAR
3. **LLM adiciona ruído que dilui as CAAFE features** — V4 é pior que V3 em todos os aspectos
4. **A melhor estratégia é V3**: stat no Level 1, stat+CAAFE no Level 2, sem LLM
5. **Isso é um resultado negativo importante**: LLM-augmented features para classificação de mecanismos de missing data não melhoram além de features estatísticas determinísticas especializadas (CAAFE)
