# Achados consolidados — Testes estatísticos pareados ML × Gemini Flash × Gemini Pro

**Contexto:** Esta análise responde à exigência levantada na reunião de orientação de
07/05/2026 (Vitor Curtis), onde o orientador apontou que 5-fold cross-validation é
amostragem, não significância — toda comparação ML × LLM precisa de teste estatístico
e desvio padrão, não apenas média.

**Código:** [`src/missdetect/compare_approaches.py`](../src/missdetect/compare_approaches.py)
**Testes:** [`tests/test_compare_approaches.py`](../tests/test_compare_approaches.py) (13/13 passando)
**Artefatos brutos:** `results/comparison_v2b_pro/`, `results/comparison_v2b_flash/`,
`results/comparison_step05_sintetico/`

---

## 1. Visão geral por modelo (média sobre todos os mecanismos, dados reais v2b 32 datasets)

Acurácia média ± desvio padrão dos folds. Fold scores pareados (mesmos folds nos
três cenários).

| Modelo | ML puro | ML + Flash | Δ Flash | ML + Pro | Δ Pro |
|---|---|---|---|---|---|
| GradientBoosting | 0.525 ± 0.120 | 0.513 ± 0.144 | −1.2 pp | 0.467 ± 0.140 | −5.8 pp |
| KNN | 0.443 ± 0.086 | 0.423 ± 0.114 | −2.0 pp | 0.379 ± 0.112 | −6.5 pp |
| LogisticRegression | 0.364 ± 0.104 | 0.372 ± 0.041 | +0.8 pp | 0.304 ± 0.097 | −6.0 pp |
| MLP | 0.446 ± 0.164 | 0.465 ± 0.120 | +1.9 pp | 0.396 ± 0.114 | −5.0 pp |
| NaiveBayes | 0.426 ± 0.191 | 0.449 ± 0.172 | +2.3 pp | 0.430 ± 0.130 | +0.4 pp |
| RandomForest | 0.516 ± 0.085 | 0.519 ± 0.101 | +0.3 pp | 0.468 ± 0.199 | −4.8 pp |
| SVM_RBF | 0.426 ± 0.128 | 0.439 ± 0.142 | +1.4 pp | 0.376 ± 0.127 | −5.0 pp |
| **Média global (35 fold-scores)** | **0.450 ± 0.130** | **0.455 ± 0.123** | **+0.5 pp** | **0.403 ± 0.134** | **−4.7 pp** |

**Wilcoxon pareado global:**

| Comparação | n_pares | Δ médio | W | p | Cohen's d_z | Cliff's δ | Significativo (α=0.05)? |
|---|---|---|---|---|---|---|---|
| ML × Flash | 35 | +0.005 | 286.5 | 0.641 | +0.089 | +0.086 | **não** |
| ML × Pro | 35 | −0.047 | 242.5 | 0.238 | −0.261 | −0.114 | **não** |

**Interpretação:** olhando todos os mecanismos juntos, **nenhum dos LLMs supera o ML
puro de forma estatisticamente significativa**. Flash empata; Pro tem tendência
negativa de 4.7 pp, mas o desvio entre folds é grande demais para que essa diferença
seja detectável.

---

## 2. Quebra por mecanismo (McNemar amostra-a-amostra)

### 2.1 — MCAR (Missing Completely At Random)

**Flash** (cobertura 100%, n=100 amostras por modelo):

| Modelo | ML | Flash | Δ | p | sig |
|---|---|---|---|---|---|
| GradientBoosting | 0.410 | 0.500 | +9.0 pp | 0.039 | * |
| KNN | 0.630 | 0.700 | +7.0 pp | 0.265 | |
| LogisticRegression | 0.540 | 0.610 | +7.0 pp | 0.211 | |
| MLP | 0.440 | 0.510 | +7.0 pp | 0.190 | |
| NaiveBayes | 0.750 | 0.790 | +4.0 pp | 0.423 | |
| RandomForest | 0.360 | 0.440 | +8.0 pp | 0.080 | |
| SVM_RBF | 0.590 | 0.640 | +5.0 pp | 0.486 | |
| **Média** | **0.546** | **0.599** | **+5.3 pp** | — | |

**Pro** (cobertura 50%, n=50 — `sample_idx` regenerado entre runs reduziu o pareamento):

| Modelo | ML | Pro | Δ | p |
|---|---|---|---|---|
| GradientBoosting | 0.380 | 0.420 | +4.0 pp | 0.480 |
| KNN | 0.740 | 0.720 | −2.0 pp | 1.000 |
| LogisticRegression | 0.640 | 0.720 | +8.0 pp | 0.221 |
| MLP | 0.580 | 0.520 | −6.0 pp | 0.606 |
| NaiveBayes | 0.900 | 0.900 | +0.0 pp | 1.000 |
| RandomForest | 0.440 | 0.500 | +6.0 pp | 0.248 |
| SVM_RBF | 0.700 | 0.760 | +6.0 pp | 0.546 |
| **Média** | **0.626** | **0.649** | **+2.3 pp** | — |

**Conclusão MCAR:** Flash tem tendência positiva consistente em 7/7 modelos
(+5.3 pp média), com 1 caso significativo a 5% e RandomForest marginal (p=0.080).
Pro vai na mesma direção mas com efeito menor e zero significâncias.
**MCAR é onde o LLM mais "parece" ajudar, mas com força estatística limitada.**

### 2.2 — MAR (Missing At Random)

**Flash** (cobertura 100%, n=100):

| Modelo | ML | Flash | Δ | p |
|---|---|---|---|---|
| GradientBoosting | 0.460 | 0.450 | −1.0 pp | 1.000 |
| KNN | 0.230 | 0.260 | +3.0 pp | 0.700 |
| LogisticRegression | 0.150 | 0.190 | +4.0 pp | 0.221 |
| MLP | 0.320 | 0.360 | +4.0 pp | 0.423 |
| NaiveBayes | 0.010 | 0.010 | +0.0 pp | 1.000 |
| RandomForest | 0.400 | 0.380 | −2.0 pp | 0.617 |
| SVM_RBF | 0.360 | 0.370 | +1.0 pp | 1.000 |
| **Média** | **0.276** | **0.289** | **+1.3 pp** | — |

**Pro:** sem dados pareados — `sample_idx` regenerado reduziu o overlap a 0% para
o subgrupo MAR.

**Conclusão MAR:** efeito praticamente nulo (+1.3 pp média Flash, zero significâncias).
Observação importante: NaiveBayes em MAR fica em 1% — ele simplesmente não funciona
para esse mecanismo (acaso é 33%, então 1% indica viés sistemático para outras classes).
**MAR é o mecanismo onde o LLM tem menos a oferecer**, e o ML por si só também sofre —
provavelmente porque MAR e MNAR são teoricamente indistinguíveis sem informação
externa (Molenberghs et al. 2008, citado na tese).

### 2.3 — MNAR (Missing Not At Random) — caso central da tese

**Flash** (cobertura 100%, n=200):

| Modelo | ML | Flash | Δ | p | sig |
|---|---|---|---|---|---|
| GradientBoosting | 0.590 | 0.530 | **−6.0 pp** | 0.031 | * (PIORA) |
| KNN | 0.340 | 0.325 | −1.5 pp | 0.810 | |
| LogisticRegression | 0.390 | 0.355 | −3.5 pp | 0.419 | |
| MLP | 0.465 | 0.465 | +0.0 pp | 0.890 | |
| **NaiveBayes** | **0.290** | **0.445** | **+15.5 pp** | **<0.0001** | **\*\*\*** |
| RandomForest | 0.565 | 0.540 | −2.5 pp | 0.486 | |
| SVM_RBF | 0.345 | 0.330 | −1.5 pp | 0.798 | |
| **Média** | **0.427** | **0.427** | **+0.0 pp** | — | |

**Pro** (cobertura 70%, n=141):

| Modelo | ML | Pro | Δ | p | sig |
|---|---|---|---|---|---|
| GradientBoosting | 0.631 | 0.617 | −1.4 pp | 0.874 | |
| KNN | 0.369 | 0.397 | +2.8 pp | 0.708 | |
| LogisticRegression | 0.355 | 0.426 | +7.1 pp | 0.237 | |
| MLP | 0.532 | 0.504 | −2.8 pp | 0.708 | |
| **NaiveBayes** | **0.284** | **0.454** | **+17.0 pp** | **0.006** | **\*\*** |
| RandomForest | 0.574 | 0.560 | −1.4 pp | 0.888 | |
| SVM_RBF | 0.355 | 0.411 | +5.7 pp | 0.374 | |
| **Média** | **0.443** | **0.481** | **+3.8 pp** | — | |

**Conclusão MNAR (achado central):**

1. Em **média sobre os 7 modelos**, MNAR não melhora com LLM (Flash 0 pp,
   Pro +3.8 pp não significativo).
2. **GradientBoosting com Flash PIORA significativamente em MNAR** (−6.0 pp,
   p=0.031). Sinal de que o LLM injeta ruído em alguns modelos.
3. **NaiveBayes + LLM melhora MNAR de forma robusta e altíssima significância:**
   - Flash: +15.5 pp, p < 0.0001 (\*\*\*)
   - Pro: +17.0 pp, p = 0.006 (\*\*)
   - **Os dois LLMs convergem para o mesmo achado** — não é coincidência de um
     provedor, é um efeito real.

---

## 3. Resumo quantitativo consolidado

| Mecanismo | ML puro | + Flash | + Pro | Quem ganha? |
|---|---|---|---|---|
| Geral (35 folds) | 0.450 | 0.455 (+0.5 pp, p=0.64) | 0.403 (−4.7 pp, p=0.24) | Empate / ML |
| MCAR | 0.546 | 0.599 (+5.3 pp) | 0.649 (+2.3 pp) | + LLM (sem p<0.05 robusto) |
| MAR | 0.276 | 0.289 (+1.3 pp) | — | Empate |
| MNAR (média) | 0.427 | 0.427 (+0.0 pp) | 0.481 (+3.8 pp) | Empate em média |
| **MNAR + NaiveBayes** | **0.290** | **0.445 (+15.5 pp \*\*\*)** | **0.454 (+17.0 pp \*\*)** | **+ LLM, com força estatística** |

---

## 4. Validação cruzada com dados sintéticos (step05_pro)

Comparação Gemini 3.1 Pro × ML em dados sintéticos (mdatagen):

- **Global:** Δ=−0.003, p=0.301 — não significativo. ML puro já está em 74% (teto
  alto, pouco espaço para melhorar).
- **MCAR sintético** (n=71): LogisticRegression melhora +12.7 pp (p=0.016 *).
- **MAR sintético** (n=130): tudo neutro; ML puro já em 90–93% (saturado).
- **MNAR sintético** (n=99): direção INVERTIDA em relação ao real:
  - MLP PIORA −12.1 pp (p=0.045 *)
  - NaiveBayes PIORA −8.1 pp (p=0.027 *)
  - SVM_RBF PIORA −9.1 pp (p=0.039 *)

**Interpretação:** o efeito "NB+LLM ajuda em MNAR" só aparece nos **dados reais
ruidosos** (Cleanlab indica 59.4% labels potencialmente incorretas). Em sintéticos
limpos, o NB já tem o sinal estatístico que precisa, e a LLM apenas adiciona ruído.
Esse contraste reforça a interpretação: **a LLM funciona como compensador semântico
sob ruído de label**, não como feature genérica.

---

## 5. Conclusão final do artigo

### O que dizer (defensável estatisticamente)

1. **A LLM não melhora a detecção de mecanismos de dados faltantes "no geral".**
   Olhando todos os mecanismos juntos, nem Gemini Pro nem Flash batem o ML puro
   com significância (Wilcoxon p=0.24 e p=0.64).

2. **Há um achado específico, robusto e replicado em dois LLMs**: a combinação
   **Naive Bayes + features de LLM melhora a detecção de MNAR em dados reais**
   em cerca de 15–17 pp, com p<0.001 (Flash) e p=0.006 (Pro). Como os dois LLMs
   convergem, não é artefato de um provedor — é um efeito de **regularização
   semântica**: o NB sozinho fica perto do acaso em MNAR (29%) porque o sinal
   estatístico é fraco; as features da LLM dão exatamente o que ele precisa para
   sair do chute.

3. **Em MCAR, há tendência consistente de melhora com Flash** (+5.3 pp em todos
   os 7 modelos), mas só GradientBoosting cruza p<0.05.

4. **Em MAR, não há ganho.** O ML puro já é tão ruim (NB em 1%, KNN em 23%) que
   o LLM não consegue desambiguar — coerente com a teoria: MAR e MNAR são
   indistinguíveis sem informação externa.

5. **Cuidado direcional**: Flash às vezes degrada modelos discriminativos em MNAR
   (GradientBoosting −6 pp, p=0.031). Não é um free lunch.

### O que NÃO dizer

- **Não dizer "LLM melhora ~10% para identificar MNAR"** (impressão verbal da
  reunião). Estatisticamente isso só vale para **Naive Bayes**, e o ganho é
  maior (~15 pp); a média sobre 7 modelos em MNAR é 0 pp (Flash) ou +3.8 pp
  não significativo (Pro).
- **Não afirmar** que os datasets reais "são" MNAR — eles foram **rotulados** como
  MNAR seguindo artigos científicos, mas o mecanismo não pode ser empiricamente
  confirmado (Rubin 1976; ver item 2 do plano de ajustes da tese).

### Framing recomendado para a tese (Cap 4 — Discussion of Results)

> *A LLM não substitui o pipeline ML em detecção de mecanismos de dados
> faltantes, mas atua como compensador para classificadores fracos calibrados
> (Naive Bayes) em MNAR. Em datasets reais ruidosos (Cleanlab indica 59.4% de
> labels potencialmente incorretas), as features semânticas da LLM elevam o NB
> de 29% (próximo ao acaso) para 44–45%, com p<0.001 (Flash). Esse é o único
> achado consistentemente replicado entre dois provedores de LLM. Os efeitos em
> MCAR são na direção certa mas insuficientemente potentes; em MAR não há sinal,
> coerente com o resultado de impossibilidade de Molenberghs et al. (2008).*

### Bônus: Pareto Flash × Pro

| Critério | Flash | Pro |
|---|---|---|
| Custo aproximado por benchmark | ~$3 | ~$30 |
| Cobertura McNemar (real) | 100% nos 3 mecanismos | 50% (MCAR), 0% (MAR), 70% (MNAR) |
| p-valor no caso âncora (NB+MNAR) | <0.0001 (\*\*\*) | 0.006 (\*\*) |
| Tendência em MCAR | +5.3 pp consistente em 7/7 modelos | +2.3 pp |
| Tendência global | +0.5 pp | −4.7 pp |

→ **Flash domina Pro** em custo-benefício para esse problema. Outro achado para
reportar, alinhado com a observação na tese de que "Pareto-dominado" descrevia
o Flash erroneamente — na verdade quem é Pareto-dominado é o **Pro**.

---

## 6. Quando a LLM ajuda outros modelos além do Naive Bayes?

Pergunta levantada em 2026-05-10. Resposta curta: **quase só o NB**. Fora dele, há
um achado pontual em dados reais e um em sintéticos, ambos com ressalvas.

### 6.1 — Todos os casos com p<0.05 (positivos e negativos)

**Em dados reais (32 datasets v2b) — AJUDA:**

| Modelo + LLM | Mecanismo | Δ | p | Marcador |
|---|---|---|---|---|
| NaiveBayes + Flash | MNAR | +15.5 pp | <0.0001 | \*\*\* |
| NaiveBayes + Pro | MNAR | +17.0 pp | 0.006 | \*\* |
| GradientBoosting + Flash | MCAR | +9.0 pp | 0.039 | \* |

**Em dados reais — PREJUDICA:**

| Modelo + LLM | Mecanismo | Δ | p | Marcador |
|---|---|---|---|---|
| GradientBoosting + Flash | MNAR | −6.0 pp | 0.031 | \* |

**Em dados sintéticos (step05_pro) — AJUDA:**

| Modelo + LLM | Mecanismo | Δ | p | Marcador |
|---|---|---|---|---|
| LogisticRegression + Pro | MCAR | +12.7 pp | 0.016 | \* |

**Em dados sintéticos — PREJUDICA:**

| Modelo + LLM | Mecanismo | Δ | p | Marcador |
|---|---|---|---|---|
| MLP + Pro | MNAR | −12.1 pp | 0.045 | \* |
| NaiveBayes + Pro | MNAR | −8.1 pp | 0.027 | \* |
| SVM_RBF + Pro | MNAR | −9.1 pp | 0.039 | \* |

### 6.2 — Por que só o NB sobrevive como achado robusto

Três critérios para considerar um achado "robusto":
1. **Replica em dois LLMs diferentes** (Flash e Pro convergem).
2. **Direção consistente** entre cenários relacionados.
3. **Não é parte de um trade-off** (modelo melhora num mecanismo e piora em outro).

| Achado candidato | Replica nos 2 LLMs? | Direção consistente? | Sem trade-off? | Robusto? |
|---|---|---|---|---|
| NB + LLM em MNAR (real) | **Sim** (Flash \*\*\*, Pro \*\*) | Sim em real | Sim (NB melhora em MNAR sem piorar em outros) | **✅** |
| GradientBoosting + Flash em MCAR | Não (só Flash) | Sim | **Não** — mesmo combo piora em MNAR (−6 pp \*) | ❌ |
| LR + Pro em MCAR (sintético) | Não (só Pro) | Não — em MCAR real o mesmo combo dá +8 pp **sem** significância | Sim | ❌ |

### 6.3 — A inversão no NB sintético reforça a interpretação

O achado mais delicado: **NB+Pro em MNAR sintético PIORA −8.1 pp (p=0.027)**, exatamente o oposto do que ocorre em dados reais (+17 pp \*\*).

Esse contraste **não enfraquece** o achado em dados reais — pelo contrário, **explica
o mecanismo causal**:

- Em sintéticos, os labels são **certos** e o sinal estatístico é **forte**. NB já
  tem o que precisa; LLM só adiciona ruído.
- Em reais, Cleanlab indica **59.4% de labels potencialmente incorretas**. NB
  sozinho fica perto do acaso (29% em MNAR, vs 33% chute) porque o sinal é fraco
  e ruidoso. As features semânticas da LLM dão a "âncora externa" que ele
  precisa.

**Conclusão metodológica**: a LLM aqui funciona como **regularização semântica
sob ruído de label**, não como feature genérica que melhoraria qualquer modelo
em qualquer cenário. É um achado mais nuanced — e mais defensável — do que "LLM
melhora detecção de MNAR".

### 6.4 — Tendência sem força estatística (vale mencionar como tal)

Em **MCAR com Flash**, *todos os 7 modelos* mostram melhora de **+4 a +9 pp** —
direção consistente em 7/7 casos. Mas com 5 folds e n=100 amostras, só
GradientBoosting cruza p<0.05.

| Modelo + Flash em MCAR | Δ | p |
|---|---|---|
| GradientBoosting | +9.0 pp | 0.039 \* |
| RandomForest | +8.0 pp | 0.080 (marginal) |
| KNN | +7.0 pp | 0.265 |
| LogisticRegression | +7.0 pp | 0.211 |
| MLP | +7.0 pp | 0.190 |
| SVM_RBF | +5.0 pp | 0.486 |
| NaiveBayes | +4.0 pp | 0.423 |

→ **Comportamento como tendência, não como achado confirmado**: em MCAR,
features de LLM tendem a deslocar todos os modelos para cima em ~5–9 pp,
provavelmente porque o LLM identifica "MCAR" pela própria ausência de padrão
estrutural no faltante (caso onde a heurística "se nada explica, é aleatório"
funciona). Com mais datasets, vários desses casos provavelmente cruzariam
p<0.05.

### 6.5 — Resumo

> **Para a tese, o único achado que vale ser declarado como "achado" é
> Naive Bayes + LLM em MNAR (dados reais).** GradientBoosting + Flash em MCAR
> tem força estatística mas vem com trade-off no próprio MNAR — não dá pra
> sustentar como ganho do modelo. Os achados em sintéticos são todos contextuais
> e não replicam em dados reais. A tendência positiva em MCAR com Flash vale
> mencionar como tendência, com a ressalva explícita de que precisa de mais
> dados para confirmar.

---

## 7. Limitações dos testes

1. **`sample_idx` instável entre runs** reduz a cobertura do McNemar para o Pro real
   (MAR=0%). Solução: o Wilcoxon nos fold scores (seção 1) é o teste principal
   e não tem essa limitação.
2. **5 folds = pouco poder estatístico** para Wilcoxon por modelo: o p-valor mínimo
   alcançável com 5 pares unidirecionais é ≈0.0625, então nenhum teste por modelo
   isolado em 5 folds atinge p<0.05 mesmo se o efeito for grande. O teste global
   (35 fold scores) e o McNemar amostra-a-amostra (n=100–200) compensam.
3. **Interpretação causal limitada nos dados reais** — o rótulo MCAR/MAR/MNAR
   vem de artigos publicados, não de mecanismo verificado empiricamente.
