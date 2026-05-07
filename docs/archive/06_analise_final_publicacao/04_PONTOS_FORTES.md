# Pontos Fortes da Pesquisa

**Data:** 2026-04-19

O que torna este trabalho publicável e relevante para a comunidade científica.

---

## A) Rigor Metodológico Excepcional

### Descoberta e correção de data leakage
- Accuracy caiu de **90.9% → 40.5%** após correção — reportado honestamente
- Identificação de 3 fontes de leakage: bootstrap overlap, features fingerprint, distribuição assumida
- Poucos trabalhos na literatura reportam este tipo de auto-correção

### Validação robusta
- **GroupKFold-5:** Garante que nenhum bootstrap do mesmo dataset aparece em treino e teste
- **LODO (Leave-One-Dataset-Out):** 23 folds, cada dataset é teste uma vez
- **Bootstrap CIs (1000 iterações):** Intervalos de confiança para todos os cenários
- GroupKFold ≈ LODO (diferença < 2pp) confirma ausência de memorização

### Auditoria forense de metadata
- 5 canais de leakage identificados e documentados:
  - A: `source` cita datasets canônicos (Mroz, Pima, airquality)
  - B: `domain` + `x0_variable` permite inferência (mantido — é input legítimo)
  - C: `predictors.role` contém linguagem MAR ("proxy")
  - D: `x0_typical_range` re-injeta sinal clínico
  - E: constância per-dataset (estrutural, inevitável com bootstrap)
- Canais A, C, D fechados; B mantido intencionalmente; E documentado

---

## B) Contribuição Genuína em LLM Domain Reasoning

### O achado principal
- `domain_prior` com metadata **neutralizada** → 63.1% accuracy (vs 33.3% chance)
- **+22.6pp** sobre baseline puramente estatístico
- O LLM infere mecanismos a partir de **domínio + nome da variável** — sem informação sobre o mecanismo real

### Por que é genuíno
1. Metadata foi neutralizada (sem nomes canônicos, sem cutoffs clínicos)
2. GroupKFold-5 ≈ LODO (0.0pp de diferença para domain_prior sozinho)
3. O LLM replica raciocínio que um estatístico faria:
   - "Insulina em estudo de diabetes" → provavelmente MNAR (pacientes sem diabetes não fazem o exame)
   - "Idade no Titanic" → provavelmente MAR (depende de classe do bilhete)
   - "Pressão em dados industriais" → pode ser MCAR (sensor falhou aleatoriamente)

### Novidade na literatura
Ninguém testou LLMs como **"domain experts"** para classificar mecanismos de missing data a partir de metadados de dataset. Esta é uma contribuição original e verificável.

---

## C) Resultado Negativo Bem Documentado

### O que foi mostrado
LLM features via análise estatística de segunda ordem **não melhoram** — e frequentemente **pioram** — a classificação.

### Evidência
- Distribuições idênticas entre classes (mediana=0.40 para MCAR, MAR, MNAR)
- Cohen's d < 0.4 para todas as 8 features
- Multicolinearidade alta (6 pares |r| > 0.5)
- Mecanismo de falha identificado e explicado

### Valor para a comunidade
Contraria a tendência crescente de usar LLMs como feature extractors para dados tabulares. Oferece evidência empírica rigorosa de **onde** o LLM agrega valor (domain reasoning) e **onde** não agrega (análise estatística de segunda ordem).

---

## D) Framework Conceitual: Dois Regimes de Accuracy

### Regime Estatístico (40-51%)
- Features baseadas em padrões nos dados
- Teto: ~51% (saturação com features estatísticas + LLM data-driven)
- CAAFE contribui +7.1pp — maior componente individual

### Regime de Domain Reasoning (56-63%)
- LLM usa domínio + variável para inferir mecanismos
- Adiciona +5.7pp sobre regime estatístico
- domain_prior sozinho: 63.1%

### Por que é valioso
Separa claramente duas fontes de informação e seus tetos, guiando pesquisa futura. O regime estatístico tem limite teórico; domain reasoning oferece caminho para superar este limite.

---

## E) CAAFE Features como Contribuição Técnica

> **Nota de atualização:** nesta fase, "CAAFE" significa features
> CAAFE-inspired determinísticas para MNAR, não a reimplementação do CAAFE
> original com LLM gerando código. A lista abaixo é histórica. A versão v2b
> atual usa `caafe_auc_self_delta`, `caafe_kl_density`,
> `caafe_kurtosis_excess` e `caafe_cond_entropy_X0_mask`.

### As 4 features
1. `caafe_tail_asymmetry` — assimetria de cauda (MNAR afeta caudas)
2. `caafe_kurtosis_excess` — excesso de curtose (MNAR distorce distribuição)
3. `caafe_cond_entropy_X0_mask` — entropia condicional mask dado X0
4. `caafe_missing_rate_by_quantile` — taxa de missing por quantil (MNAR é não-uniforme)

### Impacto
- **+7.1pp** — maior contribuição individual do regime estatístico
- **28.3%** da feature importance com apenas 16% das features (4/25)
- Implementadas como **funções puras Python** (determinísticas, sem API)
- Inspiradas em CAAFE (NeurIPS 2023) mas adaptadas para detecção de mecanismos

### Contraste com LLM features
CAAFE (determinístico): Cohen's d = **0.84** (tail_asymmetry)
LLM (melhor feature): Cohen's d = **0.39** (llm_mar_conf)

Features determinísticas especializadas superam LLM features em **2x** de poder discriminativo.

---

## F) Comparação Justa com Baselines

### MechDetect (Jung et al., 2024) e PKLM (Spohn et al., 2024) implementados
Ambos os baselines foram implementados e avaliados **nos mesmos 23 datasets**, com os mesmos splits.

### Vieses expostos
Cada baseline tem viés sistemático para uma classe:
- **PKLM:** Viés para MCAR (85.7% das predições são MCAR)
- **MechDetect:** Viés para MNAR (84.2% das predições são MNAR)
- **V3 proposto:** Único com recall **equilibrado** (MCAR 47%, MAR 56%, MNAR 40%)

### Superioridade demonstrada
V3 alcança o **melhor F1 macro** (0.488) entre todos os métodos testados, incluindo MechDetect com thresholds otimizados via CV (0.472).

---

## G) Auditoria de Labels em Benchmarks

### Descoberta
57% dos labels de mecanismo nos 23 datasets reais são **inconsistentes** com testes estatísticos (Little's MCAR, correlação point-biserial, KS test).

### Exemplo concreto
`oceanbuoys`: Rotulado MCAR na literatura, mas Little's p=0.000 e correlação mask-Xi=0.333 → claramente MAR.

### Contribuição independente
Esta auditoria é valiosa independentemente dos resultados de classificação. Ela questiona benchmarks usados por toda a comunidade e sugere a necessidade de validação estatística antes de usar labels de mecanismo como ground truth.
