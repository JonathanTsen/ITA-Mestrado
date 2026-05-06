# Seção de limitações — formulação para dissertação/artigo

**Data:** 2026-05-05  
**Contexto:** texto pronto para inserção na seção de limitações do artigo, baseado na decisão metodológica do doc 08 e nos resultados do protocolo v2.

---

## Frase metodológica recomendada (já aprovada em doc 08)

> "Os rótulos reais de mecanismo foram definidos por curadoria de literatura/domínio e usados como referência principal. O protocolo v2 foi aplicado como validação empírica complementar e quantificação de incerteza, não como substituto automático dos rótulos de domínio."

---

## Texto de limitações (português acadêmico)

### Limitação 1 — Não-identificabilidade do MNAR

O mecanismo MNAR depende, por definição, dos valores não observados da própria variável com dados faltantes. Isso implica que nenhum teste baseado exclusivamente nos dados observados pode distinguir MNAR de MCAR com certeza teórica (Mohan & Pearl, 2021). O protocolo v2 captura evidências indiretas (distribuição residual, entropia condicional, delta de previsibilidade), mas não resolve a ambiguidade fundamental. Os datasets identificados como ambíguos (confiança < 0,4) devem ser reportados como tais, não reclassificados arbitrariamente.

### Limitação 2 — Acurácia do diagnóstico automático

A acurácia do classificador Bayesiano (KDE 10-dimensional), avaliada via *cross-validation* estratificada 5-fold em 300 datasets sintéticos com ground truth, foi de **59,0% ± 6,0%**. A confusão dominante ocorre entre MCAR e MNAR (46–49% dos erros), que são os casos de menor separação no espaço de features. Essa acurácia é suficiente para evidência auxiliar, mas inadequada para substituição automática dos rótulos de domínio.

### Limitação 3 — Gap entre sintéticos e dados reais

O classificador foi calibrado em datasets sintéticos gerados por missmecha (MCAR/MAR/MNAR puros). Os dados reais apresentam: (a) mecanismos mistos ou parcialmente MNAR; (b) distribuições não-gaussianas; (c) taxas de dados faltantes distintas; (d) rótulos de literatura que podem eles próprios conter incerteza. Esse *distribution shift* explica parte da queda de acurácia nos reais (41,4% vs. 59% nos sintéticos).

### Limitação 4 — Scores CAAFE-MNAR como proxies

Os quatro scores da Camada C (auc_self_delta, kl_density, kurtosis, conditional entropy) são medidas indiretas de dependência entre X0 e sua máscara de ausência. Individualmente, nenhum tem poder discriminativo suficiente (AUC individual 0,5–0,7). O classificador Bayesiano compensa pela combinação multivariada de todos os 10 scores, mas permanece limitado pelo problema circular: MNAR é detectado via X0 imputado, que é aproximado pela mediana dos observados.

### Limitação 5 — Sensibilidade ao prior

O diagnóstico Bayesiano usa prior uniforme (P(MCAR) = P(MAR) = P(MNAR) = 1/3) como default. A análise de sensibilidade com prior informativo (P(MNAR) = 0,2 a 0,5) pode alterar as predições especialmente nos casos ambíguos. Essa sensibilidade deve ser reportada como análise complementar, não como resultado principal.

### Limitação 6 — Escassez de MCAR confirmado na literatura

Uma pesquisa exaustiva em bases de dados acadêmicas e repositórios públicos (UCI, OpenML, CRAN, Zenodo) revelou que **não existem datasets reais amplamente disponíveis com evidência estatística forte de MCAR**. As únicas fontes com citação publicada são: (a) `mice::boys` (Van Buuren, 2018, FIMD Ch. 9 — gaps aleatórios de agendamento clínico) e (b) `mice::brandsma` (ausência escolar — correlação mask~covariáveis com p > 0.15). O único MCAR *garantido* é por *planned missingness designs* (Graham et al., 2006), onde a aleatorização é imposta pelo desenho do estudo. Essa escassez é consistente com a posição teórica de que MCAR puro é uma idealização (Rubin, 1976; Van Buuren, 2018, Cap. 2) e explica parcialmente por que o protocolo v2 não consegue confirmar MCAR nos datasets existentes.

---

## Tabela de incerteza por dataset

Resultados v2b (29 datasets originais, `--prior-mnar 0.35`, `--n-permutations 100`). Tabelas completas em `data/real/sources.md`.

**Acurácia global: 31 % (9/29).** Distribuição de predições: 6 MCAR, 15 MAR, 8 MNAR.

**Concordâncias (9/29):** 8 MAR corretos (airquality, mammographic, oceanbuoys×2, sick×2, titanic×2) + 1 MNAR correto (colic_refluxph). **0 MCAR corretos.**

**Ambíguos (conf < 0.40):** echomonths_epss, hepatitis_albumin, kidney_hemo, hepatitis_protime, mroz_wages — candidatos a sensitivity analysis.

**Benchmark expandido (2026-05-05):** 4 novos MCAR adicionados (boys_hc, boys_hgt, brandsma_lpr, brandsma_apr), totalizando 33 datasets (13 MCAR, 11 MAR, 9 MNAR). Re-validação pendente.

---

## Referências para a seção de limitações

- Mohan, K., & Pearl, J. (2021). Graphical models for processing missing data. *JASA*.
- Sterne, J. A. C., et al. (2009). Multiple imputation for missing data in epidemiological and clinical research. *BMJ*.
- Van Buuren, S. (2018). *Flexible Imputation of Missing Data*. CRC Press. (Cap. 2: mecanismos; Cap. 9: boys)
- Graham, J. W., Taylor, B. J., Olchowski, A. E., & Cumsille, P. E. (2006). Planned missing data designs in psychological research. *Psychological Methods*, 11(4), 323–343.
- Brandsma, H. P. & Knuver, J. W. M. (1989). Effects of school and classroom characteristics on pupil progress. *Int. J. Educ. Res.*, 13, 777–788.
- Fredriks, A. M., van Buuren, S., et al. (2000). Continuing positive secular growth change in the Netherlands 1955–1997. *Pediatric Research*, 47, 316–323.
