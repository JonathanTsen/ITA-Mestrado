# Diagnóstico de Memorização — Flash MCAR +9pp (Fase 12)

**Data:** 2026-05-06
**Experimento:** `step13_flash_anonymous_mcar_only`
**Motivação:** O Flash melhorou o recall MCAR em +9pp (41% → 50%) comparado ao ML-only.
A hipótese era que o LLM reconhecia datasets famosos (`boys`, `brandsma`) de sua memória
de pré-treino e "colava" a resposta correta (MCAR por design de coleta), sem raciocinar
sobre os dados.

## 1. Design do experimento

### Hipótese de memorização
`mice::boys` e `mice::brandsma` são datasets de referência do livro de Van Buuren (FIMD,
2018), muito citado. O Flash pode ter sido treinado em documentação que menciona
explicitamente que esses datasets têm MCAR por design (visitas clínicas agendadas,
ausências em dia de teste). Se sim, o Flash estaria usando memória, não raciocínio.

### Teste
Criar `real_datasets_metadata_anonymous.json`: mesma estrutura do neutral, mas com:
- Nomes de variáveis substituídos por `X0`, `X1`, `X2`, `X3`, `X4`
- Domínio genérico (`"pediatrics / growth measurement cohort"` → `"health measurement study"`)
- Descrições genéricas (`"Head circumference of a boy"` → `"A continuous numeric measurement"`)
- `source` já era `"Tabular dataset (public repository)"` — mantido

Rodar Flash **apenas nos 6 datasets MCAR** com essa metadata anonimizada e comparar
o feature `llm_ctx_domain_prior` (MCAR=0.0, MAR=0.5, MNAR=1.0) entre as duas runs.

**Se memorização:** com nomes → domain_prior ≈ 0.0 (Flash sabe que é MCAR)
**Se raciocínio:** com nomes → domain_prior ≠ 0.0 (Flash não recupera a resposta)

## 2. Resultados

### 2.1 `llm_ctx_domain_prior` por dataset MCAR

| Dataset | Neutral (com nomes) | Anonymous (sem nomes) | Δ | LLM anon classifica |
|---------|:-------------------:|:---------------------:|:--:|:-------------------:|
| MCAR_boys_hc | 0.52 | **0.96** | +0.44 | MNAR |
| MCAR_boys_hgt | 0.46 | **0.83** | +0.37 | MNAR |
| MCAR_brandsma_apr | 0.46 | **0.82** | +0.36 | MNAR |
| MCAR_brandsma_lpr | 0.52 | **0.80** | +0.28 | MNAR |
| MCAR_hepatitis_albumin | 0.63 | **0.97** | +0.34 | MNAR |
| MCAR_hepatitis_alkphosphate | 0.56 | **0.85** | +0.29 | MNAR |
| **Média MCAR** | **0.53** | **0.87** | **+0.35** | **MNAR** |

### 2.2 Comparação completa dos 9 features LLM (média MCAR)

| Feature | Neutral μ | Anonymous μ | Δ |
|---------|:---------:|:-----------:|:--:|
| `llm_ctx_domain_prior` | 0.5252 | 0.8725 | +0.3473 |
| `llm_ctx_domain_confidence` | 0.7989 | 0.8172 | +0.0183 |
| `llm_ctx_stats_consistency` | 0.8768 | 0.9537 | +0.0768 |
| `llm_ctx_surprise` | 0.1302 | 0.0579 | −0.0723 |
| `llm_ctx_confidence_delta` | 0.0977 | 0.0393 | −0.0584 |
| **`llm_ctx_counter_strength`** | **0.5626** | **0.2495** | **−0.3131** |
| `llm_ctx_cause_type` | 0.5218 | 0.8725 | +0.3507 |
| `llm_ctx_n_causes` | 0.9966 | 0.9255 | −0.0711 |
| `llm_ctx_stats_agreement` | 0.9136 | 0.9790 | +0.0654 |

### 2.3 `llm_ctx_domain_prior` por mecanismo real (Flash neutral, 32 datasets)

| Mecanismo real | domain_prior médio | LLM classifica como |
|:--------------:|:-----------------:|:-------------------:|
| MCAR | 0.5252 | MAR |
| MAR | 0.6488 | MAR |
| MNAR | 0.6092 | MAR |

## 3. Diagnóstico

### 3.1 Memorização descartada

Se o Flash memorizasse que `boys` = MCAR, a run neutral teria `domain_prior ≈ 0.0`.
O valor real é **0.52** (neutro, sem preferência por MCAR). O Flash não sabe que
esses datasets são MCAR — ele diz "provavelmente MAR" para todos os 6 datasets MCAR
mesmo com os nomes originais.

### 3.2 O que realmente acontece: calibração de incerteza via contexto

Com nomes de variáveis e descrição de domínio, o LLM:
- Prevê "provavelmente MAR" (0.52) em vez de "certamente MNAR" (0.87 anônimo)
- Monta **contra-argumentos mais fortes** (`counter_strength` = 0.56 vs 0.25)
- É mais "inseguro" sobre o mecanismo — incerteza calibrada

Sem nomes, o Flash vira um detector de MNAR automático para tudo (bias MNAR).

### 3.3 Mecanismo do ganho +9pp em MCAR

O ML classifier usa `domain_prior = 0.52` (neutral) vs `domain_prior = 0.87` (anônimo)
combinado com as 25 features estatísticas. O prior menos errado (0.52 vs 0.87) e o
`counter_strength` mais alto (0.56) permitem que o ML corrija com as estatísticas e
acerte mais MCAR.

O Flash nunca disse "MCAR" para esses datasets — disse "talvez MAR", o que é menos
prejudicial do que dizer "certamente MNAR".

### 3.4 MNAR: por que Flash piora (−6pp)?

Nos datasets MNAR com censoring físico (NHANES, SUPPORT2), o domínio genérico "health
measurement study" não é suficiente para identificar o mecanismo. O Flash oscila entre
MAR e MNAR (0.61 de prior), enquanto as estatísticas de censoring (kurtosis, Q1-Q4 rates)
são fortes o suficiente para o ML acertar sem LLM. O sinal LLM adiciona ruído, não sinal.

## 4. Implicações para a tese

### 4.1 O ganho não é trapaça

O Flash não está "colando" a resposta de datasets memorizados. O mecanismo é legítimo:
contexto de domínio → calibração de incerteza → melhor combinação com estatísticas.

### 4.2 Generalizabilidade

O ganho em MCAR é parcialmente dependente de a descrição do domínio existir. Para
datasets completamente anônimos sem nenhuma descrição de variável, o Flash se torna
um MNAR-detector automático, o que piora o recall MCAR. Para datasets com descrição
(mesmo genérica), o Flash é útil.

Isso é transferível para dados novos, **desde que o analista forneça uma descrição
mínima do domínio e das variáveis** — o que é uma hipótese razoável em aplicações
reais.

### 4.3 LLM não resolve o problema de identificação de mecanismo

`domain_prior` é uniformemente ~0.5–0.65 para todos os mecanismos — o Flash não
discrimina MCAR/MAR/MNAR pelo contexto semântico. O valor do LLM está na calibração
de incerteza (features como `counter_strength`, `confidence_delta`), não na previsão
direta do mecanismo.

## 5. Arquivos gerados

| Caminho | Conteúdo |
|---|---|
| `src/missdetect/metadata/real_datasets_metadata_anonymous.json` | Metadata totalmente anonimizada (32 entradas) |
| `src/missdetect/metadata/datasets_mcar_only.txt` | Lista dos 6 datasets MCAR |
| `results/step13_flash_anonymous_mcar_only/` | Features extraídas com metadata anônima (298 bootstraps) |
| `src/missdetect/llm/context_aware.py` | Adicionado suporte a `--metadata-variant anonymous` |
| `src/missdetect/extract_features.py` | Validação atualizada para aceitar `anonymous` |
