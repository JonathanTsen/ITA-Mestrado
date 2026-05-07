# missdetect

**Classificação hierárquica de mecanismos de dados faltantes (MCAR / MAR / MNAR) com features estatísticas e aumentadas por LLM.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

> 🇬🇧 [English version](README.md)

Código de referência da dissertação de mestrado *"Classificação Hierárquica
de Mecanismos de Dados Faltantes: Engenharia de Features Estatísticas com
Aumento por LLM e Validação em Dados Reais"* (Jonathan Tsen, ITA, 2026).

---

## Resumo

Dada uma coluna univariada com valores faltantes, classificar se a
ausência é **MCAR** (completamente aleatória), **MAR** (depende de variáveis
observadas) ou **MNAR** (depende do próprio valor faltante). Comparamos um
baseline puramente estatístico contra uma variante aumentada por LLM em
1.200 datasets sintéticos e 29 datasets reais, usando um classificador
hierárquico em dois níveis robusto a ruído de rótulo.

**Achado principal:** features de LLM contribuem marginalmente para a
acurácia em dados reais (+1,9pp com Gemini Pro; nenhuma com Flash). É a
**estrutura hierárquica + Naive Bayes ponderado por Cleanlab** que faz a
diferença. Documentamos um **teto teórico** de ~60–65% (Rubin 1976 + 59,4%
de ruído de rótulo medido) e discutimos a lacuna explicitamente.

---

## Resultados-chave

| Configuração | Datasets reais | Acurácia CV (Group 5-Fold) | Holdout | Custo |
|:--|:-:|:-:|:-:|:-:|
| **V3+ Hierárquico (pico)** — 23 datasets, 25 features estatísticas apenas | 23 | **55,97%** | 50,5% | $0 |
| Step 1 V2 Neutral — Pro + estatísticas, benchmark expandido | 29 | **49,33%** | 55,19% | ~$30 |
| Step 1 V2 Neutral — Flash + estatísticas | 29 | 47,44% | 51,14% | ~$3 |
| Baseline ML-only (sem LLM) | 29 | 47,47% | 53,92% | $0 |
| Acaso (3 classes) | — | 33,3% | 33,3% | $0 |
| Teto teórico (Rubin 1976 + ruído medido) | — | ~60–65% | — | — |

NaiveBayes consistentemente supera RandomForest, GradientBoosting, MLP e SVM
por +6 a +13pp em validação cruzada — calibração de incerteza importa mais
do que capacidade do modelo neste regime de rótulos ruidosos.

Comece por [`docs/README.md`](docs/README.md), que organiza a documentacao
atual. Veja [`docs/HISTORICO.md`](docs/HISTORICO.md) para o registro
experimental completo das fases do projeto.

---

## Por que isso é interessante

Até onde sabemos, este é o **primeiro** estudo a combinar features
derivadas de LLM com classificação hierárquica para o problema MCAR/MAR/MNAR.
A literatura prévia testa apenas MCAR vs todo o resto (PKLM, teste de
Little) ou usa apenas features estatísticas (MechDetect). O resultado
negativo das features de LLM em dados reais ruidosos — que consideramos
publicável — e a lacuna documentada entre o teto sintético (~80%) e o
real (~60%) são contribuições que merecem ser reportadas.

Ressalva teórica: pelo resultado de impossibilidade de Molenberghs et al.
(2008), todo modelo MNAR tem uma contraparte MAR com ajuste idêntico aos
dados observados. Separação MAR/MNAR perfeita é, portanto, inalcançável
apenas a partir de dados observados — nossos 49–56% devem ser lidos contra
esse teto, não contra o acaso.

---

## Estrutura do repositório

```
.
├── src/missdetect/         # Pacote Python (instalável)
│   ├── extract_features.py # Pipeline de extração de features
│   ├── train_model.py      # Classificadores em nível único
│   ├── train_hierarchical_v3plus.py   # V3+ Cleanlab + roteamento soft3zone
│   ├── run_all.py          # Orquestrador end-to-end
│   ├── features/           # Features estatísticas + discriminativas + MechDetect
│   ├── llm/                # Extratores aumentados por LLM (judge, context-aware, …)
│   ├── baselines/          # MechDetect (Le et al. 2024), PKLM (Sportisse 2024)
│   ├── data_generation/    # Gerador sintético via mdatagen + preparo de reais
│   └── metadata/           # Metadados de datasets usados pelos prompts de LLM
│
├── data/                   # Datasets (ver data/README.md)
│   ├── synthetic/          # 1.200 séries geradas (MCAR/MAR/MNAR × 12 variantes)
│   └── real/               # 29 colunas curadas de UCI / OpenML / Kaggle
│       ├── raw/            # CSVs originais
│       ├── processed/      # Séries com bootstrap por mecanismo
│       └── sources.md      # Procedência e licença por dataset
│
├── results/                # Saídas reproduzíveis dos experimentos
│   ├── step1_v2_neutral/   # Mais recente: 29 datasets, Pro + metadata neutra
│   ├── step05_pro/         # Pico V3+ em 23 datasets
│   └── _archive/           # Experimentos históricos (gitignored — zip sob demanda)
│
├── docs/                   # Documentação
│   ├── README.md           # Comece aqui: mapa dos docs e notas de terminologia
│   ├── HISTORICO.md        # Narrativa mestra das 7 fases (PT-BR)
│   ├── methodology.md      # Síntese metodológica (EN)
│   ├── caafe_mnar.md       # Terminologia correta: CAAFE original vs CAAFE-MNAR
│   ├── reproducibility.md  # Replay passo-a-passo dos experimentos principais
│   ├── bibliography.md     # Bibliografia anotada por tópico
│   ├── code/               # Documentação interna do código
│   └── archive/            # 76 notas de planejamento/decisão (PT-BR)
│
├── tests/                  # Smoke tests
├── pyproject.toml          # Packaging moderno (PEP 517/518) + ruff/mypy/pytest
├── LICENSE                 # MIT
├── CITATION.cff            # Metadados de citação legíveis pelo GitHub
└── README.md               # Versão em inglês
```

---

## Início rápido

Requer Python 3.11+. Recomendamos [`uv`](https://docs.astral.sh/uv/), mas
qualquer builder PEP 517 funciona.

```bash
# 1. Clonar e entrar
git clone https://github.com/JonathanTsen/missdetect.git
cd missdetect

# 2. Instalar (uv)
uv venv
source .venv/bin/activate
uv pip install -e ".[boosting,llm]"

# 2b. Instalar (pip tradicional)
python -m venv .venv && source .venv/bin/activate
pip install -e ".[boosting,llm]"

# 3. (Opcional) Credenciais de LLM em .env
cat > .env <<'ENV'
GEMINI_API_KEY=...
OPENAI_API_KEY=...
ENV
```

Todos os comandos abaixo assumem que você está na raiz do repositório.
Os scripts usam `python -m missdetect.<modulo>` para que `sys.path` resolva
o pacote corretamente independentemente do diretório de trabalho.

### Rodar o experimento principal (baseline estatístico)

```bash
# Extrair features de dados sintéticos (sem LLM, ~5 min)
python -m missdetect.extract_features --model none --data synthetic

# Treinar os sete classificadores com CV ciente de grupos
python -m missdetect.train_model --model none --data synthetic
```

### Replicar o V3+ pico (23 datasets reais, sem LLM)

```bash
python -m missdetect.extract_features --model none --data real --metadata-variant neutral
python -m missdetect.train_hierarchical_v3plus \
  --model none --data real \
  --experiment step05_pro
```

### Replicar Step 1 V2 Neutral (29 datasets, Pro + LLM)

```bash
# Requer GEMINI_API_KEY e ~$30 de orçamento de API
python -m missdetect.extract_features \
  --model gemini-3-pro-preview \
  --llm-approach context_aware \
  --metadata-variant neutral \
  --data real

python -m missdetect.train_model \
  --model gemini-3-pro-preview \
  --data real \
  --experiment step1_v2_neutral
```

Veja [`docs/reproducibility.md`](docs/reproducibility.md) para o conjunto
completo de comandos, tempos esperados e hashes de validação.

---

## Limitações

Este README e a dissertação que o acompanha tornam quatro limitações
explícitas:

1. **MAR e MNAR são teoricamente indistinguíveis** apenas a partir de dados
   observados (Molenberghs et al. 2008). Nossa faixa de 49–56% reflete esse
   teto teórico — não um problema de modelagem.
2. **Ruído de rótulo** alto no benchmark real: 59,4% das amostras de
   bootstrap são sinalizadas pelo Cleanlab como potencialmente
   mal-rotuladas. Os rótulos são a melhor estimativa de especialistas de
   domínio, não verdade absoluta.
3. **Features de LLM agregam pouco** no regime de dados reais uma vez que as
   features estatísticas já fizeram seu trabalho. Gemini Pro entrega +1,9pp
   a ~10× o custo do Flash; Flash é dominado em Pareto pelo baseline puramente
   estatístico.
4. **Variância alta** sob Group 5-Fold CV (±14–27pp) — seis classes de
   datasets reais com características distintas dominam quais datasets vão
   para treino vs teste, e replicar com mais seeds é caro quando há LLM no
   pipeline.

---

## Citação

Se você usar este código ou os resultados experimentais, por favor cite
tanto o software quanto a dissertação. O GitHub renderiza
[`CITATION.cff`](CITATION.cff) diretamente; em BibTeX:

```bibtex
@mastersthesis{tsen2026missdetect,
  title  = {Classificação Hierárquica de Mecanismos de Dados Faltantes:
            Engenharia de Features Estatísticas com Aumento por LLM
            e Validação em Dados Reais},
  author = {Tsen, Jonathan},
  school = {Instituto Tecnológico de Aeronáutica (ITA)},
  year   = {2026},
  url    = {https://github.com/JonathanTsen/missdetect}
}
```

---

## Licença

[MIT](LICENSE) © 2026 Jonathan Tsen.

Os datasets reais embutidos são redistribuídos sob suas licenças originais
(UCI MLR / OpenML / Kaggle) — ver [`data/real/sources.md`](data/real/sources.md).

---

## Agradecimentos

Este trabalho foi desenvolvido no programa de mestrado do Instituto
Tecnológico de Aeronáutica (ITA), São José dos Campos.
