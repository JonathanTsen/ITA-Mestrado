# v2_improved - Classificação de Mecanismos de Missing Data

## Visão Geral

Esta versão melhorada implementa uma estratégia aprimorada para extrair features de séries temporais com dados faltantes, utilizando LLM (Large Language Models) de forma mais eficiente.

## 🎯 Resultados

### Comparação v1 (original) vs v2 (melhorada)

| Modelo | v1 (sem LLM) | v1 (com LLM) | **v2 (sem LLM)** |
|--------|-------------|--------------|------------------|
| NaiveBayes | 53.2% | 53.4% | **70.9%** |
| LogisticRegression | 58.1% | 57.7% | **70.7%** |
| RandomForest | 51.6% | 51.8% | **65.1%** |
| SVM_RBF | 55.9% | 53.6% | **66.1%** |

**Melhoria média: +15-18 pontos percentuais!**

## Problemas Identificados na Versão Original

1. **Prompt fraco**: A LLM recebia apenas dados brutos sem contexto estatístico
2. **Features genéricas**: Scores abstratos sem base estatística
3. **Falta de features discriminativas**: Não capturavam diferenças entre MCAR/MAR/MNAR

## Melhorias Implementadas

### 1. Features Estatísticas Robustas (`features/statistical.py`)
- Distribuição de X0 (média, std, skewness, kurtosis)
- Padrões temporais (autocorrelação, runs test)
- Análise de bursts de missing
- Posição dos missing na série

### 2. Features Discriminativas (`features/discriminative.py`)
- **MCAR vs outros**: AUC de prever missing usando X_obs
- **MAR**: Correlação e testes estatísticos entre X1 e mask
- **MNAR**: Análise de valores extremos e padrões de vizinhança
- Little's MCAR test proxy

### 3. Prompt Engineering Avançado (`llm/extractor.py`)
- Fornece estatísticas pré-calculadas (não dados brutos)
- Contexto teórico sobre MCAR/MAR/MNAR
- Schema estruturado com Pydantic
- Scores de confiança para cada mecanismo

## Estrutura do Projeto

```
v2_improved/
├── core/                    # Módulos centrais
├── features/
│   ├── statistical.py       # Features estatísticas básicas
│   └── discriminative.py    # Features que discriminam MCAR/MAR/MNAR
├── llm/
│   └── extractor.py         # Extrator de features com LLM melhorado
├── ml/                      # Módulos de machine learning
├── utils/                   # Utilitários
├── extract_features.py      # Script principal de extração
├── train_model.py           # Script de treinamento
├── compare_results.py       # Script de comparação
├── run_all.py               # Pipeline completo
└── README.md
```

## Como Usar (com UV)

### Instalação de Dependências
```bash
uv pip install numpy pandas scipy scikit-learn matplotlib tqdm python-dotenv pydantic
# Para LLM:
uv pip install langchain-openai langchain-google-genai
```

### 1. Extrair Features (sem LLM - baseline)
```bash
cd Scripts/v2_improved
uv run python extract_features.py --model none
```

### 2. Extrair Features (com LLM)
```bash
uv run python extract_features.py --model gemini-3-flash-preview
# ou
uv run python extract_features.py --model gpt-5.2
```

### 3. Modo Teste (50 arquivos)
```bash
uv run python extract_features.py --model none --test
```

### 4. Treinar Modelos
```bash
uv run python train_model.py --model none
uv run python train_model.py --model gemini-3-flash-preview
```

### 5. Comparar Resultados
```bash
uv run python compare_results.py
```

### 6. Pipeline Completo
```bash
uv run python run_all.py
# ou modo teste:
uv run python run_all.py --test
```

## Saída

Os resultados são salvos em `Output/v2_improved/<modelo>/`:
- `X_features.csv` - Matriz de features
- `y_labels.csv` - Labels (0=MCAR, 1=MAR, 2=MNAR)
- `relatorio.txt` - Relatório detalhado
- `resultados.png` - Gráfico de acurácias
- `precisao_por_classe.png` - Precision/Recall por classe

## Features Extraídas

### Features Estatísticas (35+)
| Feature | Descrição |
|---------|-----------|
| `missing_rate` | Taxa de missing em X0 |
| `X0_mean`, `X0_std` | Estatísticas de X0 observado |
| `mask_autocorr_1` | Autocorrelação lag-1 da máscara |
| `runs_z_score` | Z-score do runs test |
| `avg_burst_size` | Tamanho médio de bursts |
| `corr_mask_X1..X4` | Correlação máscara vs Xi |

### Features Discriminativas (20+)
| Feature | Indica |
|---------|--------|
| `auc_mask_from_Xobs` | MAR se alto (>0.6) |
| `corr_X1_mask` | MAR se significativo |
| `X1_mean_diff` | MAR se ≠ 0 |
| `missing_rate_extremes` | MNAR se alto |
| `little_proxy_score` | Rejeita MCAR se alto |

### Features LLM (8)
| Feature | Descrição |
|---------|-----------|
| `llm_mcar_conf` | Confiança MCAR |
| `llm_mar_conf` | Confiança MAR |
| `llm_mnar_conf` | Confiança MNAR |
| `llm_randomness` | Score de aleatoriedade |
| `llm_dep_X1` | Dependência de X1 |

## Teoria

### MCAR (Missing Completely At Random)
- P(missing) = constante
- Missing independente de todas as variáveis
- **Indicadores**: baixa correlação, padrão aleatório

### MAR (Missing At Random)
- P(missing | X0, X1) = P(missing | X1)
- Missing depende de variáveis observadas
- **Indicadores**: correlação com X1, diferença de médias

### MNAR (Missing Not At Random)
- P(missing | X0) depende de X0
- Missing depende do próprio valor faltante
- **Indicadores**: missing em extremos, padrões de vizinhança

## Requisitos

```
numpy
pandas
scipy
scikit-learn
matplotlib
tqdm
python-dotenv
pydantic
langchain-openai  # para OpenAI
langchain-google-genai  # para Gemini
```

## Configuração

Crie um arquivo `.env` em `Scripts/`:
```
OPENAI_API_KEY=sua_chave_aqui
GEMINI_API_KEY=sua_chave_aqui
```
