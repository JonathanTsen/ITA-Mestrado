# Classificador de Mecanismos de Missing Data (v2 - Otimizado)

Sistema completo para **geração de datasets sintéticos** e **classificação automática** de mecanismos de dados faltantes (MCAR, MAR, MNAR) utilizando Machine Learning e opcionalmente LLMs.

**🆕 VERSÃO OTIMIZADA:** Reduzido de 68 para **18 features relevantes** após análise exaustiva de relevância.

---

## 📁 Estrutura do Projeto

```
Scripts/
├── README.md                    # Este arquivo
├── requirements.txt             # Dependências Python
├── .env                         # Chaves de API (criar manualmente)
├── gerador.py                   # Gerador de datasets sintéticos
└── v2_improved/                 # Pipeline de ML v2 (OTIMIZADO)
    ├── README.md                # Documentação detalhada do v2
    ├── run_all.py               # Pipeline completo (executa tudo)
    ├── extract_features.py      # Extração de features (18 features)
    ├── train_model.py           # Treinamento de modelos
    ├── compare_results.py       # Comparação de resultados
    ├── analyze_feature_relevance.py  # Script de análise de relevância
    ├── features/
    │   ├── statistical.py       # Features estatísticas (4 features)
    │   └── discriminative.py    # Features discriminativas (6 features)
    ├── llm/
    │   ├── extractor_v2.py      # Extrator LLM v2 (8 features)
    │   └── __init__.py
    ├── core/                    # Módulos centrais
    ├── ml/                      # Módulos de ML
    └── utils/                   # Utilitários
```

---

## 🚀 Passo a Passo Completo

### **ETAPA 1: Configuração do Ambiente**

#### 1.1 Instalar UV (recomendado)
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Ou via Homebrew
brew install uv
```

#### 1.2 Configurar chaves de API (opcional, apenas se usar LLM)
Crie/edite o arquivo `Scripts/v2_improved/.env`:
```bash
cd "/Users/tsen/Documents/ITA/IC - ITA 2/Scripts/v2_improved"
echo "OPENAI_API_KEY=sua_chave_openai_aqui" > .env
echo "GEMINI_API_KEY=sua_chave_gemini_aqui" >> .env
```

---

### **ETAPA 2: Gerar Datasets Sintéticos**

```bash
cd "/Users/tsen/Documents/ITA/IC - ITA 2/Scripts"
uv run python gerador.py
```

**O que acontece:**
- Gera **3000 datasets** (1000 por mecanismo: MCAR, MAR, MNAR)
- Cada dataset tem **1000 linhas** e **5 colunas** (X0, X1, X2, X3, X4)
- Missing é aplicado **apenas em X0** com taxa de 1% a 10%
- Datasets salvos em `../Dataset/{MCAR,MAR,MNAR}/`

**Saída esperada:**
```
✅ Banco sintético gerado (1000 por classe) e X0 garantidamente com missing.
```

---

### **ETAPA 3: Extrair Features (OTIMIZADO - 18 features)**

```bash
cd "/Users/tsen/Documents/ITA/IC - ITA 2/Scripts/v2_improved"

# Sem LLM (baseline - 10 features: 4 estatísticas + 6 discriminativas)
uv run python extract_features.py --model none

# Com LLM (18 features: 4 estatísticas + 6 discriminativas + 8 LLM)
uv run python extract_features.py --model gemini-3-flash-preview

# Modo teste (apenas 50 arquivos - para debug)
uv run python extract_features.py --model none --test
```

**O que acontece:**
- Lê todos os datasets de `../Dataset/`
- Extrai **18 features otimizadas** por dataset:
  - **4 features estatísticas**: X0_mean, X0_q25, X0_q50, X0_q75
  - **6 features discriminativas**: auc_mask_from_Xobs, coef_X1_abs, log_pval_X1_mask, X1_mean_diff, X1_mannwhitney_pval, little_proxy_score
  - **8 features LLM** (se habilitado): llm_evidence_consistency, llm_anomaly, llm_dist_shift, llm_mcar_conf, llm_mar_conf, llm_mnar_conf, llm_mcar_vs_mnar, llm_pattern_clarity
- Salva em `../Output/v2_improved/{modelo}/`

**Saída esperada:**
```
✅ EXTRAÇÃO CONCLUÍDA!
📊 Shape X: (3000, 10)  # sem LLM
📊 Shape X: (3000, 18)  # com LLM
📊 Shape y: (3000,)
```

---

### **ETAPA 4: Treinar Modelos**

```bash
cd "/Users/tsen/Documents/ITA/IC - ITA 2/Scripts/v2_improved"

# Treinar com features do modelo "none" (baseline - 10 features)
uv run python train_model.py --model none

# Treinar com features do modelo com LLM (18 features)
uv run python train_model.py --model gemini-3-flash-preview
```

**O que acontece:**
- Carrega features de `../Output/v2_improved/{modelo}/`
- Treina **7 modelos de ML**: RandomForest, GradientBoosting, LogisticRegression, SVM, KNN, MLP, NaiveBayes
- Faz **cross-validation** (5-fold)
- Gera relatório e gráficos

**Saída esperada:**
```
✅ TREINAMENTO CONCLUÍDO!
📊 RESULTADOS:
   GradientBoosting    : 0.7120
   RandomForest        : 0.6890
   ...
🏆 Melhor modelo: GradientBoosting (0.7120)
```

---

### **ETAPA 5: Comparar Resultados (Opcional)**

```bash
cd "/Users/tsen/Documents/ITA/IC - ITA 2/Scripts/v2_improved"
uv run python compare_results.py
```

Compara resultados entre diferentes configurações (com/sem LLM).

---

### **ALTERNATIVA: Pipeline Completo (run_all.py)**

Executa todas as etapas automaticamente:

```bash
cd "/Users/tsen/Documents/ITA/IC - ITA 2/Scripts/v2_improved"

# Execução completa
uv run python run_all.py

# Modo teste (50 arquivos)
uv run python run_all.py --test
```

---

## 📊 Arquivos de Saída

Após execução, os resultados ficam em:

```
Output/
└── v2_improved/
    ├── none/                           # Resultados sem LLM (10 features)
    │   ├── X_features.csv              # Matriz de features (3000 x 10)
    │   ├── y_labels.csv                # Labels (0=MCAR, 1=MAR, 2=MNAR)
    │   ├── relatorio.txt               # Relatório detalhado
    │   ├── resultados.png              # Gráfico de acurácias
    │   └── precisao_por_classe.png     # Precision/Recall por classe
    └── gemini-3-flash-preview/         # Resultados com LLM (18 features)
        ├── X_features.csv              # Matriz de features (3000 x 18)
        ├── y_labels.csv
        ├── relatorio.txt
        ├── resultados.png
        └── precisao_por_classe.png
```

---

## 🔬 Por que 5 Colunas (X0-X4)?

### Explicação Técnica

O dataset precisa de **múltiplas colunas** para simular corretamente os 3 mecanismos:

| Coluna | Papel | Uso |
|--------|-------|-----|
| **X0** | Coluna com missing | Alvo da análise |
| **X1** | Variável observada principal | **MAR depende de X1** |
| **X2-X4** | Variáveis observadas auxiliares | Testes estatísticos robustos |

### Por que cada mecanismo precisa disso?

#### **MCAR (Missing Completely At Random)**
```
P(missing) = constante
```
- Missing é **completamente aleatório**
- Poderia funcionar com 1 coluna
- Mas precisamos de X1-X4 para **provar** que não há dependência

#### **MAR (Missing At Random)**
```
P(missing | X0, X1) = P(missing | X1)
```
- Missing em X0 **depende de X1** (variável observada)
- **Obrigatoriamente precisa de X1**
- Código em `gerador.py` linha 125:
  ```python
  gen = uMAR(X=X, y=y_dummy, missing_rate=missing_rate, x_miss="X0", x_obs="X1")
  ```

#### **MNAR (Missing Not At Random)**
```
P(missing | X0) depende do próprio X0
```
- Missing depende do **próprio valor** de X0
- Poderia funcionar com 1 coluna
- Mas X1-X4 ajudam a **distinguir** de MAR

### Por que não 1 coluna?

**Impossível distinguir os mecanismos!**

Com apenas X0:
- ❌ Não consegue simular MAR (precisa de X1)
- ❌ Não consegue calcular `auc_mask_from_Xobs` (feature discriminativa chave)
- ❌ Não consegue fazer Little's MCAR test proxy
- ❌ Não consegue detectar MAR vs MCAR (correlação com observados)

### Como as features usam as 5 colunas

#### `features/discriminative.py`:
```python
# Usa X1-X4 para prever missing (MAR detection)
X_predictors = df[["X1", "X2", "X3", "X4"]].values
clf.fit(X_scaled, mask)
feats["auc_mask_from_Xobs"] = roc_auc_score(mask, proba)  # Alto = MAR
```

#### `features/statistical.py`:
```python
# Correlação entre máscara e cada Xi
for col in ["X1", "X2", "X3", "X4"]:
    feats[f"corr_mask_{col}"] = np.corrcoef(mask, xi)[0, 1]
```

---

## 📈 Interpretação dos Resultados

### Features Mais Importantes (RandomForest)

| Feature | Alto valor indica |
|---------|-------------------|
| `auc_mask_from_Xobs` | **MAR** (missing depende de X_obs) |
| `corr_X1_mask` | **MAR** (correlação direta) |
| `X0_obs_mean_deviation` | **MNAR** (distribuição enviesada) |
| `little_proxy_score` | Rejeita **MCAR** |
| `runs_z_score` | Padrão não-aleatório |

### Acurácia Esperada

| Configuração | Acurácia | Features |
|--------------|----------|----------|
| v2 otimizado sem LLM | ~65-71% | 10 features |
| v2 otimizado com LLM | ~68-75% | 18 features |
| v2 original (todas features) | ~66-67% | 68 features |
| Random guess | 33.3% | - |

**Nota:** A versão otimizada com menos features pode ter performance igual ou superior devido à redução de overfitting.

---

## 🛠️ Troubleshooting

### Erro: "Pasta não encontrada"
```bash
# Rode primeiro o gerador
python gerador.py
```

### Erro: "OPENAI_API_KEY not found"
```bash
# Crie o arquivo .env com suas chaves
echo "OPENAI_API_KEY=sua_chave" > .env
```

### Erro: "ModuleNotFoundError"
```bash
# Reinstale dependências
pip install -r requirements.txt
```

### Processamento muito lento
```bash
# Use modo teste para debug
python extract_features.py --model none --test
```

---

## 📚 Referências

- **MCAR/MAR/MNAR**: Rubin, D. B. (1976). Inference and missing data. Biometrika.
- **Little's MCAR Test**: Little, R. J. A. (1988). A test of missing completely at random.
- **mdatagen**: Biblioteca Python para geração de dados com missing.

---

## 📝 Changelog

- **v2.0**: Reescrita completa com features discriminativas e LLM v2
- **v1.0**: Versão original (removida)
