# STEP 03: Novas Features para L2 — Divergência de Imputação + Independência Condicional

**Impacto esperado:** +3-7pp accuracy no L2
**Esforço:** Médio (~150 linhas)
**Dependências:** Nenhuma
**Pacotes:** `dcor` (distance correlation)

---

## Motivação

O L2 (MAR vs MNAR) é o gargalo principal (53% accuracy). As 4 features CAAFE capturam padrões de forma da distribuição de X0, mas não exploram:

1. **Concordância entre métodos de imputação** — se MNAR, imputações divergem
2. **Independência condicional** mask ⊥ X0 | X_obs — testa diretamente a definição de MAR
3. **Density ratio** entre observados e missing — captura shifts não-lineares

Estas 3 famílias atacam o problema de ângulos diferentes dos CAAFE, potencialmente complementares.

---

## Família 1: Divergência entre Métodos de Imputação (3 features)

### Intuição

- **MAR:** X0 é missing por causa de X1-X4 (dados observados). Qualquer método que use X1-X4 para imputar X0 (KNN, MICE) chegará a resultados similares ao imputation simples.
- **MNAR:** X0 é missing por causa de X0 em si. Nenhum método consegue recuperar X0 perfeitamente — mas cada método erra de forma diferente. A **divergência** entre imputações é um sinal de MNAR.

### Features

| Feature | Computação | Sinal MNAR |
|---------|-----------|:----------:|
| `adv_imputation_divergence_ks` | Média dos KS stats entre 3 pares de imputações | Alto (>0.1) |
| `adv_imputation_divergence_wasserstein` | Max Wasserstein distance entre pares | Alto |
| `adv_imputation_cv` | Coeficiente de variação das médias imputadas | Alto |

### Implementação

```python
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from scipy.stats import ks_2samp, wasserstein_distance

def compute_imputation_divergence(df):
    """Compara 3 métodos de imputação para detectar MNAR."""
    X0 = df["X0"].values
    mask = np.isnan(X0)
    
    if mask.sum() < 5 or (~mask).sum() < 5:
        return {"adv_imputation_divergence_ks": 0.0,
                "adv_imputation_divergence_wasserstein": 0.0,
                "adv_imputation_cv": 0.0}
    
    # Imputação com 3 métodos
    X_full = df[["X0", "X1", "X2", "X3", "X4"]].values
    
    # Método 1: Median
    imp_median = SimpleImputer(strategy="median").fit_transform(X_full)[:, 0]
    
    # Método 2: KNN (usa X1-X4 como vizinhos)
    imp_knn = KNNImputer(n_neighbors=5).fit_transform(X_full)[:, 0]
    
    # Método 3: MICE/Iterative (usa relações multivariadas)
    imp_mice = IterativeImputer(max_iter=10, random_state=42).fit_transform(X_full)[:, 0]
    
    # Extrair apenas valores imputados (onde era NaN)
    vals = {
        "median": imp_median[mask],
        "knn": imp_knn[mask],
        "mice": imp_mice[mask],
    }
    
    # KS entre pares
    pairs = [("median", "knn"), ("median", "mice"), ("knn", "mice")]
    ks_stats = [ks_2samp(vals[a], vals[b]).statistic for a, b in pairs]
    ws_dists = [wasserstein_distance(vals[a], vals[b]) for a, b in pairs]
    
    # CV das médias
    means = [v.mean() for v in vals.values()]
    cv = np.std(means) / max(np.mean(np.abs(means)), 1e-10)
    
    return {
        "adv_imputation_divergence_ks": float(np.mean(ks_stats)),
        "adv_imputation_divergence_wasserstein": float(np.max(ws_dists)),
        "adv_imputation_cv": float(cv),
    }
```

---

## Família 2: Teste de Independência Condicional (2 features)

### Intuição

A **definição de MAR** é: P(R=1 | X0, X_obs) = P(R=1 | X_obs). Ou seja, dado X_obs, a missingness é independente de X0.

Teste direto:
1. Remover o efeito de X_obs em X0 → residual_X0
2. Remover o efeito de X_obs em mask → residual_mask
3. Testar se residual_X0 e residual_mask são independentes

Se **independentes** → MAR (X_obs explica tudo). Se **dependentes** → MNAR (X0 tem informação residual sobre mask).

### Features

| Feature | Computação | Sinal MNAR |
|---------|-----------|:----------:|
| `adv_partial_dcor_X0_mask` | Distance correlation entre resíduos | Alto (>0.05) |
| `adv_residual_spearman_X0_mask` | Spearman entre resíduos | |ρ| alto |

### Implementação

```python
import dcor  # pip install dcor

def compute_conditional_independence(df):
    """Testa mask ⊥ X0 | X_obs usando resíduos parciados."""
    mask = df["X0"].isna().astype(float).values
    X0_imputed = df["X0"].fillna(df["X0"].median()).values
    X_obs = df[["X1", "X2", "X3", "X4"]].values
    
    if len(X0_imputed) < 20:
        return {"adv_partial_dcor_X0_mask": 0.0,
                "adv_residual_spearman_X0_mask": 0.0}
    
    # Residualizar X0 e mask contra X_obs
    from sklearn.linear_model import LinearRegression
    
    lr_x0 = LinearRegression().fit(X_obs, X0_imputed)
    residual_x0 = X0_imputed - lr_x0.predict(X_obs)
    
    lr_mask = LinearRegression().fit(X_obs, mask)
    residual_mask = mask - lr_mask.predict(X_obs)
    
    # Distance correlation (zero iff independent)
    partial_dcor = dcor.distance_correlation(residual_x0, residual_mask)
    
    # Spearman (complementar, mais simples)
    from scipy.stats import spearmanr
    spearman_rho, _ = spearmanr(residual_x0, residual_mask)
    
    return {
        "adv_partial_dcor_X0_mask": float(partial_dcor),
        "adv_residual_spearman_X0_mask": float(abs(spearman_rho)),
    }
```

### Nota Teórica

O X0_imputed usa mediana, que é enviesado para MNAR (não captura a cauda missing). Mas **o próprio viés é informativo**: se a imputação cria dependência residual com a mask, é evidência de MNAR. Sob MAR, a imputação mediana deveria ser "boa o suficiente" para que os resíduos fiquem independentes.

---

## Família 3: KDE Density Ratio (2 features)

### Intuição

Se plotarmos p(X0 | missing) vs p(X0 | observado):
- **MCAR:** As duas densidades são iguais (ratio ≈ 1 em todo lugar)
- **MNAR:** As densidades diferem sistematicamente (ratio varia, especialmente nas caudas)

O KS test captura isso parcialmente, mas é global. O density ratio captura **onde** as distribuições diferem.

### Features

| Feature | Computação | Sinal MNAR |
|---------|-----------|:----------:|
| `adv_density_ratio_range` | max(ratio) - min(ratio) em 10 pontos de avaliação | Alto (>2) |
| `adv_density_ratio_tail_asym` | ratio(p90) / ratio(p10) | Longe de 1.0 |

### Implementação

```python
from scipy.stats import gaussian_kde

def compute_density_ratio(df):
    """Estima o ratio de densidades p(X0|missing)/p(X0|observed)."""
    X0 = df["X0"].values
    mask = np.isnan(X0)
    X0_obs = X0[~mask]
    
    if mask.sum() < 10 or (~mask).sum() < 10:
        return {"adv_density_ratio_range": 0.0,
                "adv_density_ratio_tail_asym": 1.0}
    
    # Imputar missing values para ter algo para avaliar a density
    X0_full = np.where(mask, np.median(X0_obs), X0)
    
    # KDE em observados
    try:
        kde_obs = gaussian_kde(X0_obs)
    except np.linalg.LinAlgError:
        return {"adv_density_ratio_range": 0.0, "adv_density_ratio_tail_asym": 1.0}
    
    # KDE em todos (com imputação)
    try:
        kde_all = gaussian_kde(X0_full)
    except np.linalg.LinAlgError:
        return {"adv_density_ratio_range": 0.0, "adv_density_ratio_tail_asym": 1.0}
    
    # Avaliar em quantis
    eval_points = np.percentile(X0_obs, np.linspace(5, 95, 10))
    d_obs = kde_obs(eval_points)
    d_all = kde_all(eval_points)
    
    ratio = d_obs / np.maximum(d_all, 1e-10)
    
    return {
        "adv_density_ratio_range": float(ratio.max() - ratio.min()),
        "adv_density_ratio_tail_asym": float(ratio[-1] / max(ratio[0], 1e-10)),
    }
```

---

## Arquivo Final: `features/advanced_l2.py`

```python
"""
Features avançadas para o Nível 2 (MAR vs MNAR).

Três famílias:
1. Divergência de imputação (3 features) — se MNAR, imputações divergem
2. Independência condicional (2 features) — testa definição de MAR
3. KDE density ratio (2 features) — captura shifts não-lineares

Total: 7 features novas, usadas apenas no L2.
"""

def extract_advanced_l2_features(df: pd.DataFrame) -> dict:
    feats = {}
    feats.update(compute_imputation_divergence(df))
    feats.update(compute_conditional_independence(df))
    feats.update(compute_density_ratio(df))
    return feats
```

---

## Integração

### Em `extract_features.py`:

```python
# Após extrair CAAFE features
from features.advanced_l2 import extract_advanced_l2_features
adv_feats = extract_advanced_l2_features(df)
row.update(adv_feats)
```

### Em `train_hierarchical_v3plus.py`:

Adicionar variante **V3+**:
```python
ADV_L2_COLS = [c for c in X.columns if c.startswith("adv_")]

"V3plus_hier_caafe_adv_n2": {
    "tipo": "hierarquico",
    "features_l1": FEAT_STAT,           # 21 stat (não muda)
    "features_l2": FEAT_STAT_CAAFE + ADV_L2_COLS,  # 25 + 7 = 32 features
    "desc": "Hier: L1=stat, L2=stat+CAAFE+ADV",
}
```

---

## Validação

1. Extrair as 7 novas features para todos os datasets (sintéticos + reais)
2. Verificar distribuição: features devem ter variance > 0 e range razoável
3. Verificar Cohen's d entre MAR e MNAR para cada feature (deve ser > 0.2 para ser útil)
4. Comparar V3 (25f) vs V3+ (32f) no holdout E LOGO CV
5. SHAP no L2: verificar se novas features entram no top 10

---

## Riscos

- **Imputation divergence pode ser lenta** (MICE é iterativo) → limitar max_iter=10
- **dcor pode não estar instalado** → fallback para Spearman parcial
- **Features podem ser ruidosas em datasets pequenos** → monitorar com feature selection (Step 05)
- **KDE falha com distribuições discretas** → try/except com fallback 0.0

---

# Anexo: Resultados do Experimento

> Originalmente publicado como `RESULTADOS_STEP03.md`. Consolidado aqui em 2026-04-19.


**Data:** 2026-04-19
**Experimento:** step03_advl2
**Status:** CONCLUÍDO — **sem ganho, features ADV pioram resultados**

---

## Resumo

As 7 features avançadas para L2 (divergência de imputação, independência condicional, KDE density ratio) **pioram** os resultados em dados reais. MNAR recall cai para 0% em todas as variantes V3adv. O padrão é consistente com a ablação do plano_2: em dados reais ruidosos, mais features = mais ruído.

---

## Features Implementadas (7 novas)

| Feature | Família | Descrição |
|---------|---------|-----------|
| `adv_imputation_divergence_ks` | Divergência | KS entre 3 métodos de imputação |
| `adv_imputation_divergence_wasserstein` | Divergência | Max Wasserstein entre imputações |
| `adv_imputation_cv` | Divergência | CV das médias imputadas |
| `adv_partial_dcor_X0_mask` | Indep. Condicional | Distance correlation resíduos X0-mask |
| `adv_residual_spearman_X0_mask` | Indep. Condicional | Spearman resíduos X0-mask |
| `adv_density_ratio_range` | KDE Density | Range do ratio p(X0\|obs)/p(X0\|all) |
| `adv_density_ratio_tail_asym` | KDE Density | Assimetria ratio nas caudas |

---

## Holdout (295 amostras)

### V3 (CAAFE apenas) vs V3adv (CAAFE + ADV L2)

| Variante | Acc Max | MNAR Recall | F1 | Modelo |
|----------|:-------:|:-----------:|:--:|--------|
| V3 hard | **48.1%** | **28.0%** | **0.454** | GBT |
| V3adv hard | 45.4% | **0.0%** | 0.384 | GBT |
| V3+ soft3zone | **47.1%** | 22.0% | **0.439** | GBT |
| V3adv soft3zone | 46.1% | **0.0%** | 0.386 | GBT |

**V3adv piora accuracy em ~2-3pp e destrói MNAR recall (0% em todas as configurações).**

### Detalhado V3adv_hard por modelo

| Modelo | Accuracy | MNAR Recall | F1 |
|--------|:--------:|:-----------:|:--:|
| GradientBoosting | **45.4%** | 0.0% | 0.384 |
| RandomForest | 44.1% | 6.0% | 0.393 |
| KNN | 42.4% | 26.0% | 0.401 |
| NaiveBayes | 42.4% | 18.0% | 0.379 |
| LogisticRegression | 40.7% | 0.0% | 0.357 |
| SVM_RBF | 40.7% | 6.0% | 0.355 |
| MLP | 38.3% | 16.0% | 0.353 |
| XGBoost | 44.7% | 4.0% | 0.368 |
| CatBoost | 44.1% | 4.0% | 0.371 |

---

## LOGO CV (23 folds)

O LOGO CV rodou para V3/V3plus (sem V3adv). Resultados V3 neste experimento:

| Modelo | V3 hard | V3+ soft3zone | Melhor |
|--------|:-------:|:-------------:|:------:|
| **NaiveBayes** | 51.4% | **55.8%** | +4.3pp |
| LogisticRegression | 44.7% | **45.1%** | +0.3pp |
| KNN | 36.8% | **38.4%** | +1.6pp |
| MLP | 34.2% | **36.1%** | +2.0pp |
| XGBoost | 38.2% | 36.8% | -1.4pp |
| CatBoost | 37.5% | 36.6% | -0.9pp |

**NaiveBayes soft3zone = 55.8%** (vs 56.0% no step05_pro — diferença marginal de ~0.2pp por re-extração).

**Nota:** XGBoost e CatBoost (Step 02) não melhoram sobre os modelos originais no LOGO CV. NaiveBayes continua campeão.

---

## Análise: Por que features ADV pioram?

### 1. Labels ruidosos dominam
Com 59.4% de labels inconsistentes, features novas que medem sinais sutis (divergência de imputação, independência condicional) capturam mais o **ruído dos labels** do que o sinal real MAR vs MNAR.

### 2. MNAR recall = 0% é sintomático
As features ADV fazem o L2 classificar tudo como MAR. Provavelmente, as features ADV têm distribuição similar entre MAR e MNAR em dados reais ruidosos, mas com viés para MAR (classe majoritária no L2 ~70%).

### 3. Consistente com ablação do plano_2
A ablação mostrou: E1 (6 features) = 49.5% > E3 (21 features) = 40.3% em dados reais. Adicionar features piora com N pequeno + labels ruidosos. As 4 CAAFE são a exceção porque capturam sinais de "forma" robustos ao ruído.

### 4. Divergência de imputação: sinal fraco em dados reais
Em dados reais com missing rate 1-10%, as 3 imputações (median, KNN, MICE) concordam bastante — a divergência é mínima para ambas as classes.

---

## Conclusão

| Aspecto | Resultado |
|---------|:---------:|
| Features ADV melhoram L2? | **Não** |
| Features ADV melhoram accuracy? | **Não** (-2-3pp) |
| Features ADV preservam MNAR recall? | **Não** (0%) |
| Vale incluir no paper? | **Sim, como resultado negativo** (confirma que features adicionais pioram com labels ruidosos) |

**Decisão: Não usar features ADV L2. Manter CAAFE (4 features) como set final para L2.**

---

## Achado Secundário: XGBoost e CatBoost (Step 02)

Os 9 modelos incluem XGBoost e CatBoost (implementados no Step 02). No LOGO CV:

| Modelo | LOGO CV (hard) |
|--------|:--------------:|
| **NaiveBayes** | **51.4%** |
| LogisticRegression | 44.7% |
| RandomForest | 39.4% |
| GradientBoosting | 38.2% |
| **XGBoost** | 38.2% |
| **CatBoost** | 37.5% |
| KNN | 36.8% |
| MLP | 34.2% |
| SVM_RBF | 31.4% |

**XGBoost e CatBoost não superam GBT no LOGO CV.** NaiveBayes continua significativamente melhor (+12pp sobre o 2º lugar).

---

## Arquivos

```
Scripts/v2_improved/features/advanced_l2.py         # 7 features implementadas
Output/v2_improved/step03_advl2/real/
├── apenas_ml/baseline/                             # 21 features (baseline)
│   ├── X_features.csv
│   ├── y_labels.csv
│   └── groups.csv
├── ml_com_llm/advl2/                               # 32 features (baseline + CAAFE + ADV)
│   ├── X_features.csv
│   ├── y_labels.csv
│   └── groups.csv
└── hierarquico_v3plus/
    ├── todas_variantes_v3plus.csv                  # 9 variantes × 9 modelos
    ├── cv_logo_v3plus.csv                          # LOGO CV (V3/V3plus)
    ├── resumo_v3plus.csv
    ├── significancia_v3plus.csv
    └── training_summary.json
```
