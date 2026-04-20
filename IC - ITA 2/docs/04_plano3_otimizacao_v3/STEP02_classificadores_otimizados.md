# STEP 02: Classificadores Otimizados — XGBoost/CatBoost + Optuna

**Impacto esperado:** +3-8pp accuracy
**Esforço:** Baixo-Médio (~100 linhas)
**Dependências:** Nenhuma
**Pacotes:** `xgboost`, `catboost`, `optuna`

---

## Motivação

O pipeline atual usa `sklearn.ensemble.GradientBoostingClassifier` com hiperparâmetros fixos (dois regimes: n<100 e n>=100). Problemas:

1. **sklearn GBT é o mais lento e menos preciso** dos gradient boosting libraries
2. **Hiperparâmetros não otimizados** para o problema específico — valores genéricos
3. **CatBoost tem "ordered boosting"** que previne overfitting em datasets pequenos (~300 amostras no L2)
4. **XGBoost tem regularização L1/L2** nativa que ajuda com features redundantes

### Benchmark de Referência

| Library | Vantagem | Melhor Para |
|---------|----------|-------------|
| sklearn GBT | Simples, integrado | Prototyping |
| **XGBoost** | Regularização L1/L2, pruning | Datasets médios, features redundantes |
| **CatBoost** | Ordered boosting, anti-overfitting | **Datasets pequenos (nosso caso)** |
| LightGBM | Mais rápido, leaf-wise | Datasets grandes (>10K) — não nosso caso |

---

## Implementação

### 1. Novos Classificadores em `get_modelos()`

```python
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

def get_modelos_v3plus(n_samples: int) -> dict:
    modelos = get_modelos(n_samples)  # Mantém os 7 originais
    
    modelos["XGBoost"] = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,      # L1 regularization
        reg_lambda=1.0,     # L2 regularization
        min_child_weight=3,
        random_state=42,
        use_label_encoder=False,
        eval_metric='mlogloss',
        verbosity=0,
    )
    
    modelos["CatBoost"] = CatBoostClassifier(
        iterations=300,
        depth=4,
        learning_rate=0.05,
        l2_leaf_reg=3.0,
        random_seed=42,
        verbose=0,
        auto_class_weights='Balanced',  # Lida com desbalanceamento
    )
    
    return modelos
```

### 2. Optuna para Tuning por Nível

```python
import optuna

def optimize_level(X_train, y_train, groups_train, level_name, n_trials=100):
    """Otimiza hiperparâmetros para um nível específico da hierarquia."""
    
    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 500),
            'max_depth': trial.suggest_int('max_depth', 2, 8),
            'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        }
        
        clf = XGBClassifier(**params, random_state=42, verbosity=0)
        
        # CV interno com GroupKFold (se groups disponíveis)
        if groups_train is not None:
            cv = GroupKFold(n_splits=5)
            scores = cross_val_score(clf, X_train, y_train, 
                                     cv=cv, groups=groups_train, scoring='accuracy')
        else:
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring='accuracy')
        
        return scores.mean()
    
    study = optuna.create_study(direction='maximize', 
                                 study_name=f'optimize_{level_name}')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    return study.best_params
```

### 3. Protocolo de Tuning

```
L1 (MCAR vs não-MCAR):
  - X_train: ~750 amostras (75% de 1000)
  - Target: binário
  - CV: GroupKFold(5)
  - n_trials: 100

L2 (MAR vs MNAR):
  - X_train: ~300 amostras (excluindo MCAR do treino)
  - Target: binário
  - CV: GroupKFold(5) ou StratifiedKFold(5) se poucos grupos
  - n_trials: 100
```

---

## Integração

### Flag `--optimize` em `train_hierarchical_v3plus.py`:

```python
parser.add_argument("--optimize", action="store_true",
                    help="Usar Optuna para otimizar hiperparâmetros (lento, ~10min)")
parser.add_argument("--n-trials", type=int, default=100,
                    help="Número de trials Optuna por nível")

if args.optimize:
    print("🔍 Otimizando hiperparâmetros com Optuna...")
    params_l1 = optimize_level(X_train_l1, y_train_l1, groups_train, "L1", args.n_trials)
    params_l2 = optimize_level(X_train_l2, y_train_l2, groups_train_l2, "L2", args.n_trials)
    # Salvar params para reprodutibilidade
    json.dump({"L1": params_l1, "L2": params_l2}, open(f"{HIER_DIR}/optuna_params.json", "w"))
else:
    # Usar defaults
    params_l1 = DEFAULT_PARAMS_L1
    params_l2 = DEFAULT_PARAMS_L2
```

---

## Output Esperado

```
Output/v2_improved/{experiment}/real/hierarquico_v3plus/
├── optuna_params.json           # Melhores hiperparâmetros L1 e L2
├── optuna_history_l1.csv        # Histórico de trials L1
├── optuna_history_l2.csv        # Histórico de trials L2
├── optuna_importance_l1.png     # Importância dos hiperparâmetros L1
├── optuna_importance_l2.png     # Importância dos hiperparâmetros L2
└── model_comparison.csv         # sklearn GBT vs XGBoost vs CatBoost
```

---

## Validação

1. Comparar sklearn GBT vs XGBoost vs CatBoost **com mesmos hiperparâmetros default** → isolar efeito da library
2. Comparar default vs Optuna-tuned → isolar efeito do tuning
3. Verificar no LOGO CV que tuning não é overfitting ao holdout split
4. Tempo de execução: Optuna com 100 trials deve rodar em <10min por nível

---

## Riscos

- **Optuna pode overfittar ao CV interno** se n_trials muito alto → limitar a 100-200 trials
- **CatBoost pode ser lento** com ordered boosting → usar `task_type='CPU'`, `boosting_type='Plain'` se lento
- **Diferentes libraries podem dar resultados não-reproduzíveis** → fixar seeds, salvar params

---

# Anexo: Resultados do Experimento

> Originalmente publicado como `RESULTADOS_STEP02.md`. Consolidado aqui em 2026-04-19.


**Data:** 2026-04-19
**Experimento:** step05_pro
**Status:** CONCLUIDO

---

## Resumo Executivo

**Resultado: XGBoost e CatBoost NAO melhoram o pipeline hierarquico.**

| Metrica | Antes (GBT sklearn) | XGBoost (Optuna) | CatBoost (defaults) |
|---------|:-------------------:|:----------------:|:-------------------:|
| **Holdout (hard)** | **52.9%** (GBT) | 50.8% | 44.7% |
| **Holdout (threshold)** | **53.2%** (GBT) | 65.4%* | 44.4% |
| **LOGO CV (hard)** | 38.3% | **38.7%** | 36.4% |
| **LOGO CV (soft3zone)** | 35.8% | 38.0% | 36.6% |
| **MNAR recall (hard)** | **44.0%** (GBT) | 0.0% | 42.0% |

*XGBoost threshold 65.4% accuracy é ENGANOSO: MNAR recall = 0.0% (classifica tudo como MAR).

---

## Optuna — Hiperparametros Otimizados

### Nivel 1 (MCAR vs nao-MCAR)

| Classificador | Melhor CV Accuracy | Params Chave |
|--------------|:------------------:|-------------|
| **XGBoost** | **85.7%** | n_est=478, lr=0.0014, depth=4, reg_alpha=9.9 |
| CatBoost | 81.9% | iter=262, lr=0.102, depth=4, l2_reg=0.11 |

### Nivel 2 (MAR vs MNAR)

| Classificador | Melhor CV Accuracy | Params Chave |
|--------------|:------------------:|-------------|
| **CatBoost** | **62.2%** | iter=139, lr=0.0016, depth=2, random_strength=9.9 |
| XGBoost | 57.1% | n_est=101, lr=0.001, depth=8, min_child=10 |

**Nota:** L2 accuracy ~57-62% confirma que MAR vs MNAR e intrinsecamente dificil.

---

## Holdout (295 amostras) — com pesos Cleanlab

### V3 hard routing

| Modelo | Accuracy | MNAR Recall | F1 Macro | Acc L1 | Acc L2 |
|--------|:--------:|:-----------:|:--------:|:------:|:------:|
| **GradientBoosting** | **52.9%** | **44.0%** | **0.509** | 82.4% | 56.3% |
| XGBoost | 50.8% | 0.0% | 0.225 | 67.8% | 75.0% |
| SVM_RBF | 50.5% | 32.0% | 0.468 | 81.0% | 55.0% |
| LogisticRegression | 49.2% | 18.0% | 0.440 | 81.4% | 52.5% |
| NaiveBayes | 47.1% | 52.0% | 0.466 | 80.3% | 51.0% |
| CatBoost | 44.7% | 42.0% | 0.458 | 82.0% | 44.2% |
| RandomForest | 44.1% | 18.0% | 0.418 | 82.7% | 43.0% |
| KNN | 43.4% | 34.0% | 0.406 | 76.9% | 49.5% |
| MLP | 40.7% | 36.0% | 0.406 | 79.7% | 41.6% |

### V3+ threshold routing

| Modelo | Accuracy | MNAR Recall | F1 Macro |
|--------|:--------:|:-----------:|:--------:|
| **XGBoost** | **65.4%** | **0.0%** | 0.456 |
| GradientBoosting | 53.2% | 46.0% | 0.515 |
| LogisticRegression | 49.2% | 18.0% | 0.440 |
| NaiveBayes | 48.5% | 52.0% | 0.483 |
| SVM_RBF | 48.5% | 28.0% | 0.440 |
| CatBoost | 44.4% | 42.0% | 0.455 |
| RandomForest | 44.1% | 18.0% | 0.418 |
| KNN | 43.7% | 34.0% | 0.410 |
| MLP | 39.7% | 36.0% | 0.396 |

**XGBoost threshold anomalia:** 65.4% accuracy mas MNAR=0% e F1=0.456. O Optuna L2 otimizou para accuracy
que com classes desbalanceadas (~70% MAR / 30% MNAR) resulta em classificar tudo como MAR. O threshold
otimizado no L1 + L2 enviesado cria um pipeline que "acerta" MAR (99.3% recall!) mas ignora MNAR.

---

## LOGO CV (23 folds) — Resultado Definitivo

| Modelo | V3 hard | V3+ threshold | V3+ soft3zone | V3+ fullprob |
|--------|:-------:|:-------------:|:-------------:|:------------:|
| **NaiveBayes** | **51.4%** | 51.7% | **56.0%** | **56.0%** |
| LogisticRegression | 44.0% | 44.4% | 45.3% | **45.4%** |
| **XGBoost** | **38.7%** | 38.4% | 38.0% | 38.0% |
| KNN | 37.1% | 37.2% | **38.5%** | **38.5%** |
| RandomForest | **38.3%** | 37.5% | 36.9% | 36.9% |
| GradientBoosting | **38.3%** | 38.1% | 35.8% | 35.8% |
| **CatBoost** | **36.4%** | 36.0% | **36.6%** | **36.6%** |
| MLP | 31.6% | 31.5% | **33.2%** | **33.2%** |
| SVM_RBF | **31.8%** | 31.7% | 24.9% | 24.8% |

### Comparacao direta: novos vs existentes (LOGO CV, melhor routing)

| Modelo | LOGO CV | Ranking |
|--------|:-------:|:-------:|
| **NaiveBayes** | **56.0%** | 1 |
| LogisticRegression | 45.4% | 2 |
| **XGBoost** | **38.7%** | 3 |
| KNN | 38.5% | 4 |
| RandomForest | 38.3% | 5 |
| GradientBoosting | 38.3% | 6 |
| **CatBoost** | **36.6%** | 7 |
| MLP | 33.2% | 8 |
| SVM_RBF | 31.8% | 9 |

**XGBoost fica em 3o lugar no LOGO CV (38.7%), empatado com RandomForest e GBT.**
**CatBoost fica em 7o lugar (36.6%), pior que GBT (38.3%).**

---

## McNemar: XGBoost threshold vs V3 hard

| Modelo | b (hard>thr) | c (thr>hard) | p-value | Sig? |
|--------|:--:|:--:|:-------:|:----:|
| **XGBoost** | 1 | 44 | **<0.0001** | *** |

XGBoost threshold e significativamente diferente de hard — mas na direcao errada (sacrifica MNAR).

---

## Analise

### Por que XGBoost/CatBoost nao melhoram?

1. **Dataset pequeno (~1132 amostras, 23 grupos):** CatBoost/XGBoost brilham com >10K amostras. Com ~1000, overfitting e o risco principal.
2. **Optuna overfittou L2 para accuracy:** Com ~300 amostras nao-MCAR (70% MAR, 30% MNAR), otimizar accuracy = classificar tudo como MAR. O XGBoost tuned aprende a "ignorar" MNAR.
3. **LOGO CV alto desvio padrao (~0.34):** Todos os modelos tem std > mean, indicando que o resultado depende fortemente de qual grupo e excluido. Nenhum modelo generaliza consistentemente.
4. **NaiveBayes domina porque nao overfita:** Com features gaussianas e poucas amostras, a simplicidade do NaiveBayes e uma vantagem real, nao coincidencia.
5. **GBT sklearn ja e suficiente:** A diferenca entre GBT, XGBoost e CatBoost e minima (<2pp) neste regime de dados. O gargalo nao e o classificador.

### Conclusao

O Step 02 confirma que **o gargalo NAO e o classificador**. Trocar GBT por XGBoost/CatBoost com Optuna nao melhora e pode piorar (MNAR recall = 0%). Os proximos steps devem focar em:
- **Features melhores** (Step 03: novas features L2)
- **Feature selection** (Step 05: remover features redundantes)

---

## Arquivos Gerados

```
Output/v2_improved/step05_pro/real/hierarquico_v3plus/
├── optuna_params.json                  # Melhores params por nivel/classificador
├── optuna_history_l1_xgboost.csv       # 100 trials XGBoost L1
├── optuna_history_l1_catboost.csv      # 100 trials CatBoost L1
├── optuna_history_l2_xgboost.csv       # 100 trials XGBoost L2
├── optuna_history_l2_catboost.csv      # 100 trials CatBoost L2
├── todas_variantes_v3plus.csv          # 5 variantes × 9 modelos (atualizado)
├── resumo_v3plus.csv                   # Resumo por variante
├── cv_logo_v3plus.csv                  # LOGO CV com 9 modelos (atualizado)
├── significancia_v3plus.csv            # McNemar com 9 modelos
├── heatmap_v3plus.png                  # Heatmap atualizado
├── v3plus_comparison.png               # Barras comparativas
└── training_summary.json               # Metadata
```

---

## Recomendacao

**Manter GradientBoosting (sklearn) como classificador principal.** XGBoost e CatBoost podem ser mencionados no paper como experimento negativo, mostrando que o gargalo e features/labels, nao o classificador.

**Proximo step: Step 03 (Novas Features L2)** — o verdadeiro gargalo e o L2 (MAR vs MNAR) com accuracy ~53-62%. Novas features discriminativas para L2 tem mais potencial que classificadores mais potentes.
