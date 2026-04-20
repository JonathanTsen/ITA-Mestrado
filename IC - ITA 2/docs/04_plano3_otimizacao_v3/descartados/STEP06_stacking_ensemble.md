# STEP 06: Stacking Ensemble no L2

> **DESCARTADO** — Dependencia dos STEPs 02 (XGBoost/CatBoost) e 03 (novas features L2).
> Ambos resultaram em piora vs baseline (NaiveBayes). Sem candidatos uteis para o
> stacking, o STEP06 foi descartado sem execucao. Ver [../VISAO_GERAL.md](../VISAO_GERAL.md).

**Impacto esperado:** +2-5pp accuracy
**Esforço:** Baixo (~30 linhas)
**Dependências:** Steps 02 (novos classificadores) e 03 (novas features)
**Pacotes:** Nenhum novo (sklearn StackingClassifier)

---

## Motivação

Cada classificador captura diferentes padrões:
- **GradientBoosting:** Interações complexas, bom com poucas features
- **RandomForest:** Robusto a ruído, captura não-linearidades
- **SVM_RBF:** Bom em fronteiras complexas com poucos dados
- **LogisticRegression:** Captura relações lineares, regularizado

No L2 (MAR vs MNAR), nenhum modelo domina sozinho em todos os datasets. Stacking combina as forças de cada um.

### Evidência da Exploração

| Modelo | V3 Accuracy (real) | MNAR Recall |
|--------|:------------------:|:-----------:|
| GradientBoosting | **50.5%** | 40.0% |
| KNN | 41.7% | **38.0%** |
| NaiveBayes | 41.7% | 34.0% |
| MLP | 39.7% | **40.0%** |

Modelos diferentes se destacam em métricas diferentes → stacking pode extrair o melhor de cada.

---

## Implementação

```python
from sklearn.ensemble import StackingClassifier

def create_stacking_l2(n_samples):
    """Cria stacking ensemble para o L2."""
    base_estimators = [
        ('gbt', GradientBoostingClassifier(n_estimators=300, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1)),
        ('svm', Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(kernel='rbf', C=3, probability=True, random_state=42))
        ])),
        ('knn', Pipeline([
            ('scaler', StandardScaler()),
            ('clf', KNeighborsClassifier(n_neighbors=5))
        ])),
    ]
    
    # Meta-learner simples (evita overfitting)
    meta_learner = LogisticRegression(C=1.0, random_state=42, max_iter=1000)
    
    stacking = StackingClassifier(
        estimators=base_estimators,
        final_estimator=meta_learner,
        cv=5,                    # CV interna para gerar meta-features
        stack_method='predict_proba',  # Usar probabilidades, não classes
        passthrough=False,        # Não passar features originais ao meta-learner
    )
    
    return stacking
```

### Integração no pipeline hierárquico:

```python
# Em run_hierarchical(), substituir m_l2 por stacking quando solicitado
if use_stacking:
    m_l2 = create_stacking_l2(len(X_tr_l2_sm))
else:
    m_l2 = get_modelos(len(X_tr_l2_sm))[modelo_nome]

m_l2.fit(X_tr_l2_sm, y_tr_l2_sm)
```

---

## Variantes a Testar

| Variante | Base Models | Meta-Learner | Passthrough |
|----------|-------------|:------------:|:-----------:|
| Stack-A | GBT + RF + SVM + KNN | LogReg | Não |
| Stack-B | GBT + RF + SVM + KNN | LogReg | Sim (features originais) |
| Stack-C | XGBoost + CatBoost + RF + SVM | LogReg | Não |
| Stack-D | GBT + RF + NaiveBayes + MLP | LogReg | Não |

**Nota:** Passthrough=True pode overfittar com N pequeno (300 amostras + 25 features + 4 meta-features = 29 inputs para meta-learner com 300 amostras). Preferir passthrough=False.

---

## Stacking Completo: L1 + L2

Opcionalmente, usar stacking também no L1 (benefício menor, já está em 82%):

```python
# Stacking no L1 (opcional)
m_l1 = create_stacking_l1(len(X_tr_l1_sm))  # GBT + RF + SVM
m_l1.fit(X_tr_l1_sm, y_tr_l1_sm)

# Stacking no L2 (principal)
m_l2 = create_stacking_l2(len(X_tr_l2_sm))
m_l2.fit(X_tr_l2_sm, y_tr_l2_sm)
```

---

## Validação

1. Comparar: melhor modelo individual vs Stack-A/B/C/D no L2
2. Medir: accuracy, MNAR recall, F1 macro
3. LOGO CV: verificar que stacking generaliza (risco de overfitting com CV interna)
4. Tempo de treino: stacking com 4 modelos × 5 CV folds = 20 fits → deve ser <1min

---

## Riscos

- **Overfitting com N pequeno:** 300 amostras, 4 base models com CV=5 → cada fold treina com ~240 amostras. Se piorar no LOGO CV vs holdout, reduzir para 3 base models ou cv=3.
- **Stacking pode homogeneizar predições:** Se todos os base models concordam, stacking não adiciona valor. Verificar diversidade dos base models.
