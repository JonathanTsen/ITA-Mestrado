# STEP 01: Cleanlab — Limpeza de Labels Ruidosos

**Status:** ✅ CONCLUÍDO (2026-04-18)
**Resultado:** 59.4% labels problemáticos (672/1132), 15/23 datasets discordantes
**Esforço real:** ~200 linhas (`clean_labels.py`)
**Pacotes:** `cleanlab==2.9.0`

> **Resultados experimentais:** ver [RESULTADOS_STEP01_STEP04.md](RESULTADOS_STEP01_STEP04.md)
> (relatorio combinado STEP01+STEP04, pois as duas tecnicas foram avaliadas em conjunto).

---

## Motivação

O `validar_rotulos.py` revelou que **57% dos labels reais são inconsistentes** com testes estatísticos (Little's MCAR, correlação point-biserial, KS test). Exemplos:

| Dataset | Label Original | Diagnóstico Estatístico | Problema |
|---------|:--------------:|:-----------------------:|----------|
| MCAR_hypothyroid_t4u | MCAR | KS p<0.001, censoring_score alto | Parece MNAR (distribuição assimétrica) |
| MAR_sick_tsh | MAR | Sem correlação X1-mask | Parece MCAR |
| MNAR_colic_refluxph | MNAR | MNAR Diffuse | Difícil de distinguir de MAR |

O modelo está tentando aprender padrões de labels que contradizem a evidência estatística. Cleanlab identifica esses labels problemáticos e permite treinar num subset mais limpo.

---

## Como Funciona o Cleanlab

1. Treina o classificador com cross-validation (K-fold)
2. Obtém probabilidades out-of-sample para cada amostra
3. Usa **confident learning** para identificar labels que contradizem as predições do modelo de forma consistente
4. Opções: (a) remover amostras ruidosas, (b) re-rotular, (c) atribuir pesos menores

### Teoria: Confident Joint

A "confident joint" estima P(y_true=j, y_given=i) — a probabilidade conjunta de que o label verdadeiro seja j mas o label dado seja i. Amostras onde y_given diverge de y_true são candidatas a label errado.

---

## Implementação

### Arquivo: `v2_improved/clean_labels.py`

```python
"""
Limpeza de labels ruidosos usando Cleanlab (Confident Learning).

Identifica labels potencialmente incorretos nos dados reais e gera:
1. Relatório de qualidade por dataset
2. Labels corrigidos (y_labels_clean.csv)
3. Scores de qualidade por amostra (label_quality.csv)

Uso:
    python clean_labels.py --experiment step05_pro
    python clean_labels.py --experiment step05_pro --action prune   # Remove labels ruins
    python clean_labels.py --experiment step05_pro --action weight  # Atribui pesos
"""
```

### Lógica Principal

```python
from cleanlab.classification import CleanLearning
from cleanlab.rank import get_label_quality_scores
from sklearn.ensemble import GradientBoostingClassifier

# 1. Carregar features e labels
X = pd.read_csv(f"{BASELINE_DIR}/X_features.csv")
y = pd.read_csv(f"{BASELINE_DIR}/y_labels.csv").squeeze()
groups = pd.read_csv(f"{BASELINE_DIR}/groups.csv").squeeze()

# 2. Obter probabilidades out-of-sample (GroupKFold para não vazar)
from sklearn.model_selection import cross_val_predict, GroupKFold
clf = GradientBoostingClassifier(n_estimators=300, random_state=42)
pred_probs = cross_val_predict(clf, X, y, cv=GroupKFold(5), groups=groups, method='predict_proba')

# 3. Identificar labels problemáticos
label_quality = get_label_quality_scores(y, pred_probs)
issues = find_label_issues(y, pred_probs, return_indices_ranked_by='self_confidence')

# 4. Gerar relatório por dataset
for dataset_name in groups.unique():
    mask = groups == dataset_name
    avg_quality = label_quality[mask].mean()
    n_issues = sum(issues[mask])
    # → Report: dataset, label, quality score, n_issues

# 5. Opção A: Prune (remover)
X_clean = X.drop(issues[:n_remove])
y_clean = y.drop(issues[:n_remove])

# 5. Opção B: Weight (ponderar)
sample_weights = label_quality  # Amostras com labels bons pesam mais
```

### Modos de Uso

| Modo | Ação | Quando Usar |
|------|------|-------------|
| `--action report` | Apenas gera relatório | Sempre (primeiro passo) |
| `--action prune` | Remove N% piores labels | Se N labels ruins > 20% |
| `--action weight` | Pondera amostras por qualidade | Se labels ruins estão espalhados |
| `--action relabel` | Usa predição do modelo como label | Se confiança alta na predição |

---

## Integração com Pipeline

### Em `train_hierarchical_v3plus.py`:

```python
# Flag --clean-labels ativa limpeza
if args.clean_labels:
    from clean_labels import get_clean_data
    X, y, sample_weights = get_clean_data(X, y, groups, mode=args.clean_mode)
    # sample_weights usado no fit: clf.fit(X, y, sample_weight=sample_weights)
```

---

## Output Esperado

```
Output/v2_improved/{experiment}/real/label_analysis/
├── label_quality_scores.csv     # Score por amostra (0-1)
├── label_issues.csv             # Amostras com labels suspeitos
├── quality_by_dataset.csv       # Qualidade média por dataset
├── quality_distribution.png     # Histograma de scores
├── confusion_labels.png         # Confident joint matrix
└── clean_training_report.json   # Resumo: N removidos, accuracy antes/depois
```

---

## Validação — Resultados

1. ✅ `clean_labels.py --action report` flaggeou corretamente MCAR_hypothyroid_t4u (quality=0.001), MAR_sick_tsh (0.024), MNAR_mroz_wages (0.029)
2. ⬜ Comparar V3 com labels limpos → pendente (combinar com Step 04 no próximo passo)
3. ✅ Pesos salvos (mode=weight), sem remoção de amostras — não afeta N de treino
4. ✅ GroupKFold(10) usado para cross-validation — sem leakage entre datasets

## Observações

- O modo `--action weight` é preferível ao `--action prune` porque não reduz o N de treino (L2 já tem poucos dados)
- Quality média por classe: MAR (0.466) > MNAR (0.317) > MCAR (0.268) — MCAR é o mais confuso
- Confident joint mostra: MAR é a classe "absorvente" — 215 MNAR → MAR, 109 MCAR → MAR

## Riscos Verificados

- ✅ GroupKFold evita circular reasoning
- ⚠️ Cleanlab flaggeia 59.4% como issues — alto mas consistente com 57% do validar_rotulos.py
- ⚠️ n_jobs=1 necessário no macOS (Python 3.14) para evitar multiprocessing spawn error
