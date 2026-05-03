# STEP 01: Outputs Enriquecidos

**Fase 4A — Prerequisito para todas as demais fases**

---

## Problema

Os outputs atuais nao permitem analise rigorosa:
- Metricas por classe so em PNG (nao processaveis)
- Predicoes do teste nao salvas (impossivel analise de erro)
- Feature importance so top 20 em texto
- CV so media±std (sem scores por fold)
- Hiperparametros nao registrados
- Feature selection remove features silenciosamente

## Objetivo

Todo resultado do pipeline deve ser salvo em formato estruturado (CSV/JSON) para possibilitar comparacao automatica entre experimentos.

---

## Mudancas em `train_model.py`

### Novos arquivos a gerar

| Arquivo | O que contem | Por que e necessario |
|---------|-------------|---------------------|
| `predictions.csv` | Para cada amostra de teste: indice, grupo, y_true, y_pred, probabilidades por classe, confianca, nome do modelo | Permite analise de erro pos-hoc: quais amostras sao dificeis? Quais grupos falham? Permite ajustar thresholds sem re-treinar |
| `metrics_per_class.csv` | Para cada modelo x classe: precision, recall, f1, support | Comparacao programatica entre experimentos. Atualmente so existe em PNG |
| `feature_importance.csv` | Todas as features com RF importance (nao so top 20) | Rastrear como importancia muda entre experimentos |
| `cv_scores.csv` | Score de cada fold individual por modelo | Entender variancia entre folds, identificar folds problematicos |
| `confusion_matrices.json` | Matriz 3x3 por modelo | Analise de confusao entre classes especificas (ex: MNAR confundido com MAR) |
| `hyperparameters.json` | Parametros efetivos de cada modelo | Reprodutibilidade — saber exatamente o que foi treinado |
| `feature_selection_log.json` | Features selecionadas, removidas, scores e p-valores do SelectKBest | Saber o que foi descartado e por que |
| `training_summary.json` | Metadata: n_train, n_test, n_features, grupos treino/teste, metodo de split, metodo de CV, seed, timestamp | Contexto completo da execucao |

### Mudanca necessaria no SVC

O SVC atualmente nao tem `probability=True`, o que impede gerar probabilidades por classe. Precisa ativar isso para que predictions.csv tenha prob_MCAR/MAR/MNAR.

### Logica de implementacao

1. Apos treinar cada modelo, chamar `predict_proba(X_test)` e acumular predicoes em lista
2. Extrair classification_report com `output_dict=True` (ja faz isso) e serializar para CSV
3. Salvar `feature_importance` DataFrame completo (ja existe na linha 258, falta `.to_csv()`)
4. No loop de CV, guardar array de scores por fold em vez de so media
5. Serializar confusion_matrix e hiperparametros para JSON
6. Dentro do bloco de feature selection, logar o que foi removido e os scores

---

## Mudancas em `ensemble_model.py`

### Novo arquivo: `ensemble_decisions.csv`

| Coluna | Significado |
|--------|------------|
| sample_idx | Indice da amostra |
| group | Dataset de origem |
| y_true | Label real |
| modelo | Nome do classificador |
| pred_baseline | Predicao do modelo baseline |
| conf_baseline | Confianca maxima do baseline |
| pred_llm | Predicao do modelo LLM |
| pred_ensemble | Predicao final do ensemble |
| switched_to_llm | Se a amostra usou LLM (confianca < 0.6) |

**Por que:** Entender quais amostras trigaram o fallback para LLM e se o switch melhorou ou piorou.

**Logica:** Ja tem todas as variaveis computadas no loop (pred_bl, max_conf, pred_llm, pred_ensemble, low_conf_mask). Falta acumular por amostra em vez de so agregar.

---

## Mudancas em `analyze_feature_relevance.py`

Atualmente faz analises extensas mas so printa no console. Salvar em CSV:

| Arquivo | Conteudo |
|---------|----------|
| `variance_analysis.csv` | Por feature: media, std, min, max, nunique, % zeros, coeficiente de variacao |
| `correlation_matrix.csv` | Matriz de correlacao entre todas as features |
| `ablation_results.csv` | Accuracy com todas features, sem LLM, com top 15 |
| `llm_feature_analysis.csv` | Por feature LLM: media por classe, ANOVA F e p, correlacao com target |

**Logica:** Os DataFrames ja sao computados internamente. Adicionar `.to_csv()` em cada um.

---

## Testes de Validacao

### Teste 1: Completude dos outputs
Rodar pipeline sintetico baseline. Verificar que todos os 8 novos arquivos existem no diretorio de output. Nenhum deve estar vazio.

### Teste 2: Integridade estrutural
- `predictions.csv` deve ter 7 modelos x N_teste linhas
- `metrics_per_class.csv` deve ter 7 modelos x 3 classes = 21 linhas
- `cv_scores.csv` deve ter 7 modelos x N_folds linhas
- `confusion_matrices.json` deve ter 7 entradas, cada uma com matriz 3x3
- `training_summary.json` deve ter grupos treino e teste sem overlap

### Teste 3: Reprodutibilidade
Rodar o pipeline 2x com mesma seed. Os CSVs devem ser identicos (diff = 0).

### Teste 4: Nao-regressao
Comparar `relatorio.txt` antes e depois das mudancas. Accuracies devem ser identicas — as mudancas sao apenas de output adicional, nao alteram logica.

### Teste 5: Ensemble decisions
Rodar ensemble e verificar que `ensemble_decisions.csv` tem entradas com `switched_to_llm=True` (senao o threshold esta muito alto ou muito baixo).

---

## Criterio de Conclusao

- [x] 8 novos arquivos gerados por `train_model.py`
- [x] `ensemble_decisions.csv` gerado por `ensemble_model.py`
- [x] 4 CSVs adicionais gerados por `analyze_feature_relevance.py`
- [x] Testes de integridade passam (predictions=7xN_test, metrics=21, cv_scores=7xN_folds, confusion=7x3x3, groups sem overlap)
- [x] Resultados existentes nao mudam (nao-regressao) — apenas outputs adicionais, logica de treino inalterada
