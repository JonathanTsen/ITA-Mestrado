# STEP 05: Feature Selection Adaptativa

> **DESCARTADO** — Este step dependia do STEP03 (Novas Features L2) produzir ganho. Como
> o STEP03 piorou accuracy (-2pp) e zerou MNAR recall, o STEP05 foi descartado sem execucao.
> Mantido aqui como registro do plano original. Ver [../VISAO_GERAL.md](../VISAO_GERAL.md).

**Impacto esperado:** +1-5pp accuracy
**Esforço:** Baixo (~40 linhas)
**Dependências:** Step 03 (novas features disponíveis)
**Pacotes:** `boruta` (opcional)

---

## Motivação

A ablação do plano_2 revelou um padrão surpreendente em dados reais:

| Config | N features | Accuracy (max) |
|--------|:----------:|:--------------:|
| **E1** | **6** | **49.5%** |
| E2 | 15 | 39.7% |
| E3 | 21 | 40.3% |
| E4 | 25 | 43.4% |

**Menos features = melhor em dados reais!** As 6 features discriminativas (E1) superam 21 features baseline. Adicionar features estatísticas e MechDetect **piora** (-9.2pp). Apenas CAAFE recupera (+3.1pp sobre E3).

Isso indica que muitas features são **ruidosas** em dados reais e o classificador não consegue separar sinal de ruído com N pequeno. Precisamos de feature selection automática.

---

## Abordagem: Seleção por Nível

A seleção deve ser **diferente para L1 e L2**:

- **L1:** Já funciona bem (82%) com 21 features → seleção agressiva pode ajudar marginalmente
- **L2:** Gargalo principal → seleção é crítica (25-32 features para ~300 amostras)

---

## Método 1: Boruta (Recomendado)

Boruta é "all-relevant" feature selection: identifica **todas** as features genuinamente úteis (não apenas as top K).

```python
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier

def select_features_boruta(X_train, y_train, max_iter=100):
    """Seleciona features relevantes usando Boruta."""
    rf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42, n_jobs=-1)
    boruta = BorutaPy(rf, n_estimators='auto', max_iter=max_iter, random_state=42)
    boruta.fit(X_train.values, y_train.values)
    
    selected = X_train.columns[boruta.support_].tolist()
    tentative = X_train.columns[boruta.support_weak_].tolist()
    
    return selected, tentative
```

### Integração:

```python
if args.feature_selection:
    # L1
    selected_l1, _ = select_features_boruta(X_tr_l1, y_tr_l1)
    feat_l1 = selected_l1 if len(selected_l1) >= 3 else FEAT_STAT[:6]  # fallback
    
    # L2
    selected_l2, tentative_l2 = select_features_boruta(X_tr_l2, y_tr_l2)
    feat_l2 = selected_l2 + tentative_l2  # Incluir tentativas no L2 (mais conservador)
```

---

## Método 2: Mutual Information + Threshold (Alternativa sem pacote extra)

```python
from sklearn.feature_selection import mutual_info_classif

def select_features_mi(X_train, y_train, threshold=0.01):
    """Seleciona features com MI > threshold."""
    mi = mutual_info_classif(X_train, y_train, random_state=42)
    selected = X_train.columns[mi > threshold].tolist()
    return selected, mi
```

---

## Método 3: Drop-Column Importance (Mais robusto, mais lento)

```python
def select_features_dropcol(X_train, y_train, base_model, cv):
    """Remove features cuja remoção não piora accuracy."""
    from sklearn.model_selection import cross_val_score
    
    base_score = cross_val_score(base_model, X_train, y_train, cv=cv).mean()
    important = []
    
    for col in X_train.columns:
        X_dropped = X_train.drop(columns=[col])
        score = cross_val_score(base_model, X_dropped, y_train, cv=cv).mean()
        if score < base_score - 0.005:  # Feature é útil (remoção piora >0.5pp)
            important.append(col)
    
    return important
```

---

## Protocolo

1. Rodar Boruta separadamente para L1 e L2
2. Salvar features selecionadas em JSON para reprodutibilidade
3. Treinar V3+ com features selecionadas vs features completas
4. Se Boruta selecionar <5 features para L2, usar MI como fallback

---

## Output Esperado

```
Output/v2_improved/{experiment}/real/feature_selection/
├── boruta_l1_results.csv       # Ranking + confirmado/rejeitado para L1
├── boruta_l2_results.csv       # Ranking + confirmado/rejeitado para L2
├── mi_scores.csv               # Mutual information scores
├── selected_features.json      # Features selecionadas por nível
└── comparison_selected_vs_all.csv  # Accuracy com vs sem seleção
```

---

## Validação

1. Verificar que features CAAFE são selecionadas para L2 (sanity check — sabemos que importam)
2. Comparar accuracy: V3+(todas features) vs V3+(features selecionadas)
3. Verificar LOGO CV (seleção não deve overfittar)
4. Contar: quantas features são removidas? Se <3, seleção não ajuda muito

---

## Riscos

- **Boruta pode ser instável** com N pequeno → repetir 3x com seeds diferentes, usar interseção
- **Feature selection no treino pode não generalizar** → sempre avaliar no teste/LOGO CV
- **Remover features demais** pode perder informação complementar → usar threshold conservador
