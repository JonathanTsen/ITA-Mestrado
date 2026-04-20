# STEP 04: Roteamento Probabilístico L1→L2

**Status:** ✅ CONCLUÍDO (2026-04-18)
**Resultado:** +4.6pp LOGO CV (51.4% → 56.0%, NaiveBayes + soft3zone)
**Esforço real:** ~500 linhas (`train_hierarchical_v3plus.py`)
**Pacotes:** Nenhum novo

> **Resultados experimentais:** ver [RESULTADOS_STEP01_STEP04.md](RESULTADOS_STEP01_STEP04.md)
> (relatorio combinado STEP01+STEP04, pois as duas tecnicas foram avaliadas em conjunto).

---

## Motivação

Atualmente, o L1 faz predição **hard**: `pred_l1 = m_l1.predict(X)` retorna 0 ou 1. Isso significa:

1. Uma amostra com prob(não-MCAR)=0.51 é tratada igual a prob=0.99
2. **22% das amostras de L1 são classificadas errado** → esses erros propagam irrecuperavelmente para L2
3. Amostras MCAR erroneamente marcadas como não-MCAR vão para L2, que nunca viu MCAR → predição aleatória
4. Amostras não-MCAR erroneamente marcadas como MCAR nunca passam por L2 → recall de MAR/MNAR cai

### O Problema em Números (dados reais, V3)

| Cenário | N amostras | Accuracy no L2 |
|---------|:----------:|:--------------:|
| L1 correto E não-MCAR | ~180 | 53% |
| L1 errado → MCAR verdadeiro no L2 | ~20 | ~0% (modelo não sabe lidar) |
| L1 errado → não-MCAR ignorado | ~40 | — (nunca chega ao L2) |

---

## Solução: Soft Routing com Probabilidades Calibradas

### Conceito

Em vez de usar threshold fixo em 0.5:

```
prob(não-MCAR) < 0.35  → MCAR (alta confiança)
prob(não-MCAR) > 0.65  → L2 (alta confiança)
0.35 ≤ prob ≤ 0.65     → ZONA INCERTA → enviar para L2 com peso reduzido
```

### Três Abordagens (testar as 3)

#### Abordagem A: Threshold Otimizado
Encontrar threshold ótimo no treino que maximiza accuracy geral:

```python
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score

# Calibrar probabilidades
m_l1_cal = CalibratedClassifierCV(m_l1, method='sigmoid', cv=5)
m_l1_cal.fit(X_tr_l1_sm, y_tr_l1_sm)

# Achar threshold ótimo no treino (com validação interna)
best_threshold = 0.5
best_acc = 0
for threshold in np.arange(0.3, 0.7, 0.01):
    pred_l1_val = (m_l1_cal.predict_proba(X_val_l1)[:, 1] >= threshold).astype(int)
    # Simular pipeline hierárquico completo
    acc = evaluate_hierarchical(pred_l1_val, m_l2, X_val, y_val)
    if acc > best_acc:
        best_acc = acc
        best_threshold = threshold

# Aplicar no teste
probs_l1 = m_l1_cal.predict_proba(X_te_l1)[:, 1]
pred_l1 = (probs_l1 >= best_threshold).astype(int)
```

#### Abordagem B: Soft Voting com 3 Zonas
```python
probs_l1 = m_l1_cal.predict_proba(X_te_l1)[:, 1]

y_pred = np.zeros(len(y_te), dtype=int)

# Zona MCAR (alta confiança)
zone_mcar = probs_l1 < threshold_low  # e.g., 0.35
y_pred[zone_mcar] = 0

# Zona não-MCAR (alta confiança)
zone_notmcar = probs_l1 > threshold_high  # e.g., 0.65
if zone_notmcar.any():
    pred_l2 = m_l2.predict(X_te_l2[zone_notmcar])
    y_pred[zone_notmcar] = np.where(pred_l2 == 0, 1, 2)

# Zona incerta → considerar as 3 classes
zone_uncertain = ~zone_mcar & ~zone_notmcar
if zone_uncertain.any():
    prob_mcar = 1 - probs_l1[zone_uncertain]
    pred_l2_unc = m_l2.predict_proba(X_te_l2[zone_uncertain])
    prob_mar = probs_l1[zone_uncertain] * pred_l2_unc[:, 0]
    prob_mnar = probs_l1[zone_uncertain] * pred_l2_unc[:, 1]
    
    all_probs = np.column_stack([prob_mcar, prob_mar, prob_mnar])
    y_pred[zone_uncertain] = all_probs.argmax(axis=1)
```

#### Abordagem C: Full Probabilistic (sem zonas)
```python
# Sempre combinar probabilidades de ambos os níveis
prob_mcar = 1 - probs_l1  # P(MCAR)
prob_l2 = m_l2.predict_proba(X_te_l2)  # P(MAR|não-MCAR), P(MNAR|não-MCAR)
prob_mar = probs_l1 * prob_l2[:, 0]    # P(não-MCAR) × P(MAR|não-MCAR)
prob_mnar = probs_l1 * prob_l2[:, 1]   # P(não-MCAR) × P(MNAR|não-MCAR)

all_probs = np.column_stack([prob_mcar, prob_mar, prob_mnar])
y_pred = all_probs.argmax(axis=1)
```

---

## Implementação

### Modificar `run_hierarchical()`:

```python
def run_hierarchical(X_full, y, train_idx, test_idx, feat_l1, feat_l2, 
                     modelo_nome, routing="hard"):
    """
    routing: "hard" (original), "threshold", "soft3zone", "fullprob"
    """
    # ... treinar L1 e L2 como antes ...
    
    if routing == "hard":
        # Comportamento original
        pred_l1 = m_l1.predict(X_te_l1)
        # ...
    elif routing == "fullprob":
        # Abordagem C
        probs_l1 = m_l1.predict_proba(X_te_l1)[:, 1]
        prob_l2 = m_l2.predict_proba(X_te_l2)
        # Combinar probabilidades
        # ...
```

### Nova CLI flag:

```python
parser.add_argument("--routing", choices=["hard", "threshold", "soft3zone", "fullprob"],
                    default="hard", help="Estratégia de roteamento L1→L2")
```

---

## Calibração

**Importante:** `predict_proba()` de GradientBoosting e RandomForest são **mal calibradas** por default. Usar `CalibratedClassifierCV`:

```python
from sklearn.calibration import CalibratedClassifierCV

# Calibrar L1
m_l1_base = get_modelos(len(X_tr_l1_sm))[modelo_nome]
m_l1 = CalibratedClassifierCV(m_l1_base, method='sigmoid', cv=5)
m_l1.fit(X_tr_l1_sm, y_tr_l1_sm)

# Calibrar L2
m_l2_base = get_modelos(len(X_tr_l2_sm))[modelo_nome]
m_l2 = CalibratedClassifierCV(m_l2_base, method='sigmoid', cv=5)
m_l2.fit(X_tr_l2_sm, y_tr_l2_sm)
```

**Nota:** `method='sigmoid'` é melhor para datasets pequenos (menos parâmetros que 'isotonic').

---

## Validação — Resultados

1. ✅ 4 estratégias comparadas: hard, threshold, soft3zone, fullprob
2. ✅ LOGO CV (23 folds) confirma: soft3zone/fullprob +4.6pp para NaiveBayes
3. ⚠️ MNAR recall no holdout cai para soft3zone (28% vs 40% hard) — tradeoff accuracy vs recall
4. ⬜ Calibration curves não geradas (adicionar no futuro)
5. ✅ McNemar significativo para 3/7 modelos (LogReg p=0.002, MLP p=0.024, NaiveBayes p=0.016)

### Descobertas Chave

- **soft3zone = fullprob** para modelos com probabilidades bem calibradas (NaiveBayes, KNN, MLP)
- **threshold** é a estratégia mais conservadora — quase igual a hard mas com +4pp MNAR recall no GBT
- **GradientBoosting não melhora** com soft routing (probabilidades já são "confiantes", poucos casos na zona incerta)
- **SVM_RBF piora significativamente** com soft3zone/fullprob (calibração sigmoid instável com ~300 amostras L2)

### Risco Materializado

- ⚠️ Calibração instável para SVM_RBF: soft3zone derruba accuracy de 31.8% → 24.9% no LOGO CV
- Mitigação: usar routing por modelo (soft3zone para NaiveBayes/LogReg/KNN/MLP, hard para GBT/RF/SVM)
