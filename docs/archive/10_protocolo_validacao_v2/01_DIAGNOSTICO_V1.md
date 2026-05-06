# Diagnóstico do protocolo v1 de validação de rótulos

**Data:** 2026-05-03
**Arquivo analisado:** `src/missdetect/validar_rotulos.py`

---

## Contexto

O protocolo v1 usava 3 testes estatísticos para verificar se os rótulos de domínio (MCAR/MAR/MNAR) dos 29 datasets reais eram coerentes com evidência empírica. O resultado era que **57% dos rótulos discordavam de pelo menos um teste** — em particular, **6 de 7 MNAR testavam como MCAR**.

A questão é: quanto dessa inconsistência é limitação teórica (MNAR é não-identificável — Molenberghs 2008) e quanto é fragilidade dos testes?

---

## Problema 1 — Little's MCAR test é frágil em dados reais

**Função:** `test_little_mcar` → `MCARTest.little_mcar_test` (MissMecha) + fallback `_little_proxy`.

**Limitações:**
- Assume **normalidade multivariada** — raramente verdade em dados clínicos (`hypothyroid`, `colic`) e manufatura (`cylinderbands`).
- **Hipersensível a N grande:** com 3.772 linhas (`hypothyroid_t4u`, `sick_*`), rejeita H0 para diferenças triviais.
- O fallback proxy combina t-tests univariados via Fisher's method — perde a estrutura conjunta multivariada.

**Evidência:** hypothyroid_t4u tem Little p=0.000 com N=3.772, mas o PKLM (que não assume normalidade) daria um veredicto diferente. O teste rejeita por sensibilidade ao tamanho da amostra, não por ausência real de MCAR.

---

## Problema 2 — Correlação ponto-biserial só captura linearidade

**Função:** `test_mar_correlation` → `stats.pointbiserialr(mask, X_i)` para cada i.

**Limitações:**
- Não detecta dependência **não-linear** (ex.: missing concentrado nos quartis extremos de X1).
- Não detecta **interações** (ex.: missing depende de X1 × X2, mas nenhuma marginal sozinha é significativa).
- Threshold `|corr| > 0.1` é **arbitrário e não calibrado** — com N=200 e 10% missing, |corr|=0.1 pode ser ruído; com N=3.000 é sinal forte.
- Não corrige para **múltiplas comparações** (4 Xi testados, p-min sem Bonferroni).

---

## Problema 3 — KS test "obs vs imputados-com-mediana" é conceitualmente mal-formulado

**Função:** `test_mnar_ks` → `stats.ks_2samp(X0_obs, X0_fillna_median)`.

**Este é o problema mais grave.** O teste compara a distribuição de X0 observado contra a mesma distribuição com um spike artificial na mediana:

```
X0_obs:     [valores contínuos sem o que falta]
X0_imputed: [mesmos valores + pico na mediana]
```

O KS **sempre detecta diferença** quando a imputação introduz um artefato na mediana. Isso não é um teste de MNAR — é um teste de "imputar com mediana distorce a distribuição", que é tautologicamente verdadeiro.

Mesmo se reformulado corretamente, o KS tem **poder de ~5,8% para MNAR** com missing rate ≤10% (Sportisse 2024, PKLM paper).

---

## Problema 4 — Lógica de diagnóstico é uma negação fraca

**Função:** `diagnose(label, little_p, max_corr, mar_sig, ks_p)`.

A lógica de decisão para MNAR é:

```
MNAR ← rejeita MCAR + KS significativo + sem correlação mask-Xi
```

Isso é uma **regra por exclusão tripla**: "não é MCAR, não parece MAR, e o KS (mal-formulado) diz algo". Não usa **nenhuma evidência positiva** de auto-dependência de X0. Resultado: 6/7 MNAR reais caem em "MCAR confirmado".

---

## Problema 5 — Recursos existentes não aproveitados

O codebase já tinha implementados, mas não integrados ao validar_rotulos.py:

| Recurso | Arquivo | O que faz | Aproveitado? |
|---------|---------|-----------|:---:|
| PKLM (Spohn 2024) | `baselines/pklm.py` | Teste MCAR não-paramétrico via classificação | ❌ |
| CAAFE-MNAR features | `features/caafe_mnar.py` | `tail_asymmetry`, `kurtosis_excess`, `cond_entropy`, `quantile_ratio` | ❌ |
| AUC mask~Xobs | `classificar_mnar.py` | AUC de RF prevendo mask de observáveis | ❌ (só para Focused/Diffuse) |

Integrar esses recursos é a base do protocolo v2.

---

## Conclusão

Dos 57% de inconsistência reportados pelo v1:
- ~30% são provavelmente **fragilidade dos testes** (Little hipersensível, KS tautológico, correlação linear)
- ~25% são **limitação teórica real** (MNAR não-identificável com dados observados)

O protocolo v2 ataca o primeiro componente. O segundo é irredutível — o que se pode fazer é quantificar a incerteza (Camada D Bayesiana) em vez de forçar uma decisão binária.
