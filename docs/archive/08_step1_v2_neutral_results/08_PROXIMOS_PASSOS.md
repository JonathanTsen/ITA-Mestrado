# Próximos Passos

**Data:** 2026-04-25
**Estado atual:** Step 1 V2 Neutral concluído (49.3% CV); abaixo do target de 60%+

---

## 1. Direções priorizadas

Ordenadas por **razão custo/benefício esperado**.

### 🟢 Prioridade 1 — Validação barata da regressão (custo: $0)

**Objetivo:** confirmar que a regressão vs `forensic_neutral_v2` é causada pela expansão do benchmark, não por bug de código.

**Ação:** subset analysis — calcular acurácia de `step1_v2_neutral` apenas sobre os 23 datasets de `forensic_neutral_v2` (sem re-extrair).

**Como executar:**
```bash
cd "IC - ITA 2/Scripts/v2_improved"
uv run python -c "
import pandas as pd
B = '../../Output/v2_improved/step1_v2_neutral/real/ml_com_llm/gemini-3-pro-preview'
X = pd.read_csv(f'{B}/X_features.csv')
y = pd.read_csv(f'{B}/y_labels.csv')
g = pd.read_csv(f'{B}/groups.csv')

# Lista de datasets antigos (23, do forensic_neutral_v2)
OLD_23 = {
    'MAR_airquality_ozone', 'MAR_colic_resprate', 'MAR_hearth_chol',
    'MAR_kidney_hemo', 'MAR_mammographic_density', 'MAR_oceanbuoys_airtemp',
    'MAR_oceanbuoys_humidity', 'MAR_sick_t3', 'MAR_titanic_age', 'MAR_titanic_age_v2',
    'MCAR_autompg_horsepower', 'MCAR_breastcancer_barenuclei', 'MCAR_cylinderbands_bladepressure',
    'MCAR_cylinderbands_esavoltage', 'MCAR_hypothyroid_t4u',
    'MNAR_adult_capitalgain', 'MNAR_colic_refluxph', 'MNAR_cylinderbands_varnishpct',
    'MNAR_hepatitis_protime', 'MNAR_kidney_pot', 'MNAR_mroz_wages',
    'MNAR_pima_insulin', 'MNAR_pima_skinthickness'
}

mask = g['group'].isin(OLD_23)
X_sub, y_sub, g_sub = X[mask].reset_index(drop=True), y[mask].reset_index(drop=True), g[mask].reset_index(drop=True)
print(f'Subset: {len(X_sub)} samples, {g_sub[\"group\"].nunique()} groups')

# domain_prior solo
df = pd.concat([y_sub, X_sub[['llm_ctx_domain_prior']]], axis=1)
df['true'] = df['label'].map({0:'MCAR', 1:'MAR', 2:'MNAR'})
df['pred'] = pd.cut(df['llm_ctx_domain_prior'], bins=[-0.01, 0.25, 0.75, 1.01], labels=['MCAR','MAR','MNAR'])
acc = (df['pred'].astype(str) == df['true']).mean()
print(f'domain_prior alone (23 antigos): {acc*100:.1f}%')
print(f'Comparar com forensic_neutral_v2: 63.1%')
"
```

**Decisão:** se acc ≈ 55-65% nos 23 antigos, confirma H1 e descarta H4 (regressão de código). Se acc ainda for ~45%, há regressão silenciosa que precisa investigação.

**Tempo:** < 5 min. **Custo:** $0.

---

### 🟢 Prioridade 2 — Step 2 (Causal Reasoning DAG) (custo: ~$30-36)

**Objetivo:** atacar diretamente o MAR-bias residual nos 6 datasets críticos.

**Hipótese:** o LLM "encontra causas MAR" sem nomear variável específica. Forçar decomposição em 2 etapas (DAG → classificação) eleva recall MNAR e MCAR onde o Step 1 falha.

**Plano técnico (já documentado em `docs/07_next_steps_domain_reasoning/02_step2_causal_reasoning.md`):**

1. Modificar `llm/context_aware.py` para implementar `_extract_causal_dag()` (Etapa 1) e `_classify_with_dag()` (Etapa 2)
2. Adicionar 3 novas features: `llm_ctx_n_type_A`, `llm_ctx_n_type_B`, `llm_ctx_n_type_C`
3. Validar no conjunto-alvo de 6 datasets críticos antes de rodar full benchmark

**Adaptações específicas baseadas em `05_DATASETS_PROBLEMATICOS.md`:**

```
TIPO C2 (NOVO): MNAR por SELEÇÃO MÉDICA
Quando exame laboratorial só é pedido após suspeita de anormalidade,
o valor missing depende implicitamente do valor latente esperado.
Exemplos: insulina, eletrólitos, T4U, T3.

INSTRUÇÃO ESPECIAL: Em domínios clínicos, NÃO presuma MAR baseado
em correlação estatística. Pergunte: "A decisão de MEDIR depende
do valor esperado da própria variável?" Se sim → MNAR. Se a decisão
é independente → MAR.

HIPÓTESE NULA TÉCNICA: Em domínios industriais (manufatura, sensores),
considere MCAR como hipótese nula a ser refutada.
```

**Como executar:**
```bash
# 1. Implementar mudanças em llm/context_aware.py (estimativa: 1-2h código)
# 2. Validar em subset (1 dataset MCAR, 1 MAR, 1 MNAR difícil) — ~$2 USD
# 3. Rodar full benchmark dividido em 2 metades:
uv run python extract_features.py \
    --model gemini-3-pro-preview --data real --llm-approach context \
    --metadata-variant neutral --datasets-include data/datasets_part1.txt \
    --experiment step2_dag_part1 --workers 10
# 4. Idem part2; merge; train
```

**Target:** elevar CV avg de 49.3% → ≥55%, com foco em recall dos 6 datasets críticos (4-12% atual → ≥30% target).

**Tempo:** ~2h código + 1h33min × 2 + 10min train ≈ **5h totais**. **Custo:** ~$30-36.

---

### 🟡 Prioridade 3 — Re-execução `forensic_neutral_v2` controlada (custo: ~$25-30)

**Objetivo:** confirmar que a regressão NÃO é por mudança de código.

**Ação:** rodar `forensic_neutral_v2` (23 datasets) com o código atual e comparar com 56.2% reportado.

**Como executar:**
```bash
# Criar lista forensic_v2_datasets.txt com os 23 antigos
# Rodar:
uv run python extract_features.py \
    --model gemini-3-pro-preview --data real --llm-approach context \
    --metadata-variant neutral --datasets-include data/forensic_v2_datasets.txt \
    --experiment forensic_neutral_v2_rerun --workers 10
uv run python train_model.py --model gemini-3-pro-preview --data real \
    --experiment forensic_neutral_v2_rerun
```

**Decisão:**
- Se acc ≈ 55-58% → confirma H4 descartada; regressão é metodológica (bom)
- Se acc < 50% → há regressão silenciosa; investigar diff de código entre 2026-04-19 e hoje

**Tempo:** ~1h30min. **Custo:** ~$25-30.

**Observação:** se Prioridade 1 (subset analysis) já produziu evidência convincente, esta validação pode ser pulada.

---

### 🟡 Prioridade 4 — Stacking Step 1 + Cleanlab pesos (custo: $0)

**Objetivo:** aplicar a técnica que deu +2.7pp em V3+ original sobre o experimento atual.

**Ação:** rodar Cleanlab para identificar rótulos suspeitos no benchmark de 29 datasets, treinar com pesos ajustados.

**Como executar:**
```bash
# Já existe Scripts/v2_improved/clean_labels.py
uv run python clean_labels.py --data real --experiment step1_v2_neutral
# Re-treinar com pesos
uv run python train_hierarchical_v3plus.py --experiment step1_v2_neutral
```

**Target:** elevar CV avg de 49.3% → ~52%.

**Tempo:** ~10 min. **Custo:** $0 (sem chamadas LLM novas).

---

### 🔴 Prioridade 5 — Step 3 (Self-Consistency com Pro) (custo: ~$80-100)

**Objetivo:** votação ponderada de 5 perspectivas para reduzir variância.

**Risco:** Self-Consistency com Flash falhou (38.4% — degradou para baseline) porque perspectivas dominais ativaram o mesmo prior. Self-Consistency com Pro pode ter o mesmo problema, mas Pro tem capacidade superior de raciocinar contra-prior.

**Custo alto:** 5 chamadas/bootstrap × 1.421 bootstraps × 2 metades = **14.210 chamadas Pro** ≈ $80-100.

**Recomendação:** **adiar** até depois de Step 2. Se Step 2 atingir target, SC pode não ser necessário. Se Step 2 falhar, SC é a próxima tentativa.

---

### 🔴 Prioridade 6 — Roteamento adaptativo agente↔stat (custo: variável)

**Objetivo:** chamar LLM apenas em datasets onde o classificador estatístico tem alta entropia (incerteza).

**Hipótese:** ML+CAAFE puro (sem LLM) já atinge ~47% em datasets fáceis. LLM agrega valor apenas em casos ambíguos. Filtrar permite **reduzir custo em ~60%** mantendo accuracy.

**Implementação:**
1. Treinar classificador estatístico puro
2. Identificar amostras com p_max < 0.6 (entropia alta)
3. Chamar LLM apenas para essas
4. Combinar predições

**Tempo:** 1 dia código + reexecução parcial. **Custo:** ~$15 (60% redução).

---

## 2. Sequência recomendada (3 sprints)

### Sprint 1 (1-2 dias, ~$0): validações baratas

1. **Prioridade 1** — subset analysis dos 23 antigos (validar H1)
2. **Prioridade 4** — Cleanlab + stacking sobre `step1_v2_neutral`

**Decisão pós-Sprint 1:** se subset analysis confirma H1 e Cleanlab atinge ~52% CV, podemos defender narrativa "Step 1 + Cleanlab = solução pragmática" na tese.

### Sprint 2 (3-5 dias, ~$30-36): Step 2 Causal DAG

3. **Prioridade 2** — implementar Step 2, validar em 6 datasets críticos, rodar full benchmark

**Decisão pós-Sprint 2:**
- Se Step 2 ≥ 55% CV → solução completa, escrever paper. Skip Step 3.
- Se Step 2 ≈ 50-54% CV → ganho marginal; considerar Step 3 ou roteamento adaptativo
- Se Step 2 < 50% CV → revisitar abordagem (pode haver problema fundamental no benchmark)

### Sprint 3 (opcional, $0-100): refinamentos

4. **Prioridade 3** — re-rodar `forensic_neutral_v2` com 23 datasets (se Prioridade 1 inconclusiva)
5. **Prioridade 6** — roteamento adaptativo (otimização de custo)
6. **Prioridade 5** — Step 3 self-consistency com Pro (se Step 2 não atingiu target)

---

## 3. Critérios de sucesso

Para considerar a linha de pesquisa Step 1 → Step 2 **bem-sucedida**:

| Métrica | Atual (Step 1) | Mínimo Step 2 | Ideal Step 2 |
|---------|:--------------:|:-------------:|:------------:|
| CV avg (NB) | 49.3% | **≥ 55%** | ≥ 60% |
| F1 macro | 0.55 | ≥ 0.58 | ≥ 0.62 |
| Recall 6 datasets críticos (média) | ~5% | ≥ 30% | ≥ 50% |
| `MNAR_pima_insulin` recall | 4% | ≥ 40% | ≥ 60% |
| Variância CV (std/avg) | 0.29 | ≤ 0.25 | ≤ 0.20 |

Se Step 2 atinge ≥ 55% CV com ≥ 30% recall nos 6 críticos, **escrever paper** com:
- Step 1 V2 Neutral como ablação ("prompt engineering simples atinge 49% CV")
- Step 2 Causal DAG como contribuição principal ("decomposição causal explícita atinge 55%+ CV")
- Análise dos 9 datasets críticos como caso de estudo
- Auditoria formal de leakage (Canais A-F) como contribuição metodológica

---

## 4. Decisões pendentes para o usuário

1. **Aprovar Sprint 1?** — validações baratas custam ~$0 e dão sinal forte. Recomendo prosseguir.

2. **Investir em Sprint 2 (Step 2)?** — ~$30-36 e 5h de execução. Tem alta probabilidade de mover a aguilha; é a recomendação principal.

3. **Como reportar Step 1 V2 na tese?** — três opções:
   - **(a)** Como progresso da pesquisa, comparando com `forensic_neutral_v2` (transparente sobre regressão)
   - **(b)** Como avaliação sobre benchmark expandido, sem comparação direta (foca no `step10_flash` baseline)
   - **(c)** Aguardar Step 2 e reportar ambos juntos (Step 1 como ablação)

   **Recomendação:** (c) se Sprint 2 for aprovado; (a) caso contrário (transparência > ocultação).
