# STEP 07: SMOTE Avançado — SMOTE-ENN / Borderline-SMOTE

**Status:** ✅ CÓDIGO PRONTO (flag `--balancing` implementada, não testada ainda)
**Esforço real:** Implementado por edição manual no `train_hierarchical_v3plus.py`
**Pacotes:** `imblearn` (já instalado)

---

## Motivação

O pipeline original usa SMOTE básico com k=3 vizinhos. Com 59.4% de labels ruidosos:

1. **SMOTE gera amostras sintéticas na região de overlap** MAR/MNAR → amplifica ruído
2. **Sem SMOTE (com pesos):** soft3zone/fullprob piora muito (45.8% vs 50.5%) porque L2 fica desbalanceado (70% MAR / 30% MNAR)
3. **SMOTE-ENN** resolve: gera amostras + limpa fronteira ruidosa

---

## Implementação (já feita)

### Flag `--balancing` em `train_hierarchical_v3plus.py`:

```bash
--balancing {smote, smote_enn, borderline, smote_tomek, none}
```

### Função `apply_balancing()` (substituiu `apply_smote()`):

```python
def apply_balancing(X_in, y_in, method="smote"):
    if method == "smote":
        sampler = SMOTE(random_state=42, k_neighbors=k)
    elif method == "smote_enn":
        sampler = SMOTEENN(smote=SMOTE(...), random_state=42)
    elif method == "borderline":
        sampler = BorderlineSMOTE(random_state=42, k_neighbors=k)
    elif method == "smote_tomek":
        sampler = SMOTETomek(smote=SMOTE(...), random_state=42)
    return sampler.fit_resample(X_in, y_in)
```

---

## Testes a Rodar

### Teste 1: SMOTE-ENN sem pesos
```bash
python train_hierarchical_v3plus.py --data real --experiment step05_pro --routing all --balancing smote_enn
```
**Esperado:** soft3zone deve melhorar (fronteira MAR/MNAR mais limpa antes de calibrar)

### Teste 2: SMOTE-ENN com pesos Cleanlab
```bash
python train_hierarchical_v3plus.py --data real --experiment step05_pro --routing all --balancing smote_enn --clean-labels weight
```
**Nota:** Com pesos, SMOTE é desativado (pesos incompatíveis com SMOTE). Este teste verifica se a lógica funciona.

### Teste 3: Borderline-SMOTE
```bash
python train_hierarchical_v3plus.py --data real --experiment step05_pro --routing all --balancing borderline
```
**Esperado:** Melhoria marginal, Borderline foca na fronteira de decisão.

---

## Validação

1. Comparar accuracy e MNAR recall de cada método vs SMOTE padrão
2. Verificar tamanho do dataset após SMOTE-ENN (não deve perder >30%)
3. LOGO CV com cada método
4. Verificar se soft3zone+SMOTE-ENN > soft3zone+SMOTE no LOGO CV

---

# Anexo: Resultados do Experimento

> Originalmente publicado como `RESULTADOS_STEP07.md`. Consolidado aqui em 2026-04-19.


**Data:** 2026-04-19
**Experimento:** step05_pro (mesmos dados de STEP01+04)
**Status:** CONCLUÍDO — SMOTE regular permanece como melhor opção

---

## Comparação de Métodos de Balanceamento

### LOGO CV — NaiveBayes + soft3zone (melhor combinação conhecida)

| Método | LOGO CV (%) | Delta vs SMOTE |
|--------|:-----------:|:--------------:|
| **SMOTE (baseline)** | **56.0** | — |
| SMOTE-Tomek | 55.9 | -0.1pp |
| SMOTE-ENN | 55.0 | -1.0pp |
| None (sem balanceamento) | 49.7 | -6.3pp |
| Borderline-SMOTE | 47.6 | -8.4pp |

### Holdout — Melhor modelo por método

| Método | Acc Max | Modelo | MNAR Recall |
|--------|:-------:|--------|:-----------:|
| SMOTE | 50.5% | GradientBoosting | 40.0% |
| SMOTE-ENN | 47.5% | LogisticRegression | 24.0% |
| SMOTE-Tomek | 47.5% | GradientBoosting | 22.0% |
| None | 49.8% | GradientBoosting | 34.0% |
| Borderline | 47.1% | GradientBoosting | 26.0% |

---

## Análise

### Por que SMOTE regular vence?

1. **Labels ruidosos dominam o efeito.** Com 59.4% de labels inconsistentes, a "fronteira de decisão" que ENN tenta limpar não é confiável — ENN remove amostras que podem estar corretas mas cujos vizinhos têm labels errados.

2. **SMOTE-ENN remove amostras demais.** Em datasets pequenos (~300 amostras no L2), a limpeza agressiva do ENN reduz o tamanho efetivo do treino, prejudicando mais do que ajudando.

3. **Borderline-SMOTE foca na região errada.** A fronteira MAR/MNAR é intrinsecamente ruidosa (Rubin 1976 — MAR e MNAR não são distinguíveis por dados observados). Gerar amostras sintéticas nessa fronteira amplifica ruído.

4. **SMOTE-Tomek é o mais próximo.** Tomek links remove apenas pares de amostras de classes diferentes que são nearest neighbors mutuamente — menos agressivo que ENN. Por isso, resultado quase idêntico ao SMOTE regular (-0.1pp).

5. **Sem balanceamento é claramente pior (-6.3pp).** Confirma que o SMOTE é necessário — o desbalanceamento MAR:MNAR (~70:30) no L2 prejudica significativamente o treinamento.

### Conclusão

**SMOTE regular (k=3) permanece como a melhor estratégia de balanceamento.** Nenhuma variante testada supera o baseline. A hipótese inicial — que SMOTE-ENN limparia a fronteira ruidosa — não se confirmou porque o ruído vem dos labels, não do overlap natural entre classes.

---

## Implicações para próximos steps

- **SMOTE-ENN não é recomendado como complemento ao Cleanlab.** Embora a motivação fosse diferente (Cleanlab corrige labels, ENN corrige fronteira), na prática o ENN remove informação útil em datasets pequenos.
- **Manter SMOTE regular como default.** Não precisa trocar.
- **Foco nos steps de maior impacto:** Step 02 (CatBoost+Optuna) e Step 03 (novas features L2).

---

## Arquivos

```
Output/v2_improved/step05_pro/real/hierarquico_v3plus/
├── cv_logo_smote_regular.csv     # LOGO CV com SMOTE (baseline)
├── cv_logo_smote_enn.csv         # LOGO CV com SMOTE-ENN
├── cv_logo_smote_tomek.csv       # LOGO CV com SMOTE-Tomek
├── todas_variantes_smote_enn.csv # Holdout detalhado SMOTE-ENN
└── (outros CSVs/PNGs padrão)
```
