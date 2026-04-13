# STEP 07: Implementar PKLM

**Status: PENDENTE**
**Estimativa: 1-2 dias**
**Papel no paper: Baseline de comparação + feature adicional (não é o foco principal)**

---

## Motivação

PKLM (Spohn et al., 2024) é um teste não-paramétrico para MCAR baseado em Random Forest + KL divergence. Diferente do teste de Little (que assume normalidade), PKLM funciona com qualquer distribuição.

Para o benchmark, PKLM serve como:
1. **Baseline de comparação** para detecção de MCAR
2. **Feature adicional** (`pklm_score`) no nosso pipeline
3. **Primeiro estágio** do classificador hierárquico (alternativa ao Little's proxy)

**Referência:** `Artigos_Relevantes/08_Classificacao_Mecanismos_Missing_Data/PKLM_Flexible_MCAR_Test_Using_Classification_2024.pdf`

---

## Como PKLM Funciona

1. Dividir os dados observados em K subsets baseados no padrão de missing
2. Treinar Random Forest para distinguir os subsets
3. Calcular KL divergence entre as distribuições de probabilidade preditas
4. Se KL > threshold → rejeitar MCAR (missing depende de algo)

**Nota:** PKLM é um **teste** (MCAR vs não-MCAR), não um classificador 3-way. Não distingue MAR de MNAR.

---

## Implementação

**Arquivo a criar:** `v2_improved/baselines/pklm.py`

### Passo 1: Implementar o teste PKLM

```python
def pklm_test(df, missing_col='X0', n_permutations=100):
    """
    Retorna:
    - pklm_statistic: KL divergence observada
    - pklm_pvalue: p-valor via permutação
    - rejects_mcar: bool (p < 0.05)
    """
    # 1. Criar mask de missing
    mask = df[missing_col].isna().astype(int)
    
    # 2. Usar variáveis observadas como features
    X = df.drop(columns=[missing_col]).values
    
    # 3. Treinar RF para prever mask a partir de X
    # 4. Calcular KL divergence das probabilidades
    # 5. Permutation test: repetir com mask shuffled
    # 6. p-valor = proporção de KL_permuted >= KL_observed
```

### Passo 2: Usar como feature

Extrair `pklm_score` (a estatística KL) e `pklm_pvalue` como features adicionais:
- Se pklm_pvalue alto → MCAR (não rejeita)
- Se pklm_pvalue baixo → não-MCAR (MAR ou MNAR)

### Passo 3: Comparar com Little's proxy

| Teste | Tipo | Assume Normalidade | Poder em N pequeno |
|-------|------|:------------------:|:------------------:|
| Little's test | Paramétrico | Sim | Baixo |
| Little's proxy (KS) | Não-paramétrico | Não | Médio |
| PKLM | Não-paramétrico (RF) | Não | Alto (esperado) |

### Passo 4: Integrar no pipeline

Adicionar `pklm_score` e `pklm_pvalue` como features opcionais:
```bash
python extract_features.py --model none --extra-features pklm
```

---

## Testes de Validação

### Teste 1: PKLM detecta não-MCAR
Em datasets MAR e MNAR sintéticos, PKLM deve rejeitar MCAR em > 80% dos casos.

### Teste 2: PKLM não rejeita MCAR verdadeiro
Em datasets MCAR sintéticos, PKLM deve NÃO rejeitar em > 90% dos casos (taxa de falso positivo < 10%).

### Teste 3: PKLM vs Little's proxy
PKLM deve ter **poder superior** ao Little's proxy, especialmente em:
- Amostras pequenas (< 100 observações)
- Distribuições não-normais (exponencial, beta)

### Teste 4: PKLM como feature melhora pipeline
Adicionar pklm_score/pvalue ao pipeline (23 features) deve manter ou melhorar accuracy.

---

## Critério de Conclusão

- [ ] PKLM implementado e testado em dados sintéticos
- [ ] Comparação com Little's proxy (poder, taxa de falso positivo)
- [ ] pklm_score integrado como feature no pipeline
- [ ] Testado como primeiro estágio do hierárquico (alternativa ao Little's proxy)
- [ ] Resultados documentados
