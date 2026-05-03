# Analise: gerador.py

**Arquivo:** `Scripts/gerador.py`
**Funcao:** Gera datasets sinteticos com mecanismos de dados faltantes (MCAR, MAR, MNAR)

---

## Bugs e Erros de Logica

### BUG-G1: Excecao silenciosa esconde se mdatagen funciona [ALTO]

**Linhas 131-133:** `except Exception` captura TODAS as excecoes do `mdatagen` e cai silenciosamente no fallback manual. Nao ha log, contagem, ou indicacao de quantas vezes o fallback e usado.

**Impacto:** E impossivel saber se o `mdatagen` esta realmente contribuindo para a geracao de dados ou se 100% dos datasets usam o fallback manual.

**Correcao sugerida:** Logar a excecao com `logging.warning()` e manter um contador de fallbacks.

### BUG-G2: Sem registro de qual metodo gerou cada dataset [ALTO]

Nao ha registro se um dataset foi gerado pelo `mdatagen` ou pelo fallback manual. Os dois metodos podem implementar os mecanismos de forma diferente, criando um dataset inconsistente onde o processo gerador varia entre amostras de forma descontrolada.

**Impacto na tese:** Problema significativo de reprodutibilidade e metodologia.

**Correcao sugerida:** Salvar metadados (JSON) por dataset indicando o metodo utilizado.

### BUG-G3: Sem guarda `if __name__ == "__main__"` [MEDIO]

**Linhas 80-150:** O codigo executa no nivel do modulo. `import gerador` de outro modulo dispara a geracao completa de 3000 datasets.

**Correcao sugerida:** Envolver o codigo de execucao em `if __name__ == "__main__":`.

### BUG-G4: `CLEAN_OUTPUT = True` hardcoded e destrutivo [MEDIO]

**Linhas 90-94:** `shutil.rmtree` deleta todos os datasets existentes incondicionalmente a cada execucao. Sem confirmacao, sem backup, sem forma de desabilitar sem editar o codigo.

**Risco:** Se uma extracao longa de features LLM foi concluida e alguem executa `gerador.py` acidentalmente, todos os datasets sao regenerados e as features extraidas ficam desalinhadas.

---

## Problemas Estatisticos / Matematicos

### STAT-G1: MAR e MNAR tem estrutura de probabilidade identica [ALTO - impacto na tese]

**Linhas 44-54 (MAR) e 59-69 (MNAR):** Usam a mesma formula sigmoid. A unica diferenca e:
- MAR: `z = (X1 - mean(X1)) / std(X1)`
- MNAR: `z = (X0 - mean(X0)) / std(X0)`

Como X0 e X1 sao ambos `Uniform(0,1)` independentes, as distribuicoes sao estatisticamente identicas. A assinatura estatistica de MAR e MNAR e muito similar, limitando fundamentalmente o desempenho do classificador (~70% de teto).

### STAT-G2: Colunas independentes dificultam distincao MAR vs MCAR [MEDIO]

**Linha 113:** Colunas X0-X4 sao geradas independentemente. Em dados reais, colunas sao correlacionadas, e essa correlacao afeta como MAR se manifesta. A falta de correlacao torna MAR e MCAR quase indistinguiveis estatisticamente.

### STAT-G3: Rescaling de probabilidade pode criar padroes degenerados [MEDIO]

**Linha 48:** `prob = prob * (p / (prob.mean() + 1e-12))` para taxas altas de missing, o fator de rescaling pode ser muito grande, e `np.clip` capa tudo em 1.0, criando um padrao quase deterministico ao inves de probabilistico.

### STAT-G4: Dados Uniform(0,1) sao irrealistas [BAIXO]

Dados reais tipicamente seguem distribuicoes normal, assimetrica ou multi-modal. O classificador treinado nesses dados sinteticos pode nao generalizar para padroes reais de missingness.

### STAT-G5: Mesma seed para os tres mecanismos [BAIXO]

**Linha 109:** `seed = 10_000 + k` e usado para MCAR, MAR e MNAR. O dataset `k` tem a mesma matriz base X para todos os mecanismos. Pode ser intencional para comparacao controlada, mas deve ser documentado.

---

## Edge Cases Nao Tratados

### EDGE-G1: Garantia minima de 1 valor faltante e fraca [BAIXO]

**Linhas 38-39, 53-54, 67-68:** Se nenhum valor cai abaixo do threshold de probabilidade, exatamente 1 valor e forcado como faltante, criando um artefato distribucional.

### EDGE-G2: Sem validacao do output do mdatagen [BAIXO]

**Linha 129:** `_get_generated_dataset(gen)` nao valida se o retorno tem as colunas e formato esperados.

---

## Problemas de Qualidade de Codigo

### QUAL-G1: Sem flag `--test` [BAIXO]

Outros scripts aceitam `--test`, mas `gerador.py` nao. Para gerar um set pequeno de teste, e preciso editar o codigo fonte.

### QUAL-G2: Sem barra de progresso [BAIXO]

Gerar 3000 datasets leva tempo significativo sem feedback visual.

---

## Potencial Data Leakage

### LEAK-G1: Label no nome do arquivo e diretorio [MEDIO]

**Linha 147:** `fname = f"{mech}_seed{seed}_mr{missing_rate}.txt"`. O mecanismo esta no nome do arquivo E o arquivo esta em diretorio nomeado pelo mecanismo. Qualquer codigo de extracao que acidentalmente use o caminho do arquivo tem acesso direto ao label.
