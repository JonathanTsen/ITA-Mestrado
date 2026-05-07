# CAAFE-MNAR neste repositorio

Este arquivo corrige a terminologia sobre CAAFE no projeto.

## Resumo simples

O CAAFE original **usa LLM**, mas **nao e uma LLM**.

Neste repositorio, as features `caafe_*` **nao sao geradas por LLM**. Elas sao
features deterministicas, escritas manualmente em Python, inspiradas na ideia
do CAAFE e adaptadas para detectar MNAR.

O nome correto para usar na tese e:

> features CAAFE-inspired deterministicas para deteccao de MNAR

ou, de forma curta:

> CAAFE-MNAR

## O que e o CAAFE original

CAAFE significa **Context-Aware Automated Feature Engineering**. O metodo foi
proposto por Hollmann, Muller e Hutter no artigo:

> *Large Language Models for Automated Data Science: Introducing CAAFE for
> Context-Aware Automated Feature Engineering*, NeurIPS 2023.

A ideia original e um ciclo de engenharia automatica de features:

1. O usuario fornece uma tabela e uma descricao do dataset.
2. Uma LLM, por exemplo GPT-4, sugere codigo Python para criar novas colunas.
3. Esse codigo roda na tabela.
4. Um modelo de Machine Learning testa se as novas colunas melhoram o
   resultado.
5. As features boas ficam; as ruins sao descartadas.

Entao, de forma bem simples:

- **LLM**: sugere codigo.
- **ML**: testa se a feature ajudou.
- **CAAFE**: e o metodo que combina essas duas partes.

## O que este projeto implementa

Este projeto **nao reimplementa o loop CAAFE original**.

Aqui, o arquivo
[`src/missdetect/features/caafe_mnar.py`](../src/missdetect/features/caafe_mnar.py)
calcula quatro features fixas:

- `caafe_auc_self_delta`
- `caafe_kl_density`
- `caafe_kurtosis_excess`
- `caafe_cond_entropy_X0_mask`

Elas sao calculadas por Python puro, sem chamada para API de LLM durante a
extracao.

## Logica das features

As features CAAFE-MNAR tentam responder uma pergunta:

> Parece que o valor de `X0` influencia o fato de `X0` estar faltando?

Se a resposta parece "sim", isso e sinal de **MNAR**, porque em MNAR o valor
ausente depende dele mesmo.

De forma simples:

- `caafe_auc_self_delta`: mede se adicionar `X0` imputado ajuda a prever a
  mascara de missing. Se ajuda, pode haver sinal de MNAR.
- `caafe_kl_density`: compara a distribuicao de `X0` entre linhas observadas e
  linhas missing. Se as distribuicoes sao muito diferentes, pode haver missing
  seletivo.
- `caafe_kurtosis_excess`: olha se a distribuicao observada de `X0` ficou com
  caudas ou formato estranho, o que pode acontecer quando valores extremos
  somem.
- `caafe_cond_entropy_X0_mask`: mede quanto saber `X0` reduz a incerteza sobre
  a mascara de missing. Se reduz muito, `X0` esta relacionado ao missing.

## Frase correta para a tese

Use:

> Implementamos features deterministicas inspiradas no CAAFE para deteccao de
> MNAR.

Evite:

> Implementamos o CAAFE.

Evite tambem:

> As features CAAFE foram geradas por uma LLM.

A formulacao mais honesta e precisa e: **CAAFE original usa LLM para gerar
features; este repositorio usa features fixas em Python inspiradas nessa
filosofia, focadas em MNAR**.

## Observacao sobre arquivos antigos

A pasta [`archive/`](archive/) guarda documentos historicos. Alguns textos
antigos usam nomes como `caafe_tail_asymmetry` e
`caafe_missing_rate_by_quantile`. Esses nomes foram usados em fases anteriores.
A versao atual v2b usa as quatro features listadas acima.
