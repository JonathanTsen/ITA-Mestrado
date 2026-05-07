# Documentacao do projeto

Comece por este arquivo. Ele aponta para a documentacao mais atual e evita
confundir resultados historicos do `archive/` com a versao atual do projeto.

## Leitura recomendada

1. [`methodology.md`](methodology.md) — metodologia atual: problema, features,
   modelos, validacao e resultados principais.
2. [`caafe_mnar.md`](caafe_mnar.md) — explicacao correta do CAAFE neste
   repositorio. Este e o arquivo principal para entender a diferenca entre o
   CAAFE original e as features `caafe_*` implementadas aqui.
3. [`HISTORICO.md`](HISTORICO.md) — linha do tempo dos experimentos e resultados
   das fases.
4. [`bibliography.md`](bibliography.md) — referencias usadas na tese.
5. [`reproducibility.md`](reproducibility.md) — como reproduzir os experimentos.

## Nota importante sobre CAAFE

Neste repositorio, `caafe_*` **nao significa que uma LLM esta rodando para
gerar features durante a extracao**.

O CAAFE original, de Hollmann, Muller e Hutter (NeurIPS 2023), e um metodo de
AutoML que usa uma LLM para sugerir codigo Python de novas features.

O que este projeto implementa atualmente e diferente:

> features CAAFE-inspired deterministicas para deteccao de MNAR.

Ou seja: sao features escritas manualmente em Python, inspiradas na ideia do
CAAFE, mas especializadas para detectar sinais de MNAR. O nome curto usado na
documentacao atual e **CAAFE-MNAR**.

## Onde evitar confusao

A pasta [`archive/`](archive/) guarda documentos historicos. Alguns arquivos
antigos usam "CAAFE" de forma mais solta porque refletem fases anteriores do
projeto. Para a tese e para a descricao final do metodo, use a terminologia de
[`caafe_mnar.md`](caafe_mnar.md) e [`methodology.md`](methodology.md).
