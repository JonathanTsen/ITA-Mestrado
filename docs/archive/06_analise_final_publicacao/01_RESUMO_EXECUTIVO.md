# Resumo Executivo da Pesquisa

**Data:** 2026-04-19

---

## Tema

Classificação automática de mecanismos de dados faltantes (MCAR, MAR, MNAR) usando Machine Learning com features estatísticas e features derivadas de Large Language Models (LLMs).

## Pergunta Central

> Features derivadas de LLMs melhoram a classificação de mecanismos de missing data em relação a features puramente estatísticas?

## Resposta Encontrada

**Sim, mas de forma inesperada.** O valor principal do LLM vem do **raciocínio sobre domínio** (domain reasoning), não da análise estatística de segunda ordem.

- Uma única feature LLM (`domain_prior`) alcança **63.1%** usando apenas nome de domínio + nome da variável
- O pipeline completo de features estatísticas alcança apenas **40.5%**
- Features LLM baseadas em análise estatística de segunda ordem são **ruído** — pioram o desempenho

## Números-Chave

| Métrica | Valor |
|---------|-------|
| Baseline (21 features estatísticas) | 40.5% |
| + CAAFE (4 features determinísticas) | 47.6% (+7.1pp) |
| + LLM data-driven (5 features) | 50.5% (+2.9pp) |
| + LLM domain_prior (1 feature) | **56.2%** (+5.7pp) |
| domain_prior sozinho | **63.1%** (+22.6pp vs baseline) |
| Chance level (3 classes) | 33.3% |

## Dados

- **Sintéticos:** 1200 amostras, 12 variantes (3 MCAR + 5 MAR + 4 MNAR), gerados via MissMecha
- **Reais:** 23 datasets (5 MCAR + 11 MAR + 7 MNAR), 1132 amostras bootstrapped
- **Validação:** GroupKFold-5 + LODO (Leave-One-Dataset-Out)

## Contribuições Originais

1. Framework de classificação 3-way para mecanismos de missing data
2. CAAFE features para detecção de MNAR (+7.1pp)
3. Evidência de LLM domain reasoning genuíno (+23pp sobre baseline)
4. Resultado negativo sobre LLM features estatísticas
5. Auditoria de labels em benchmarks reais (57% inconsistentes)
6. Benchmark de 23 datasets reais com validação estatística

## Veredicto

- **Para dissertação de mestrado:** Claramente suficiente
- **Para journal:** Publicável com a narrativa correta (domain reasoning, não accuracy bruta)
