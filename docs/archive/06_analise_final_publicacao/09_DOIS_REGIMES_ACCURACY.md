# Framework Conceitual: Dois Regimes de Accuracy

**Data:** 2026-04-19

Este é um dos achados conceituais mais importantes da pesquisa. A decomposição em dois regimes oferece um framework para entender os limites da classificação de mecanismos de missing data e guiar pesquisa futura.

---

## 1. Visão Geral

```
Accuracy
  63% ─── ┌─────────────────────────────────┐ ← domain_prior sozinho
          │    REGIME DE DOMAIN REASONING    │
  56% ─── │    (LLM usa domínio + variável)  │ ← pipeline completo (D)
          └─────────────────────────────────┘
  51% ─── ┌─────────────────────────────────┐ ← + LLM data-driven
          │                                 │
  48% ─── │     REGIME ESTATÍSTICO          │ ← + CAAFE
          │     (features dos dados)        │
  41% ─── └─────────────────────────────────┘ ← baseline (21 features)
          │                                 │
  33% ─── ├─────────────────────────────────┤ ← chance level (3 classes)
```

---

## 2. Regime Estatístico (40-51%)

### Definição
Accuracy alcançável usando **apenas padrões nos dados**, sem informação de domínio ou contexto.

### Componentes

| Camada | Features | Accuracy | Δ | Fonte |
|--------|:--------:|:--------:|:-:|-------|
| Baseline | 21 estatísticas + discriminativas + MechDetect | 40.5% | +7.2pp vs chance | Padrões estatísticos nos dados |
| + CAAFE | 4 features distribucionais | 47.6% | +7.1pp | Propriedades de X0 (tail, kurtosis, entropy) |
| + LLM data-driven | 5 features de raciocínio sobre estatísticas | 50.5% | +2.9pp | LLM analisa estatísticas resumo |

### Teto: ~51%
Após 30 features estatísticas (21 baseline + 4 CAAFE + 5 LLM data-driven), a accuracy satura em ~50.5%. Adicionar mais features estatísticas não ajuda.

### Por que o teto existe
1. **MNAR é teoricamente indetectável** por variáveis observadas (X0 está faltante)
2. **MCAR vs MAR** é difícil sem contexto — padrões estatísticos são similares
3. **Label noise** (57%) limita o que qualquer modelo pode aprender

### Contribuição principal neste regime: CAAFE (+7.1pp)
As 4 features CAAFE capturam propriedades de X0 **apesar** de estar faltante:
- `tail_asymmetry`: MNAR afeta caudas assimetricamente
- `kurtosis_excess`: MNAR distorce a curtose
- `cond_entropy_X0_mask`: entropia da mask dado X0 revela dependência
- `missing_rate_by_quantile`: MNAR é não-uniforme por quantil

---

## 3. Regime de Domain Reasoning (56-63%)

### Definição
Accuracy alcançável quando o LLM adiciona **raciocínio sobre o domínio e contexto** dos dados, além das features estatísticas.

### Componentes

| Camada | Features | Accuracy | Δ | Fonte |
|--------|:--------:|:--------:|:-:|-------|
| Regime estatístico completo | 30 | 50.5% | — | Dados |
| + domain_prior | 31 (30 + 1) | 56.2% | +5.7pp | LLM infere mecanismo dado domínio + variável |
| domain_prior sozinho | 1 | 63.1% | — | Puro domain reasoning |

### O que o LLM faz
Dado apenas:
- `domain`: ex. "endocrinology", "oceanography", "census"
- `x0_variable`: ex. "Insulin", "air_temperature", "wages"

O LLM infere a **probabilidade** de cada mecanismo:
- dp=0.0 → provavelmente MCAR
- dp=0.5 → provavelmente MAR
- dp=1.0 → provavelmente MNAR

### Exemplos de raciocínio

| Domínio | Variável | Inferência LLM | Mecanismo Real | Correto? |
|---------|----------|:--------------:|:--------------:|:--------:|
| Endocrinology | Insulin | MNAR (dp=1.0) | MNAR | ✅ |
| Maritime | air_temperature | MAR (dp=0.5) | MAR | ✅ |
| Transportation | age | MAR (dp=0.5) | MAR | ✅ |
| Economics | wages | MNAR (dp=1.0) | MNAR | ✅ |
| Medical | bare_nuclei | MAR (dp=0.5) | MCAR | ❌ |
| Industrial | blade_pressure | MAR (dp=0.5) | MCAR | ❌ |

### Desempenho por classe

| Classe | LLM Correto | Padrão de erro |
|--------|:-----------:|----------------|
| MAR | **96.5%** | Raramente erra — MAR tem contexto forte |
| MNAR | 34.0% | 61% confundido com MAR — sem evidência forte de dependência em X0 |
| MCAR | 27.6% | 67% classificado como MAR — LLM assume posição moderada na incerteza |

### Por que domain_prior sozinho (63%) > pipeline completo (56%)
O ML pipeline mistura domain_prior com 30 features estatísticas ruidosas. As features estatísticas **diluem** o sinal do domain_prior, especialmente para MCAR (onde as features estatísticas confundem com MAR).

Isso sugere que, para datasets com metadata disponível, o domain_prior deve ter **peso maior** no ensemble.

---

## 4. Implicações para Pesquisa Futura

### O que este framework diz

1. **Sem domain knowledge, o teto é ~51%.** Melhorar features estatísticas além deste ponto requer uma mudança de paradigma — ou dados muito mais limpos.

2. **Com domain knowledge, o teto sobe para ~63%.** Mas depende da disponibilidade de metadata (domínio, nomes de variáveis).

3. **O gap entre 56% e 63%** é causado pela diluição do domain_prior por features ruidosas. Melhorar a integração (ex: weighted ensemble, soft voting) pode fechar este gap.

4. **O teto absoluto é ~65%** dado 57% de label noise. Para ir além, é necessário limpar labels ou usar datasets com ground truth verificável.

### Caminhos para superar os tetos

| Teto | Caminho | Dificuldade |
|------|---------|:-----------:|
| 51% (estatístico) | Domain reasoning via LLM | ✅ Feito neste trabalho |
| 56% (pipeline atual) | Melhor integração domain_prior + estatísticas | Média |
| 63% (domain_prior) | Metadata mais rica (descrição detalhada de missing) | Média |
| 65% (label noise) | Limpar labels ou usar dados com ground truth | Alta |
| >65% | Novos dados, novos testes, nova teoria | Pesquisa aberta |

---

## 5. Analogia para a Dissertação

> **Analogia médica:** Classificar mecanismos de missing data é como diagnosticar uma doença. 
> - O **regime estatístico** é como fazer exames de laboratório: útil, mas com limite de accuracy (exames normais não descartam tudo).
> - O **regime de domain reasoning** é como a anamnese do médico: "dado que o paciente é diabético e o exame é insulina, a ausência provavelmente é informativa (MNAR)."
> - Combinar ambos (exames + anamnese) dá o melhor resultado, mas a anamnese sozinha frequentemente é mais informativa que os exames sozinhos.

---

## 6. Resumo Visual

```
┌───────────────────────────────────────────────────┐
│                                                   │
│   FONTE DE INFORMAÇÃO        ACCURACY RANGE       │
│   ─────────────────          ──────────────       │
│                                                   │
│   Chance (nenhuma)           33.3%                │
│        ↓ +7.2pp                                   │
│   Features estatísticas      40.5%                │
│        ↓ +7.1pp                                   │
│   + CAAFE                    47.6%   ┐            │
│        ↓ +2.9pp                      │ Regime     │
│   + LLM data-driven         50.5%   ┘ Estatístico│
│        ↓ +5.7pp                                   │
│   + LLM domain_prior        56.2%   ┐ Regime     │
│                                      │ Domain     │
│   domain_prior sozinho       63.1%   ┘ Reasoning  │
│                                                   │
│   Teto teórico (label noise) ~65%                 │
│                                                   │
└───────────────────────────────────────────────────┘
```
