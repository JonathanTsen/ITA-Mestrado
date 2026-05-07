# Por que LLMs ajudam em MCAR mas não em MAR/MNAR?

## Resumo

O padrão observado nos três experimentos (ML-only, Flash, Pro) é consistente:

| Mecanismo | Sinal estatístico | LLM ajuda? | Resultado       |
|-----------|-------------------|------------|-----------------|
| MCAR      | Fraco             | **Sim**    | Flash +22pp     |
| MAR       | Forte             | **Não**    | ML-only +4-7pp  |
| MNAR      | Moderado          | Empate     | Diferença <4pp  |

## MCAR — LLM preenche a lacuna da estatística

MCAR (Missing Completely At Random) por definição **não possui padrão estatístico**
entre a missingness e as variáveis observadas. A ausência de dados é puramente
aleatória — não depende de nenhuma variável, observada ou não.

Consequência para as features ML:
- `auc_mask_from_Xobs` ≈ 0.5 (aleatório)
- `log_pval_X1_mask` → p-values altos (não-significativos)
- `X1_mean_diff` ≈ 0 (sem diferença entre grupos)
- `X0_censoring_score` baixo (sem truncamento)
- `caafe_kl_density` baixo (distribuições similares)

Quando **todas** as features estatísticas retornam valores ambíguos, o classificador
tem pouca informação para distinguir MCAR dos outros mecanismos. O resultado é um
F1 de 0.49 para ML-only.

As features LLM context-aware adicionam **informação de domínio**:
- `llm_ctx_domain_prior`: "para esta variável neste domínio, qual o mecanismo
  esperado a priori?" — o LLM pode raciocinar que "altura em um estudo pediátrico
  não tem razão para faltar sistematicamente"
- `llm_ctx_surprise`: "os dados estatísticos são surpreendentes dado o contexto?"
- `llm_ctx_confidence_delta`: diferença entre a confiança baseada em contexto e
  a baseada em estatística

Para MCAR, o LLM consegue identificar que **não há razão de domínio** para a
missingness ser sistemática. Esse sinal complementa a evidência estatística fraca,
elevando o F1 de MCAR para 0.60 (Flash) e 0.58 (Pro).

## MAR — ML já captura o sinal; LLM adiciona ruído

MAR (Missing At Random) significa que a missingness depende de **variáveis
observadas**. Exemplo: em um dataset hospitalar, o exame de potássio é mais
frequentemente solicitado para pacientes com creatinina elevada.

Essa dependência gera padrões estatísticos **fortes e diretamente mensuráveis**:
- `auc_mask_from_Xobs` alto (a máscara é predizível pelos preditores)
- `log_pval_X1_mask` → p-values muito baixos (relação significativa)
- `X1_mean_diff` ≠ 0 (médias diferem entre observados e missing)
- `X1_mannwhitney_pval` baixo (distribuições diferem)

As features ML foram **projetadas exatamente para capturar MAR** — correlação
entre missingness e variáveis observadas é o cerne do design.

Quando o sinal estatístico já é forte:
1. Adicionar 9 features LLM de natureza mais "subjetiva" **dilui o sinal** —
   o classificador tem 34 dimensões em vez de 25, com as 9 extras contribuindo
   pouco informacionalmente
2. O domain prior do LLM pode até **contradizer** a evidência estatística em
   alguns casos, introduzindo ruído
3. A feature selection (limitada a n/10 = 159 features) não é restritiva o
   suficiente para eliminar as features LLM pouco úteis

Resultado: ML-only lidera MAR com F1 = 0.424 vs 0.381 (Flash) e 0.357 (Pro).

A queda é **monotonicamente pior** com mais features LLM sofisticadas:
ML-only (0.424) > Flash (0.381) > Pro (0.357) — sugerindo que o modelo Pro
produz features LLM mais "confiantes" que confundem mais o classificador.

## MNAR — ambos capturam sinais complementares

MNAR (Missing Not At Random) significa que a missingness depende do **próprio
valor não-observado**. Exemplo: pessoas com renda muito alta tendem a não
reportar renda em surveys.

MNAR cria padrões **indiretos** e **sutis**:
- Distribuições truncadas ou com caudas cortadas
- `X0_censoring_score` elevado (padrão de censoring)
- `X0_tail_missing_ratio` alto (missing concentrado nas caudas)
- `caafe_kl_density` moderado (divergência parcial)

O ML captura esses padrões, mas com ruído — o sinal não é tão claro quanto MAR.

O LLM pode contribuir com raciocínio de domínio:
- "Renda é tipicamente sujeita a auto-censura"
- "Variáveis de saúde com valores extremos tendem a ser sub-reportadas"

Ambas as abordagens capturam **partes complementares** do sinal, mas nenhuma
tem vantagem decisiva. O resultado é um empate prático:
- ML-only: 0.566
- Flash:   0.533
- Pro:     0.527

A diferença de ~4pp não é estatisticamente significativa dada a variância do CV.

## Implicações para a tese

### 1. Confirma a hipótese de complementaridade seletiva

LLMs não são universalmente úteis para detecção de mecanismo de missing — são
especificamente valiosos quando as **features estatísticas são inconclusivas**
(caso MCAR). Isso sugere uma arquitetura de ensemble seletivo:

```
Se (confiança_estatística < threshold):
    usar features ML + LLM
Senão:
    usar apenas features ML
```

### 2. Flash > Pro sugere que o gargalo não é reasoning

O modelo Pro (mais caro, mais lento, com reasoning mais sofisticado) não
superou o Flash. Isso indica que:
- O **formato do prompt e a estrutura da metadata** são mais importantes que
  a capacidade de reasoning do LLM
- Investir em melhor engenharia de features/prompt traria mais retorno que
  usar um modelo maior

### 3. Feature importance confirma a dominância estatística

Em todos os três experimentos, as features CAAFE e estatísticas ocupam os
top 5-7 de importância. A melhor feature LLM (`llm_ctx_domain_confidence`)
fica consistentemente em 7o-10o lugar com importância ~0.034-0.035.

As features LLM contribuem, mas como **complemento marginal**, não como
fonte primária de discriminação.

### 4. MAR é o principal desafio

MAR é consistentemente a classe mais difícil (F1 = 0.36-0.42), apesar de ter
o sinal estatístico mais forte. Isso pode indicar:
- Confusão com MNAR em datasets onde ambos os mecanismos co-ocorrem
- Necessidade de features mais sofisticadas para distinguir MAR de MNAR
  (ambos envolvem dependência, mas de variáveis observadas vs não-observadas)
- O benchmark v2b pode ter datasets com rótulos ambíguos na fronteira MAR/MNAR
