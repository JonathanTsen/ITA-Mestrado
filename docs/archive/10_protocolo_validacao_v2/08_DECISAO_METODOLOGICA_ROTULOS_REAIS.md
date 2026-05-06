# Decisão metodológica — uso de rótulos reais no artigo

**Data:** 2026-05-04  
**Contexto:** os rótulos reais iniciais foram curados manualmente com base em literatura/domínio. O protocolo v2 trouxe um método estatístico reproduzível para diagnóstico de mecanismo, mas com limitações.

---

## 1) Pergunta central

Devo substituir os rótulos reais da literatura pelo diagnóstico v2?

**Resposta curta:** não substituir de forma cega.  
**Resposta prática:** usar literatura como referência principal e v2 como evidência auxiliar de incerteza/disputa.

---

## 2) O que mudou de fato

1. **Os dados reais não mudaram** (mesmos arquivos/datasets).
2. **Os rótulos de literatura não foram sobrescritos**.
3. O que mudou foi a **predição do diagnóstico v2** e a forma de avaliação:
   - Bayes treino=teste em sintéticos: 78,3%
   - Bayes 5-fold CV em sintéticos: 59,0% ± 6,0%

Interpretação simples: a queda 78→59 veio de avaliação mais rigorosa (CV), não de mudança nos dados sintéticos.

---

## 3) Decisão para o artigo

### 3.1 Política de rótulo

- **Rótulo principal (ground truth operacional):** literatura/domínio (`MCAR=9`, `MAR=11`, `MNAR=9`).
- **Rótulo auxiliar (evidência empírica):** diagnóstico v2 robusto (`MCAR=9`, `MAR=14`, `MNAR=6`).

### 3.2 Regra de interpretação

- Quando literatura e v2 concordam: tratar como caso mais estável.
- Quando discordam: tratar como **rótulo disputável** (não como erro automático de um dos lados).
- Quando confiança do v2 é baixa: tratar como **ambíguo** e reportar incerteza.

### 3.3 Frase metodológica recomendada

> “Os rótulos reais de mecanismo foram definidos por curadoria de literatura/domínio e usados como referência principal. O protocolo v2 foi aplicado como validação empírica complementar e quantificação de incerteza, não como substituto automático dos rótulos de domínio.”

---

## 4) Justificativa técnica da decisão

1. Entre “intuição pura” e v2, o v2 é superior por ser reproduzível, auditável e baseado em testes/likelihood.
2. Porém, v2 ainda não tem desempenho suficiente para ser juiz único:
   - CV honesta em sintéticos: 59,0% ± 6,0%
   - Confusão forte MCAR↔MNAR fora da amostra
3. Logo, a escolha mais defensável cientificamente é **combinar** as duas fontes de evidência, explicitando limites.

---

## 5) Como reportar nos resultados

1. Reportar métricas separadas:
   - desempenho em sintéticos (CV),
   - concordância v2 vs literatura em reais.
2. Mostrar tabela com: `true_label`, `v2_prediction`, `v2_confidence`, `nota`.
3. Incluir seção de limitações:
   - MNAR não-identificável apenas por dados observados,
   - v2 como ferramenta de diagnóstico e sensibilidade, não oráculo.

---

## 6) Resumo executivo

- **Não voltar para v1/intuição.**
- **Não substituir literatura por v2 de forma automática.**
- **Usar literatura + v2 juntos**, com incerteza explícita, é a estratégia mais robusta para o artigo.
