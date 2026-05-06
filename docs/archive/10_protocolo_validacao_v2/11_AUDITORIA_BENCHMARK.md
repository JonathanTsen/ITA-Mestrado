# Auditoria do benchmark — remoção de classificações duvidosas

**Data:** 2026-05-06

---

## Método

Cada um dos 39 datasets foi avaliado em dois eixos:
1. **Evidência de domínio**: a justificativa publicada para o mecanismo é específica e verificável?
2. **Concordância v2b**: o protocolo v2b concorda com a classificação de domínio?

Critério de remoção: evidência de domínio **vaga ou ausente** E v2b **discorda fortemente** (conf > 0.5 na direção oposta) ou dataset **ambíguo** (conf < 0.4 em qualquer direção).

Critério de reclassificação: evidência de domínio **consistente com outra categoria** E v2b **confirma** a outra categoria com alta confiança.

---

## Removidos (7 datasets)

| Dataset | Rótulo | Razão da remoção |
|:--|:--|:--|
| `creditapproval_a14` | MCAR | Campo anonimizado — impossível verificar mecanismo. v2b: MNAR (conf=1.0). |
| `echomonths_epss` | MCAR | n=130, apenas 14 missing. "Janela acústica insuficiente" pode ser MNAR (anatomia cardíaca). v2b: MNAR (AMBÍGUO, conf=0.36). |
| `autompg_horsepower` | MCAR | Apenas 6/398 missing (1.5%). Ausência de evidência ≠ evidência de MCAR. |
| `hearth_chol` | MAR | Domínio: "estado clínico observado" (MAR). v2b: MNAR (P=0.70). Colesterol pode não ser medido por razões dependentes do valor esperado. |
| `kidney_hemo` | MAR | Domínio: "severidade" (MAR). v2b: MNAR (AMBÍGUO, conf=0.23). Hemoglobina baixa pode causar a própria não-medição. |
| `colic_resprate` | MAR | Domínio: "severidade" (MAR). v2b: MNAR (P=0.77, conf=0.62). auc_self_delta alto indica X0 prediz a própria ausência. |
| `cylinderbands_varnishpct` | MNAR | Domínio: "quality-dependent" (vago). v2b: MAR (conf=0.99). AUC muito alto — missingness totalmente predizível. |

## Reclassificados MCAR → MAR (6 datasets)

| Dataset | Rótulo antigo | Evidência para MAR |
|:--|:--|:--|
| `oceanbuoys_humidity` | MCAR | Arquivo já nomeado MAR_*. v2b: MAR (conf=1.0). Falha do sensor correlaciona com condições ambientais. |
| `oceanbuoys_airtemp` | MCAR | Idem. |
| `hypothyroid_t4u` | MCAR | "Não ordenado rotineiramente" = test-ordering com base em TSH/TT4/FTI (MAR). v2b: MAR (conf=1.0). |
| `breastcancer_barenuclei` | MCAR | "Clerical gap" é vago para 16/699. v2b: MAR (conf=1.0). AUC alto. |
| `cylinderbands_bladepressure` | MCAR | "Sensor failure" mas falha correlaciona com condições de impressão. v2b: MAR (conf=1.0). |
| `cylinderbands_esavoltage` | MCAR | Mesma lógica. v2b: MAR (conf=0.77). |

## Resultado

| | Antes | Removidos | Reclassificados | Depois |
|:--|:--:|:--:|:--:|:--:|
| MCAR | 13 | −3 | −6 → MAR | **6** |
| MAR | 12 | −3 | +6 de MCAR | **13** ¹ |
| MNAR | 14 | −1 | — | **13** |
| **Total** | **39** | **−7** | **0** (net) | **32** |

¹ Inclui `support2_pafi` já reclassificado de MNAR na fase 10.
