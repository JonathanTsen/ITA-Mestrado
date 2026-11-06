"""Baseline de regressão logística simples com quatro estatísticas interpretáveis (passo 39).

Mostra quanto o conjunto completo de descritores acrescenta sobre um modelo mínimo. As quatro estatísticas são
descritores já calculados pelo 34b (mesmo cálculo, mesma instância): a taxa de ausência, o AUC de M a partir de Z
(boosting, CV), o desvio do AUC de M a partir de Ŷ(Z) e a assimetria de Y observado. O modelo é avaliado pelo
avaliador único (`evaluate.run_method`) como solução direta com aprendiz `logreg`.
"""

LOGIT_SIMPLE_FEATURES = (
    "g0_missing_rate",
    "g2_cv_auc_boosting",
    "g3_auc_mask_from_yhat_dev",
    "g1_skewness",
)
