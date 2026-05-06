# Cross-validation do Bayes/KDE — resultado e aprendizado

**Data:** 2026-05-04
**Artefato:** `data/calibration.json`
**Comando executado:**

```bash
uv run python -m missdetect.calibrar_protocolo \
  --n-per-class 100 \
  --n-permutations 200 \
  --output-dir data \
  --checkpoint data/calibration_progress_sequential_backup.csv \
  --seed 42
```

O checkpoint já continha os 300 sintéticos processados, então a execução apenas recalculou thresholds, métricas e o novo bloco `validation_metrics.bayes_cv`.

## O que foi implementado

`src/missdetect/calibrar_protocolo.py` agora calcula `bayes_cv` com 5-fold CV estratificado:

1. Divide os 300 vetores sintéticos em 5 folds preservando MCAR/MAR/MNAR.
2. Em cada fold, ajusta os KDEs só nos 4/5 folds de treino.
3. Prediz os exemplos do fold deixado fora.
4. Salva accuracy global, desvio entre folds, accuracies por fold, confusion matrix e recall por classe em `data/calibration.json`.

## Resultado

| Métrica | Valor |
|---|---:|
| Bayes treino=teste | 78,3% |
| Bayes 5-fold CV | **59,0% ± 6,0%** |
| Queda estimada por overfitting | **19,3 pp** |

Accuracies por fold:

```text
55,0%, 63,3%, 68,3%, 51,7%, 56,7%
```

Confusion matrix do CV:

```text
Predito →      MCAR   MAR   MNAR
Verdadeiro ↓
MCAR            50     4     46
MAR             11    80      9
MNAR            49     4     47
```

Recall por classe:

| Classe | Recall CV |
|---|---:|
| MCAR | 50,0% |
| MAR | 80,0% |
| MNAR | 47,0% |

## O que aprendemos

1. O número de 78,3% não deve ser reportado como desempenho preditivo honesto. Ele mede ajuste nos mesmos sintéticos usados para fitar o KDE e superestima a generalização.
2. A estimativa honesta fora da amostra é 59,0% ± 6,0%. Isso é só moderadamente melhor que acaso em 3 classes balanceadas (33,3%).
3. MAR é o único mecanismo que generaliza razoavelmente. A camada B (`auc_obs`, `mi_max`) tem sinal real para dependência da máscara em variáveis observadas.
4. MCAR e MNAR continuam quase indistinguíveis fora da amostra: 46% dos MCAR viram MNAR e 49% dos MNAR viram MCAR.
5. O gargalo não é só overfitting do KDE. A raiz principal é falta de sinal observável para MNAR, já apontada em `04_ANALISE_ROBUSTA.md`.
6. Para a dissertação, o resultado correto é um achado negativo: o protocolo v2 é útil para quantificar incerteza e detectar MAR, mas não valida MNAR de forma confiável apenas com dados observados.

## Implicação para próximos passos

P3 pode seguir, mas deve usar `bayes_cv` como métrica honesta dos sintéticos. Antes de defender ganhos fortes de rótulos v2, é melhor executar P5: substituir os scores MNAR mortos (`caafe_quantile_ratio`, `caafe_tail_asym`) por evidências mais informativas, ou tratar MNAR como hipótese de domínio/sensibilidade em vez de classificação estatística observável.
