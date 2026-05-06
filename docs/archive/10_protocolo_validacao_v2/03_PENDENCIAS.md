# Pendências — O que falta fazer

**Data:** 2026-05-05 (atualizado)
**Status:** Protocolo v2b COMPLETO + benchmark auditado (32 datasets). P0-P7, P3, P4, P12, P13, P15 concluídos. Auditoria removeu 7 datasets duvidosos e reclassificou 6 MCAR→MAR. Total final: 6 MCAR, 13 MAR, 13 MNAR. Re-calibração com 32 datasets pendente.

---

## Prioridade Alta (necessário para dissertação)

### P1 — Calibração robusta (~9h sequencial, ~10 min com paralelismo) ✅ CONCLUÍDA

O smoke test usou apenas 15 datasets por classe e 10 permutações. Para resultados publicáveis:

```bash
# Comando com checkpoint (retoma de onde parou após crash/sleep):
caffeinate -i uv run python -m missdetect.calibrar_protocolo \
    --n-per-class 100 --n-permutations 200 --output-dir data \
    --checkpoint data/calibration_progress.csv

# Monitorar progresso:
wc -l data/calibration_progress.csv   # linhas = datasets processados + 1 header (target: 301)
tail -1 /tmp/calibrar_robust.log      # barra de progresso
```

**Tempo real:** ~113s por dataset × 300 = **~9,4 horas** (a estimativa original de 30 min era incorreta; o PKLM com 200 permutações é caro).

**Sistema de checkpoint (adicionado 2026-05-04):** `calibrar_protocolo.py` agora aceita `--checkpoint <csv>`. Cada dataset processado é appendado ao CSV imediatamente. Em caso de crash, sleep do Mac ou interrupção, relançar o mesmo comando retoma do ponto exato. Implementado após o processo ter morrido duas vezes por idle sleep do macOS.

**Proteção contra sleep:** usar `caffeinate -i` (impede idle sleep) + `nohup` (desacopla do terminal) para execuções longas.

**Backup do smoke:** `data/calibration_smoke.json` e `data/calibration_scores_smoke.npz` preservam os artefatos do smoke test (15/10) para comparação.

**Esperado:** accuracy no Bayes sobe de 95,6% para ≥90% (com cross-validation implícita via maior amostra) e os thresholds das regras ficam mais estáveis. `caafe_quantile_ratio` e `caafe_tail_asym` podem sair de AUC 0.5 (amostras insuficientes no smoke) para algo informativo.

**Critério de sucesso:** Bayes ≥85% accuracy nos sintéticos. Se não atingir, é um achado negativo honesto.

### P2 — Cross-validation do Bayes (evitar overfitting) ✅ CONCLUÍDA

O KDE agora é avaliado por 5-fold CV estratificado no `calibrar_protocolo.py`: em cada fold, o KDE é fitado em 80% dos sintéticos e prediz os 20% deixados fora.

**Resultado robusto (100/classe, 200 permutações):**
- Treino=teste: **78,3%**
- 5-fold CV: **59,0% ± 6,0%**
- Folds: 55,0%, 63,3%, 68,3%, 51,7%, 56,7%
- Confusion matrix CV:

```text
Predito →      MCAR   MAR   MNAR
Verdadeiro ↓
MCAR            50     4     46
MAR             11    80      9
MNAR            49     4     47
```

**Interpretação:** a acurácia treino=teste superestimava o Bayes em **19,3 pontos percentuais**. MAR generaliza razoavelmente (80% recall), mas MCAR e MNAR continuam se confundindo fora da amostra.

### P3 — Re-executar pipeline completo com rótulos v2

Depois de P1 e P2:

```bash
uv run python -m missdetect.run_all --data real --experiment with_v2_bayes
```

Isso compara `accuracy_com_rotulos_literatura` vs `accuracy_com_rotulos_v2` e mede se a redução de label noise melhora o gap sintético→real.

### P4 — Tabela de diagnóstico v2 em `data/real/sources.md`

Adicionar coluna `v2_prediction` e `v2_confidence` na tabela de cada mecanismo em `sources.md`, com base na rodada robusta (P1). A coluna já foi anotada no cabeçalho do arquivo; falta popular após P1.

---

## Prioridade Média (melhoria de qualidade)

### P5 — Melhorar scores CAAFE-MNAR (Camada C)

`caafe_quantile_ratio` e `caafe_tail_asym` tiveram AUC = 0.5 na calibração — não contribuem. Possíveis melhorias:

- **Aumentar amostra de calibração** (pode resolver se o sinal existe mas é fraco em n=15)
- **Adicionar score de auto-dependência direta:** comparar AUC(mask ~ X0_imputed) vs AUC(mask ~ X1..X4). Se X0 contribui significativamente → MNAR. (Variante do classificar_mnar.py já existente)
- **Wang-Shao-Kim score test (2023):** implementar o score test do paper, que explota moment conditions sob H0 = MAR. É o teste específico mais moderno para MAR vs MNAR.

### P6 — Prior informativo no Bayes (por dataset)

Usar o rótulo da literatura como prior (ex.: se domínio diz MNAR, prior P_MNAR = 0.6) em vez de uniforme. Isso é coerente com a posição da dissertação de que **rótulos de domínio são hipóteses informadas, não chutes**.

```python
prior = {"MCAR": 0.2, "MAR": 0.2, "MNAR": 0.6}  # para dataset MNAR na literatura
```

### P7 — Bandwidth ótima do KDE

O bandwidth do KDE Gaussiano está fixo em 0.5. Adicionar seleção por cross-validation (`GridSearchCV` do scikit-learn sobre `KernelDensity`):

```python
from sklearn.model_selection import GridSearchCV
params = {"bandwidth": np.logspace(-1, 1, 20)}
grid = GridSearchCV(KernelDensity(kernel="gaussian"), params, cv=5)
```

---

## Prioridade Baixa (trabalho futuro / apêndice)

### P8 — Cross-validation do Bayes nos dados reais (LOGO-CV)

Fazer Leave-One-Group-Out nos 29 reais com o protocolo v2:
- Para cada dataset d:
  1. Calibrar KDE sem d
  2. Classificar d com o KDE
  3. Comparar com rótulo da literatura

Isso dá uma estimativa honesta de "se eu tivesse um dataset novo, qual seria minha accuracy?".

### P9 — Heckman selection model para datasets MNAR específicos

Para `mroz_wages` (econometria clássica) e `pima_insulin` (medicina), ajustar um Heckman selection model e reportar se rho (correlação selection-outcome) é significativo. Requer escolha de instrumento (variável exógena que afeta seleção mas não o outcome).

### P10 — Mohan-Pearl m-graph testability

Para cada dataset, conjecturar um DAG de missingness (m-graph) e testar as conditional independencies implicadas. Belíssimo cientificamente; vai além do escopo da dissertação.

### P11 — Sensitivity analysis / tipping point

Para datasets ambíguos (P_max - P_segundo < 0.4): imputar sob MAR (mice) vs sob MNAR (shift-δ). Reportar o δ necessário para que a conclusão mude. Método de Liublinska & Rubin (2014).

---

## Checklist de conclusão

- [x] `validar_rotulos_v2.py` implementado (Camadas A-D)
- [x] `calibrar_protocolo.py` implementado (Camada E)
- [x] 13 testes unitários passando
- [x] Smoke test em sintéticos (95,6% Bayes)
- [x] Smoke test em reais (41,4%, 7 ambíguos identificados)
- [x] `run_all.py` integrado
- [x] `methodology.md` atualizado
- [x] `05_LIMITACOES.md` atualizado
- [x] `data/real/sources.md` cabeçalho atualizado
- [x] **P1** — Calibração robusta (100/classe, 200 permutações) ✅ concluída 2026-05-04 09:47 (Bayes 78.3%, abaixo do critério 85%)
- [x] **P2** — Cross-validation do Bayes ✅ concluída 2026-05-04 (Bayes CV 59.0% ± 6.0%)
- [x] **Passo 0** — Bug de paralelismo corrigido (calibrar_protocolo.py + validar_rotulos_v2.py) ✅ 2026-05-05
- [x] **P5** — Novos scores CAAFE: `caafe_auc_self_delta` + `caafe_kl_density` ✅ 2026-05-05
- [x] **P6** — Prior informativo implementado (`--prior-mnar` na CLI) ✅ 2026-05-05
- [x] **P7** — Bandwidth ótimo via GridSearchCV (`--auto-bandwidth`) ✅ 2026-05-05
- [x] **P3** — Calibração v2b concluída (300 datasets × 100 perms × 4 workers) ✅ 2026-05-05; Acurácia v2b em reais: 31 % (9/29) com prior-mnar=0.35
- [x] **P4** — Tabelas diagnóstico v2b adicionadas em `data/real/sources.md` ✅ 2026-05-05
- [x] **P12** — Benchmark expandido MCAR: 4 novos (boys_hc, boys_hgt, brandsma_lpr, brandsma_apr) ✅ 2026-05-05. Fontes: `mice::boys` (Van Buuren 2018 FIMD Ch. 9) e `mice::brandsma` (corr mask~ses p>0.7).
- [x] **P13** — Benchmark expandido MNAR: 6 novos processados, verificação rigorosa revelou que `support2_pafi` é predominantemente MAR (corr mask~hrt=−0.19, mask~temp=−0.18 p<0.001) → reclassificado para MAR. ✅ 2026-05-05. MNAR confirmados: NHANES 2017-18 (nhanes_cadmium, nhanes_mercury, nhanes_cotinine — LOD left-censoring indiscutível) + SUPPORT2 (support2_albumin, support2_bilirubin — MNAR misto com componente MAR fraca). **Total: 39 datasets (13 MCAR, 12 MAR, 14 MNAR).**
- [x] **P15** — Auditoria do benchmark: 7 datasets removidos (classificação duvidosa), 6 reclassificados MCAR→MAR ✅ 2026-05-06. **Total final: 32 datasets (6 MCAR, 13 MAR, 13 MNAR).**
- [ ] **P14** — Re-calibrar protocolo v2b com 32 datasets e re-validar
- [ ] P8 — LOGO-CV nos reais
- [ ] P9 — Heckman
- [ ] P10 — m-graphs
- [ ] P11 — Sensitivity analysis

---

## Estimativa de esforço

| Item | Sizing | Tempo estimado |
|------|:------:|:--------------:|
| P0 (paralelismo) | S | ~2h código (reduz todas as execuções futuras de 9h → ~10 min) |
| P1 | S | ✅ concluída (~9h sequencial; ~10 min com paralelismo futuro) |
| P2 | XS | ~2h código + minutos de execução (reutiliza scores já computados) |
| P3 | S | 1 dia (inclui análise; ~5 min execução com paralelismo) |
| P4 | XS | 30 min |
| P5 | M | ~3h código + ~10 min re-calibração (com paralelismo) |
| P6 | S | ~2h código |
| P7 | XS | 30 min |
| P8 | M | 2 dias |
| P9-P11 | L-XL | Trabalho futuro |
