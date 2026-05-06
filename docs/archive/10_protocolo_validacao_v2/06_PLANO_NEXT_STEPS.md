# Plano — Corrigir falhas do Protocolo v2 e próximos passos

**Data:** 2026-05-04
**Status:** Calibração robusta (100/200) concluída. Bayes treino=teste 78.3%, Bayes 5-fold CV 59.0% ± 6.0%, reais 41.4%.
**Referência:** Análise completa em [04_ANALISE_ROBUSTA.md](04_ANALISE_ROBUSTA.md) e CV em [07_CROSS_VALIDATION_BAYES.md](07_CROSS_VALIDATION_BAYES.md)
**Paralelismo:** Detalhes técnicos em [05_PLANO_PARALELISMO.md](05_PLANO_PARALELISMO.md)

---

## Contexto

A calibração robusta (100/200) completou e revelou 3 problemas críticos:
1. **Bayes 59.0% ± 6.0% em 5-fold CV** nos sintéticos (78.3% treino=teste era otimista)
2. **41.4%** nos reais (accuracy não melhorou vs smoke)
3. **MNAR→MCAR com confiança 1.0** em 4 datasets reais (falha catastrófica)

A análise exaustiva em `04_ANALISE_ROBUSTA.md` identificou 5 causas-raiz. Este plano endereça as corrigíveis (Causas 2, 3, 4) e documenta as fundamentais (Causas 1, 5).

A calibração robusta anterior levou **9.4h sequencial**. Com o paralelismo implementado no Passo 0, qualquer re-calibração futura leva **~30–45 min**.

---

## Diagnóstico resumido das causas

| # | Causa | Tipo | Prioridade |
|---|-------|------|------------|
| 1 | MCAR ↔ MNAR indistinguíveis no espaço de scores | **Fundamental** (Rubin 1976) | — |
| 2 | Dimensões 6-7 (CAAFE) mortas (constantes) | **Fixável** | Alta |
| 3 | Shift auc_obs/mi_max entre sintéticos e reais | **Fixável parcialmente** | Alta |
| 4 | Treino=teste no Bayes (sem cross-validation) | **Fixável** | Alta |
| 5 | MNAR com confiança 1.0 errada (KDE degenerado) | **Parcialmente fixável** | Média |

---

## Passos — em ordem de execução

### Passo 0 — Implementar paralelismo em todo o pipeline [~2h código]

**Por que é o primeiro:** Reduz o tempo de qualquer re-calibração futura de 9.4h → ~30 min. Todos os passos seguintes que requerem re-calibração (Passo 2) ou validação (Passo 6) se beneficiam.

**Máquina:** 12 cores (Apple Silicon).

**4 níveis de paralelismo a implementar:**

#### Nível 4 — RF `n_jobs=1` (pré-requisito, libera cores)

| Arquivo | Linha | Mudança |
|---------|:---:|---------|
| `src/missdetect/validar_rotulos_v2.py` | 171 | `n_jobs=-1` → `n_jobs=1` |
| `src/missdetect/baselines/pklm.py` | 121 | `n_jobs=-1` → `n_jobs=1` |
| `src/missdetect/baselines/pklm.py` | 164 | `n_jobs=-1` → `n_jobs=1` |

Com datasets pequenos (1000 rows × 5 cols), o overhead de multiprocessing do RF é maior que o ganho. `n_jobs=1` é mais rápido por chamada individual e libera todos os cores para paralelismo nos níveis superiores.

#### Nível 3 — Permutações paralelas (MAIOR impacto unitário)

**Arquivo `src/missdetect/validar_rotulos_v2.py` → `auc_mask_from_xobs()` (linhas 194–207):**

```python
# ANTES: loop sequencial (~55s)
aucs_perm = np.empty(n_permutations)
for i in range(n_permutations):
    m = mask.copy()
    rng.shuffle(m)
    aucs_perm[i] = _cv_auc(X, m, rng.randint(2**31))

# DEPOIS: paralelo com joblib (~6s com 12 cores)
from joblib import Parallel, delayed

perm_seeds = rng.randint(0, 2**31, size=n_permutations)

def _single_perm_auc(seed, X, mask):
    rng_i = np.random.RandomState(seed)
    m = mask.copy()
    rng_i.shuffle(m)
    return _cv_auc(X, m, seed)

aucs_perm = np.array(
    Parallel(n_jobs=-1, prefer="threads")(
        delayed(_single_perm_auc)(s, X, mask) for s in perm_seeds
    )
)
```

**Arquivo `src/missdetect/baselines/pklm.py` → `pklm_test()` (linhas 89–94):**

```python
# ANTES: loop sequencial (~53s)
kl_permuted = np.zeros(n_permutations)
for i in range(n_permutations):
    mask_shuffled = mask.copy()
    rng.shuffle(mask_shuffled)
    kl_permuted[i] = _compute_kl_divergence(X, mask_shuffled, n_estimators, rng)

# DEPOIS: paralelo com joblib (~6s com 12 cores)
from joblib import Parallel, delayed

perm_seeds = rng.randint(0, 2**31, size=n_permutations)

def _single_perm_kl(seed, X, mask, n_estimators):
    rng_i = np.random.RandomState(seed)
    m = mask.copy()
    rng_i.shuffle(m)
    return _compute_kl_divergence(X, m, n_estimators, rng_i)

kl_permuted = np.array(
    Parallel(n_jobs=-1, prefer="threads")(
        delayed(_single_perm_kl)(s, X, mask, n_estimators) for s in perm_seeds
    )
)
```

**Reprodutibilidade:** Seeds pré-geradas antes do loop garantem resultados idênticos independente do número de workers.

**Speedup por dataset:** ~113s → ~15s (7.5×)

#### Nível 2 — Camadas A ‖ B em paralelo

**Arquivo `src/missdetect/validar_rotulos_v2.py` → `validate_one()` (linhas 363–366):**

```python
# ANTES: sequencial (~110s)
a = layer_a_mcar(df, n_permutations=n_permutations)
b = layer_b_mar(df, n_permutations=n_permutations)
c = layer_c_mnar(df)

# DEPOIS: A e B em paralelo (~57s, ou ~8s com nível 3 também)
from concurrent.futures import ThreadPoolExecutor

with ThreadPoolExecutor(max_workers=2) as pool:
    fut_a = pool.submit(layer_a_mcar, df, n_permutations)
    fut_b = pool.submit(layer_b_mar, df, n_permutations)
    c = layer_c_mnar(df)
    a = fut_a.result()
    b = fut_b.result()
```

Camadas A e B são 100% independentes (não compartilham estado). Camada C é desprezível (~0.1s).

**Speedup por dataset (combinado com nível 3):** ~15s → ~8s (1.9×)

#### Nível 1 — Datasets em paralelo

**Arquivo `src/missdetect/calibrar_protocolo.py` → `_collect_scores()` (linhas 110–143):**

```python
# ANTES: 1 dataset de cada vez
for fname in tqdm(remaining, desc=mech, leave=False):
    df = pd.read_csv(p / fname, sep="\t")
    a = layer_a_mcar(df, n_permutations=n_permutations)
    ...

# DEPOIS: N datasets simultaneamente
from concurrent.futures import ProcessPoolExecutor

def _process_one_file(args):
    fpath, mech, n_permutations = args
    df = pd.read_csv(fpath, sep="\t")
    res = validate_one(df, n_permutations=n_permutations)
    vec = scores_to_vec({"layer_a": res["layer_a"], ...})
    return {"file": fpath.name, "true_label": mech, ...}, mech, vec

with ProcessPoolExecutor(max_workers=4) as pool:
    results = pool.map(_process_one_file, tasks)
```

**Mesmo padrão em `validar_rotulos_v2.py` → `_main()` (linhas 466–487)** para os 29 reais.

**Checkpoint com paralelismo:** Trocar append-por-linha por batch save com `multiprocessing.Lock()` ou salvar ao final de cada mecanismo.

**Speedup total (300 datasets):** 4 workers × ~8s/dataset = ~10 min (vs 9.4h original = **~56× speedup**)

#### Tabela-resumo de impacto

| Nível | Mudança | Arquivos | Speedup | Tempo 300 datasets |
|:---:|---|---|:---:|:---:|
| — | Sequencial (atual) | — | 1× | 9.4h |
| 4 | RF `n_jobs=1` | 2 arquivos, 3 edits | ~1× | 9h |
| 4+3 | + permutações ‖ | 2 funções | ~8× | ~1.2h |
| 4+3+2 | + camadas A‖B | 1 função | ~14× | ~40 min |
| **4+3+2+1** | **+ datasets ‖** | **2 loops** | **~56×** | **~10 min** |

**Dependências:** `joblib` já instalado (vem com scikit-learn). Nenhuma dependência nova.

**Verificação do Passo 0:**
```bash
# Teste rápido: 3 datasets/classe, 5 permutações — deve completar em <30s
uv run python -m missdetect.calibrar_protocolo \
    --n-per-class 3 --n-permutations 5 --output-dir /tmp/test_par --seed 42

# Comparar com sequencial para validar reprodutibilidade
# (se seeds pré-geradas corretamente, resultados idênticos)
```

---

### Passo 1 — Cross-validation do Bayes (P2) ✅ CONCLUÍDO

**Por que é o mais urgente:** O 78.3% é treino=teste; sem CV não sabemos a accuracy real. A dissertação não pode reportar accuracy sem CV.

**Paralelismo aplicável:** Não — são apenas 5 folds × KDE fit + predict nos 300 vetores já computados. Roda em segundos.

**Implementado:** `src/missdetect/calibrar_protocolo.py` agora adiciona `validation_metrics.bayes_cv` com 5-fold CV estratificado. O KDE é fitado em 4/5 dos vetores 10D e avaliado no 1/5 deixado fora.

**Checkpoint:** Reutiliza `data/calibration_progress.csv` (300 datasets já processados). Sem re-rodar coleta.

**Resultado observado:** CV accuracy **59.0% ± 6.0%**, abaixo da estimativa original de 70–75%.

**Aprendizado:** a diferença entre 78.3% treino=teste e 59.0% CV mostra overfitting de **19.3 pp**. MAR generaliza melhor (80% recall), mas MCAR e MNAR continuam se confundindo fora da amostra.

**Verificação:**
```bash
uv run python -c "import json; d=json.load(open('data/calibration.json')); \
  print('cv_accuracy=', d['validation_metrics']['bayes_cv']['accuracy'], \
        '±', d['validation_metrics']['bayes_cv']['accuracy_std'])"
```

---

### Passo 2 — Substituir dimensões mortas (P5 parcial) [~3h código + ~10 min re-calibração]

**Por que:** 2 de 10 dimensões são constantes (AUC=0.5). Substituí-las por scores informativos pode dar +3–5% nos sintéticos.

**Paralelismo aplicável:** A re-calibração após mudar o vetor usa o paralelismo do Passo 0 (~10 min vs 9h).

**O que fazer:** Adicionar 2 novos scores ao vetor:

**Score novo 1 — Auto-dependência AUC(mask ~ X0_imputed):**

```python
def auc_self_dependence(df, missing_col="X0", n_permutations=50):
    """Delta AUC quando X0 imputado é incluído como feature para prever mask."""
    # auc_with_x0: RF prevendo mask usando [X1..X4, X0_imputed]
    # auc_without_x0: já calculado na camada B (auc_obs)
    # delta = auc_with_x0 - auc_without_x0
    # Se delta > 0 → X0 contribui → evidência de MNAR
```

**Paralelismo dentro desta função:** Permutações para p-value usam `joblib.Parallel` (já implementado no Passo 0).

**Score novo 2 — Razão de densidades (proxy MNAR):**

```python
def density_ratio_test(df, missing_col="X0"):
    """KL-divergence entre distribuição de X0_obs nos grupos R=0/R=1 de Xi."""
    # Sem permutações — cálculo direto, ~0.01s
```

**Paralelismo:** Não necessário (cálculo instantâneo).

**Substituir no vetor:** `log1p(quantile_ratio)` → `auc_self_delta`, `tail_asym` → `kl_density_ratio`

**Arquivos a modificar:**
- `src/missdetect/validar_rotulos_v2.py`: 2 novas funções, atualizar `layer_c_mnar()`, `scores_to_vec()`, `VEC_KEYS`
- `src/missdetect/calibrar_protocolo.py`: atualizar campos do checkpoint
- `tests/test_validar_rotulos_v2.py`: atualizar/adicionar testes

**Cuidado:** Invalida `data/calibration_progress.csv` e `data/calibration_scores.npz`. Apagar e re-rodar:
```bash
rm data/calibration_progress.csv
caffeinate -i uv run python -m missdetect.calibrar_protocolo \
    --n-per-class 100 --n-permutations 200 --output-dir data \
    --checkpoint data/calibration_progress.csv --seed 42
# ~10 min com paralelismo (vs 9h sem)
```

---

### Passo 3 — Bandwidth otimizado via GridSearchCV (P7) [~30 min]

**Por que:** Bandwidth fixo em 0.5 pode não ser ótimo. Com Scott's rule, bw ≈ 0.72 para n=100 em 10D.

**Paralelismo aplicável:** `GridSearchCV` aceita `n_jobs=-1` nativamente:

```python
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

params = {"bandwidth": np.logspace(-1, 1, 20)}
best_bw = {}
for cls in ("MCAR", "MAR", "MNAR"):
    grid = GridSearchCV(
        KernelDensity(kernel="gaussian"), params, cv=5, n_jobs=-1
    )
    grid.fit(arrays[cls])
    best_bw[cls] = grid.best_params_["bandwidth"]
```

**Arquivo:** `src/missdetect/calibrar_protocolo.py` — adicionar `--auto-bandwidth`

---

### Passo 4 — Prior informativo por dataset (P6) [~2h]

**Por que:** 11/29 reais são ambíguos (conf < 0.4).

**Paralelismo aplicável:** Não diretamente (é uma mudança de lógica, não computacional).

**O que fazer:** Parametrizar prior em `diagnose_bayes`:

```python
def diagnose_bayes(scores, kde_by_class, prior=None):
    if prior is None:
        prior = {"MCAR": 1/3, "MAR": 1/3, "MNAR": 1/3}
    # posterior ∝ likelihood × prior
    p = {m: np.exp(log_lik[m] - log_max) * prior[m] for m in log_lik}
```

**Arquivo:** `src/missdetect/validar_rotulos_v2.py`

---

### Passo 5 — Documentar limitações e rótulos duvidosos [~1h]

**Paralelismo:** Não aplicável (documentação manual).

**O que fazer:**
1. Atualizar `data/real/sources.md` com colunas `v2_prediction`, `v2_confidence`, `nota`
2. Marcar os 4 MCAR→MAR como "rótulo disputável — evidência empírica de MAR"
3. Marcar os 4 MNAR→MCAR como "MNAR não-identificável estatisticamente"
4. Marcar os 5 ambíguos como "candidatos a sensitivity analysis"
5. Atualizar `03_PENDENCIAS.md`

---

### Passo 6 — Re-rodar pipeline completo com rótulos v2 (P3) [~5 min]

**Depende de:** Passos 0–4 concluídos.

**Paralelismo aplicável:** A validação dos 29 reais usa paralelismo do Passo 0 (nível 1: datasets ‖):
```bash
uv run python -m missdetect.validar_rotulos_v2 --data real --experiment v2_final \
    --calibration data/calibration.json --bayes-scores data/calibration_scores.npz \
    --n-permutations 200
# ~5 min com paralelismo (vs ~45 min sem)
```

Depois:
```bash
uv run python -m missdetect.run_all --data real --experiment with_v2_bayes
```

---

## Ordem de execução com tempos estimados

```
Passo 0 (paralelismo)  ← 2h código, reduz tudo que vem depois
  ↓
Passo 1 (CV)           ← concluído: 59.0% ± 6.0%
  ↓
Passo 5 (documentar)   ← 1h, sem código
  ↓
Passo 3 (bandwidth)    ← 30 min código + execução
  ↓
Passo 4 (prior)        ← 2h código
  ↓
Passo 2 (novos scores) ← 3h código + ~10 min re-calibração (graças ao Passo 0)
  ↓
Passo 6 (pipeline)     ← ~5 min execução (graças ao Passo 0)
```

**Tempo total estimado:** ~10h de código + ~25 min de execução (vs ~10h código + ~20h execução sem Passo 0).

---

## Mapa de paralelismo por passo

| Passo | Paralelismo | Impacto no tempo |
|:---:|---|---|
| **0** | **RF n_jobs=1, permutações ‖, camadas A‖B, datasets ‖** | **9.4h → 10 min** |
| 1 | Não necessário (opera em vetores já computados) | Minutos |
| **2** | **Re-calibração usa Passo 0** | **9h → 10 min** |
| **3** | **GridSearchCV `n_jobs=-1`** | **Nativo** |
| 4 | Não necessário (mudança de lógica) | Instantâneo |
| 5 | Não aplicável (documentação) | — |
| **6** | **Validação dos 29 reais usa Passo 0** | **45 min → 5 min** |

---

## Arquivos modificados por passo

| Arquivo | P0 | P1 | P2 | P3 | P4 | P5 | P6 |
|---------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `src/missdetect/validar_rotulos_v2.py` | ✓ | | ✓ | | ✓ | | |
| `src/missdetect/baselines/pklm.py` | ✓ | | | | | | |
| `src/missdetect/calibrar_protocolo.py` | ✓ | ✓ | ✓ | ✓ | | | |
| `tests/test_validar_rotulos_v2.py` | | | ✓ | | | | |
| `data/real/sources.md` | | | | | | ✓ | |
| `docs/.../03_PENDENCIAS.md` | | | | | | ✓ | |

---

## O que NÃO pode ser corrigido

1. **MCAR ↔ MNAR confusão:** Irredutível. Rubin (1976) prova que MNAR não é testável com dados observados. O protocolo v2 quantifica essa incerteza em vez de forçar decisão.

2. **Gap sintético→real (78% → 41%):** Parcialmente redutível via P5 (scores melhores) e P6 (prior informativo), mas o gap fundamental vem de rótulos potencialmente incorretos, complexidade dos mecanismos reais e confounders não modelados.

3. **Confiança 1.0 errada:** O KDE Gaussiano produz likelihoods com razões de 10^30+. Solução parcial: clipping da razão de likelihoods (cap confidence em 0.95).

---

## Verificação end-to-end

Após todos os passos:
1. Calibração completa em < 15 min (não 9h)
2. `calibration.json` contém `bayes_cv.accuracy` e `bayes_cv.accuracy_std`
3. CV accuracy documentada nos sintéticos: 59.0% ± 6.0%
4. Novos scores (auc_self_delta, kl_density_ratio) com AUC ROC > 0.5
5. `results/v2_robust/real/validacao_rotulos_v2/validacao_v2.csv` tem 29 linhas
6. `data/real/sources.md` tem colunas v2_prediction/v2_confidence populadas
7. Testes passam: `uv run --extra dev python -m pytest tests/test_validar_rotulos_v2.py`
