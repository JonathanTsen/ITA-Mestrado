# Plano de paralelização — Protocolo v2

**Data:** 2026-05-04
**Máquina:** macOS, 12 cores (Apple Silicon)
**Estado atual:** 300 datasets × ~113s/dataset = ~9.4h sequencial
**Meta:** < 1h com paralelismo máximo

---

## Diagnóstico de tempo por dataset (~113s)

```
validate_one(df, n_permutations=200)
├── layer_a_mcar(df, 200)              ~55s  (48% do total)
│   ├── little_mcar_test(df)             ~0.1s  (desprezível)
│   ├── pklm_test(df, 200)              ~53s   (gargalo A)
│   │   ├── _compute_kl_divergence()      ~0.2s  (1× observado)
│   │   └── for i in range(200):          ~53s   (200× permutação)
│   │       └── _compute_kl_divergence()   ~0.26s cada
│   └── levene_stratified(df)            ~0.01s (desprezível)
│
├── layer_b_mar(df, 200)               ~57s  (50% do total)
│   ├── auc_mask_from_xobs(df, 200)    ~56s   (gargalo B)
│   │   ├── _cv_auc(X, mask, seed)       ~0.25s (1× observado)
│   │   └── for i in range(200):         ~55s   (200× permutação)
│   │       └── _cv_auc(X, perm, seed)    ~0.27s cada
│   └── mutual_info_mask_xobs(df)       ~0.5s  (desprezível)
│
└── layer_c_mnar(df)                   ~0.1s  (desprezível)
```

**95% do tempo está em 2 loops de 200 permutações** (PKLM + AUC mask).

---

## 4 níveis de paralelismo identificados

### Nível 1 — Entre datasets (loop externo)

**Arquivo:** `calibrar_protocolo.py` → `_collect_scores()`, linha 124
**Arquivo:** `validar_rotulos_v2.py` → `_main()`, linha 475

```
ANTES:  for fname in remaining:     # 300 datasets sequenciais
DEPOIS: ProcessPoolExecutor(N):     # N datasets simultâneos
```

- Cada dataset é 100% independente
- Pré-requisito: RF `n_jobs=1` (senão over-subscription)
- Checkpoint: coletar resultados em batch, não append por linha

| Workers | Cores/worker | Speedup teórico | Tempo estimado |
|:---:|:---:|:---:|:---:|
| 4 | 3 | ~4× | 2.3h |
| 6 | 2 | ~6× | 1.6h |
| 10 | 1 | ~8× | 1.2h |

### Nível 2 — Entre camadas (A ‖ B dentro de cada dataset)

**Arquivo:** `validar_rotulos_v2.py` → `validate_one()`, linha 363

```
ANTES:                            DEPOIS:
a = layer_a_mcar(df, 200)  # 55s     with ThreadPoolExecutor(2):
b = layer_b_mar(df, 200)   # 55s         fut_a = submit(layer_a_mcar, df, 200)
c = layer_c_mnar(df)       # 0.1s        fut_b = submit(layer_b_mar, df, 200)
# total: 110s                        c = layer_c_mnar(df)
                                      a, b = fut_a.result(), fut_b.result()
                                      # total: ~57s
```

- A e B são independentes (não compartilham estado)
- Speedup: ~1.8× por dataset (110s → 57s)
- Funciona com ou sem nível 1

### Nível 3 — Permutações dentro de cada camada (MAIOR impacto unitário)

**Arquivo:** `validar_rotulos_v2.py` → `auc_mask_from_xobs()`, linha 197
**Arquivo:** `baselines/pklm.py` → `pklm_test()`, linha 91

```
ANTES (auc_mask_from_xobs):              DEPOIS:
aucs_perm = np.empty(200)               seeds = rng.randint(0, 2**31, size=200)
for i in range(200):                     def _one_perm(seed):
    m = mask.copy()                          m = mask.copy()
    rng.shuffle(m)                           np.random.RandomState(seed).shuffle(m)
    aucs_perm[i] = _cv_auc(X, m, ...)       return _cv_auc(X, m, seed)
                                         aucs_perm = joblib.Parallel(n_jobs=N)(
                                             delayed(_one_perm)(s) for s in seeds)
```

```
ANTES (pklm_test):                       DEPOIS:
kl_permuted = np.zeros(200)              seeds = rng.randint(0, 2**31, size=200)
for i in range(200):                     def _one_pklm_perm(seed):
    mask_shuffled = mask.copy()              m = mask.copy()
    rng.shuffle(mask_shuffled)               np.random.RandomState(seed).shuffle(m)
    kl_permuted[i] = _compute_kl(...)        return _compute_kl_divergence(X, m, ...)
                                         kl_permuted = joblib.Parallel(n_jobs=N)(
                                             delayed(_one_pklm_perm)(s) for s in seeds)
```

- Cada permutação é 100% independente
- Seeds pré-geradas para reprodutibilidade
- Pré-requisito: RF `n_jobs=1` dentro de `_cv_auc` e `_compute_kl_divergence`

| Cores (n_jobs) | Tempo por loop | Speedup |
|:---:|:---:|:---:|
| 1 (atual) | ~55s | 1× |
| 4 | ~14s | 4× |
| 6 | ~10s | 5.5× |
| 12 | ~6s | 9× |

### Nível 4 — Dentro de cada RF (já existe, contra-produtivo)

**Arquivos:** `validar_rotulos_v2.py` linha 171, `pklm.py` linhas 121, 164

```
ANTES:  RandomForestClassifier(n_jobs=-1)   # usa 12 cores para 1 RF pequeno
DEPOIS: RandomForestClassifier(n_jobs=1)    # 1 core, libera os demais
```

- Datasets pequenos (1000 rows × 5 cols) → overhead de multiprocessing > ganho
- Com `n_jobs=1`, cada RF leva ~0.02s a mais, mas libera 11 cores para outros níveis
- **Mudança obrigatória** para habilitar níveis 1, 2 e 3

---

## Estratégia recomendada: tudo combinado (Nível 1 + 2 + 3 + 4)

### Alocação de cores (12 disponíveis)

```
┌─────────────────────────────────────────────────────┐
│ Nível 1: ProcessPoolExecutor(max_workers=4)         │
│                                                     │
│  Worker 0          Worker 1        Worker 2         │
│  ┌────────────┐   ┌──────────┐   ┌──────────┐      │
│  │ Nível 2:   │   │ Nível 2: │   │ Nível 2: │ ...  │
│  │ A ‖ B      │   │ A ‖ B    │   │ A ‖ B    │      │
│  │            │   │          │   │          │      │
│  │ Nível 3:   │   │ Nível 3: │   │ Nível 3: │      │
│  │ 3 cores    │   │ 3 cores  │   │ 3 cores  │      │
│  │ para perms │   │          │   │          │      │
│  └────────────┘   └──────────┘   └──────────┘      │
│                                                     │
│  4 workers × 3 cores = 12 cores                     │
└─────────────────────────────────────────────────────┘
```

- **Nível 4:** RF `n_jobs=1` em todos os 5 locais
- **Nível 3:** `joblib.Parallel(n_jobs=3)` nos 2 loops de permutação
- **Nível 2:** `ThreadPoolExecutor(2)` em `validate_one` (A ‖ B)
- **Nível 1:** `ProcessPoolExecutor(4)` no loop de datasets

### Tempo estimado por dataset

```
Sem paralelismo:   layer_a (55s) + layer_b (57s) = 112s
Com nível 2:       max(layer_a, layer_b) = 57s
Com nível 2+3:     max(55/3, 57/3) = ~19s
Com nível 1+2+3:   19s / 4 workers = ~5s effective throughput
```

### Tempo total estimado

| Cenário | Tempo | Speedup |
|---------|:---:|:---:|
| Atual (sequencial, RF n_jobs=-1) | **9.4h** | 1× |
| Só nível 4 (RF n_jobs=1, sem overhead) | ~9h | ~1× |
| Nível 3 (permutações ‖, 12 cores) | ~1.2h | 8× |
| Nível 2+3 (camadas ‖ + perms ‖) | ~45min | 12× |
| **Nível 1+2+3+4 (tudo)** | **~25–35min** | **~18×** |

---

## Plano de implementação (mudanças por arquivo)

### Mudança 1 — RF `n_jobs=1` (5 locais)

| Arquivo | Linha | Atual | Novo |
|---------|:---:|---|---|
| `validar_rotulos_v2.py` | 171 | `n_jobs=-1` | `n_jobs=1` |
| `baselines/pklm.py` | 121 | `n_jobs=-1` | `n_jobs=1` |
| `baselines/pklm.py` | 164 | `n_jobs=-1` | `n_jobs=1` |

Alternativa: aceitar `n_jobs` como parâmetro propagado, default 1.

### Mudança 2 — Permutações paralelas em `auc_mask_from_xobs`

**Arquivo:** `validar_rotulos_v2.py`, linhas 194–206

Antes:
```python
rng = np.random.RandomState(random_state)
auc_obs = _cv_auc(X, mask, rng.randint(2**31))
aucs_perm = np.empty(n_permutations)
for i in range(n_permutations):
    m = mask.copy()
    rng.shuffle(m)
    aucs_perm[i] = _cv_auc(X, m, rng.randint(2**31))
```

Depois:
```python
from joblib import Parallel, delayed

rng = np.random.RandomState(random_state)
auc_obs = _cv_auc(X, mask, rng.randint(2**31))
perm_seeds = rng.randint(0, 2**31, size=n_permutations)

def _single_perm_auc(seed, X, mask):
    rng_i = np.random.RandomState(seed)
    m = mask.copy()
    rng_i.shuffle(m)
    return _cv_auc(X, m, seed)

n_par = min(n_permutations, max(1, os.cpu_count() // _N_DATASET_WORKERS))
aucs_perm = np.array(
    Parallel(n_jobs=n_par, prefer="threads")(
        delayed(_single_perm_auc)(s, X, mask) for s in perm_seeds
    )
)
```

### Mudança 3 — Permutações paralelas em `pklm_test`

**Arquivo:** `baselines/pklm.py`, linhas 89–94

Antes:
```python
kl_permuted = np.zeros(n_permutations)
for i in range(n_permutations):
    mask_shuffled = mask.copy()
    rng.shuffle(mask_shuffled)
    kl_permuted[i] = _compute_kl_divergence(X, mask_shuffled, n_estimators, rng)
```

Depois:
```python
from joblib import Parallel, delayed

perm_seeds = rng.randint(0, 2**31, size=n_permutations)

def _single_perm_kl(seed, X, n_estimators):
    rng_i = np.random.RandomState(seed)
    m = mask.copy()
    rng_i.shuffle(m)
    return _compute_kl_divergence(X, m, n_estimators, rng_i)

n_par = min(n_permutations, max(1, os.cpu_count() // _N_DATASET_WORKERS))
kl_permuted = np.array(
    Parallel(n_jobs=n_par, prefer="threads")(
        delayed(_single_perm_kl)(s, X, n_estimators) for s in perm_seeds
    )
)
```

### Mudança 4 — Camadas A ‖ B em `validate_one`

**Arquivo:** `validar_rotulos_v2.py`, linhas 363–366

Antes:
```python
def validate_one(df, n_permutations=200, thresholds=None, bayes_kde=None):
    a = layer_a_mcar(df, n_permutations=n_permutations)
    b = layer_b_mar(df, n_permutations=n_permutations)
    c = layer_c_mnar(df)
```

Depois:
```python
from concurrent.futures import ThreadPoolExecutor

def validate_one(df, n_permutations=200, thresholds=None, bayes_kde=None):
    with ThreadPoolExecutor(max_workers=2) as pool:
        fut_a = pool.submit(layer_a_mcar, df, n_permutations)
        fut_b = pool.submit(layer_b_mar, df, n_permutations)
        c = layer_c_mnar(df)
        a = fut_a.result()
        b = fut_b.result()
```

### Mudança 5 — Datasets em paralelo em `_collect_scores`

**Arquivo:** `calibrar_protocolo.py`, linhas 110–143

Antes:
```python
for fname in tqdm(remaining, desc=mech, leave=False):
    df = pd.read_csv(p / fname, sep="\t")
    a = layer_a_mcar(df, n_permutations=n_permutations)
    b = layer_b_mar(df, n_permutations=n_permutations)
    c = layer_c_mnar(df)
    ...
```

Depois:
```python
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

_N_DATASET_WORKERS = min(4, mp.cpu_count() // 3)

def _process_one_file(args):
    fpath, mech, n_permutations = args
    df = pd.read_csv(fpath, sep="\t")
    a = layer_a_mcar(df, n_permutations=n_permutations)
    b = layer_b_mar(df, n_permutations=n_permutations)
    c = layer_c_mnar(df)
    scores = {"layer_a": a, "layer_b": b, "layer_c": c}
    vec = scores_to_vec(scores)
    row = {"file": fpath.name, "true_label": mech, **a, **b, **c}
    row.update(dict(zip(VEC_KEYS, vec, strict=False)))
    return row, mech, vec

with ProcessPoolExecutor(max_workers=_N_DATASET_WORKERS) as pool:
    tasks = [(p / fname, mech, n_permutations) for fname in remaining]
    for row, mech, vec in pool.map(_process_one_file, tasks):
        vecs_by_class[mech].append(vec)
        rows.append(row)
        # checkpoint: batch save every N results
```

**Nota:** Checkpoint muda de append-por-linha para batch-save periódico (a cada 10 resultados ou ao finalizar um mecanismo). Alternativa: usar `mp.Lock()` para sincronizar writes.

### Mudança 6 — Mesmo padrão no CLI de `validar_rotulos_v2.py`

**Arquivo:** `validar_rotulos_v2.py`, linhas 466–487

O loop `for fname in tqdm(files)` pode ser paralelizado com `ProcessPoolExecutor` da mesma forma que a mudança 5. Os 29 datasets reais levariam ~2–3 min em vez de ~45 min.

---

## Reprodutibilidade

### Problema
O código atual usa `rng.randint(2**31)` dentro de loops sequenciais. A sequência de seeds é determinística. Paralelizar muda a ordem de execução → seeds diferentes → resultados diferentes.

### Solução
Pré-gerar todas as seeds antes do loop paralelo:
```python
seeds = rng.randint(0, 2**31, size=n_permutations)
# agora cada permutação i recebe seeds[i] independente da ordem
```

Isso garante que os **mesmos seeds** são usados independente de quantos workers rodem, e os resultados numéricos são idênticos ao sequencial (dada a mesma seed[i] → mesmo shuffle → mesmo AUC).

---

## Checkpoint com paralelismo (Mudança 5)

### Problema
O checkpoint atual faz `pd.DataFrame([row]).to_csv(path, mode="a")` após cada dataset. Com múltiplos processos escrevendo no mesmo arquivo, há race condition.

### Opções

| Opção | Prós | Contras |
|-------|------|---------|
| **A. Lock de arquivo** (`filelock`) | Simples, append atômico | Dependência extra |
| **B. Batch save** (a cada N resultados) | Sem dependência extra | Perde N-1 resultados se crashar |
| **C. Um CSV por worker** → merge no final | Sem lock, sem perda | Mais complexo no merge |
| **D. Queue + writer thread** | Desacoplado, robusto | Mais código |

**Recomendação:** Opção B com N=1 via `mp.Lock()` (disponível no stdlib):
```python
lock = mp.Manager().Lock()

def _process_and_save(args):
    row, mech, vec = _process_one_file(args)
    with lock:
        pd.DataFrame([row]).to_csv(checkpoint_path, mode="a", header=not checkpoint_path.exists(), index=False)
    return row, mech, vec
```

---

## Dependências

`joblib` já está instalado (vem com scikit-learn). Nenhuma dependência nova necessária.

Verificar com:
```bash
uv run python -c "import joblib; print(joblib.__version__)"
```

---

## Ordem de implementação

```
1. Mudança 4 (RF n_jobs=1)         — 3 edits pontuais, zero risco
2. Mudança 3 (perms PKLM ‖)        — 1 função, maior ganho unitário
3. Mudança 2 (perms AUC ‖)         — 1 função, segundo maior ganho
4. Mudança 4 (A ‖ B)               — 1 função, complementar
5. Mudança 5 (datasets ‖ calibrar) — refatoração maior, checkpoint
6. Mudança 6 (datasets ‖ validar)  — espelha mudança 5
```

Após cada mudança, rodar:
```bash
uv run --extra dev python -m pytest tests/test_validar_rotulos_v2.py
uv run python -m missdetect.calibrar_protocolo --n-per-class 3 --n-permutations 5 --output-dir /tmp/test_par
```

---

## Teste de validação: comparar resultados paralelo vs sequencial

```bash
# Sequencial (baseline)
uv run python -m missdetect.calibrar_protocolo --n-per-class 5 --n-permutations 10 --output-dir /tmp/seq --seed 42

# Paralelo (após implementação)
uv run python -m missdetect.calibrar_protocolo --n-per-class 5 --n-permutations 10 --output-dir /tmp/par --seed 42

# Comparar
diff <(python -c "import json; print(json.dumps(json.load(open('/tmp/seq/calibration.json')), sort_keys=True, indent=2))") \
     <(python -c "import json; print(json.dumps(json.load(open('/tmp/par/calibration.json')), sort_keys=True, indent=2))")
```

Se a pré-geração de seeds for feita corretamente, o diff deve ser vazio (resultados idênticos).
