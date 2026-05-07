# Infraestrutura — bugs corrigidos para a Fase 12

A re-execução exigiu corrigir resíduos do refactor da Fase 11 (commit
`1f47a54`, "reorganize repo into standard Python package"). Este documento
registra o que foi consertado para que a sessão seja reproduzível.

## 1. Paths legados no `paths.py`

**Sintoma:** `extract_features` reclamava `Pasta não encontrada:
.../Dataset/real_data/processado_chunks/MCAR`. O refactor moveu os dados
para `data/real/processed/` e os outputs para `results/`, mas
`src/missdetect/utils/paths.py` ainda referenciava a estrutura antiga.

**Correção:**

```diff
- OUTPUT_BASE = os.path.join(BASE_DIR, "Output", "v2_improved")
+ OUTPUT_BASE = os.path.join(BASE_DIR, "results")

  "real": {
-   "MCAR": os.path.join(BASE_DIR, "Dataset", "real_data", "processado_chunks", "MCAR"),
-   "MAR":  os.path.join(BASE_DIR, "Dataset", "real_data", "processado_chunks", "MAR"),
-   "MNAR": os.path.join(BASE_DIR, "Dataset", "real_data", "processado_chunks", "MNAR"),
+   "MCAR": os.path.join(BASE_DIR, "data", "real", "processed_chunks", "MCAR"),
+   "MAR":  os.path.join(BASE_DIR, "data", "real", "processed_chunks", "MAR"),
+   "MNAR": os.path.join(BASE_DIR, "data", "real", "processed_chunks", "MNAR"),
  },
```

(O caminho dos sintéticos também foi atualizado para `data/synthetic/`.)

## 2. Bootstraps não regenerados após refactor

**Sintoma:** mesmo após corrigir o `paths.py`, `data/real/processed_chunks/`
não existia — só havia os arquivos consolidados em `data/real/processed/`
(um arquivo por dataset, ex: `MCAR_boys_hc.txt` com 749 linhas). O
pipeline espera bootstraps individuais (`MCAR_boys_hc_boot001.txt`...
`_boot050.txt`) gerados por `subdividir_dados_reais.py`.

**Causa:** o refactor moveu os arquivos finais consolidados, mas o pipeline
de bootstrap (`subdividir_dados_reais.py`) ainda apontava para os caminhos
antigos.

**Correção:** atualizar `src/missdetect/data_generation/subdividir_dados_reais.py`:

```diff
  BASE = os.path.dirname(os.path.abspath(__file__))
- PROCESSADO = os.path.join(BASE, "..", "Dataset", "real_data", "processado")
- OUTPUT = os.path.join(BASE, "..", "Dataset", "real_data", "processado_chunks")
+ REPO_ROOT = os.path.normpath(os.path.join(BASE, "..", "..", ".."))
+ PROCESSADO = os.path.join(REPO_ROOT, "data", "real", "processed")
+ OUTPUT = os.path.join(REPO_ROOT, "data", "real", "processed_chunks")
```

E rodar uma vez:
```bash
uv run python -m missdetect.data_generation.subdividir_dados_reais
```

Resultado: 1.593 arquivos de bootstrap em `data/real/processed_chunks/`
(298 MCAR + 645 MAR + 650 MNAR — distribuição condizente com 6+13+13
datasets × ~50 bootstraps cada, descontando filtragem por
`missing_rate ≥ 1%`).

## 3. 10 datasets faltando em `real_datasets_metadata_neutral.json`

**Sintoma:** o metadata neutral (usado pelo extrator `context_aware` Flash/Pro)
tinha apenas 22 entradas. Os 10 datasets adicionados nas Fases 9-10
(`boys_hc`, `boys_hgt`, `brandsma_apr`, `brandsma_lpr`, `support2_pafi`,
`nhanes_cadmium`, `nhanes_cotinine`, `nhanes_mercury`, `support2_albumin`,
`support2_bilirubin`) não tinham descrições neutras.

**Risco:** sem entradas, o extrator `context_aware` ou crashava ou caía em
caminho default sem domain prior — invalidando a comparação com Step 1
da Fase 6.

**Correção:** acrescentei 10 entradas seguindo o template padrão do JSON
(domain, source, x0_variable, x0_units, x0_description, x0_typical_range,
predictors X1-X4, missing_context neutro padronizado, original_missing_rate,
n_total_original, _metadata_variant: "neutral"). Total agora: 32 entradas
cobrindo todos os 32 datasets do benchmark v2b.

Os predictors foram escolhidos com base no source dataset original
(ex: para `MCAR_boys_hc` → age/hgt/wgt/bmi do `mice::boys`; para
`MNAR_nhanes_cadmium` → age/gender/bmi/smoke_indicator do NHANES PBCD_J).
Nenhuma entrada revela o mecanismo na descrição (`_metadata_variant: neutral`
mantido).

## 4. `.env` ausente no workspace

**Sintoma:** `ModuleNotFoundError: No module named 'langchain_google_genai'`
no primeiro run do Flash. Resolvido instalando `[llm]` extras
(`uv pip install -e ".[llm,boosting]"`).

Em seguida: `GEMINI_API_KEY` retornava string vazia. O `extract_features.py`
faz `load_dotenv(os.path.join(V2_DIR, ".env"))` apontando para
`src/missdetect/.env`, mas não existia.

**Correção:** copiei o `.env` do projeto original
(`/Users/jonathan/Documents/ITA/ITA-Mestrado/IC - ITA 2/Scripts/v2_improved/.env`)
para ambos os locais que o pipeline procura:
- `<repo>/.env` (raiz; usado por `reproducibility.md`)
- `<repo>/src/missdetect/.env` (lido por `extract_features.py`)

O `.env` está corretamente listado em `.gitignore` (verificar antes de
commitar).

## 5. Filtro `--datasets-include` continua compatível

Sem mudança. O regex `_BOOT_RE = r"_boot\d+\.txt$"` strip do sufixo de
bootstrap funciona com a estrutura nova (`MCAR_boys_hc_boot042.txt` →
`MCAR_boys_hc`), batendo com as entradas de `datasets_v2b_32.txt`.

## 6. Discrepância: docs vs flag canônica `--llm-approach`

`docs/reproducibility.md` instrui `--llm-approach context_aware`, mas o
código aceita apenas `--llm-approach context` (ver
`src/missdetect/utils/args.py:8`, tupla `LLM_APPROACHES`). Não corrigi a
documentação nesta sessão (fora do escopo do experimento), mas registrei
no plano para correção futura.

## 7. Lista de arquivos novos / modificados

| Arquivo | Tipo |
|---|---|
| `src/missdetect/utils/paths.py` | modificado (paths novos) |
| `src/missdetect/data_generation/subdividir_dados_reais.py` | modificado (paths novos) |
| `src/missdetect/metadata/real_datasets_metadata_neutral.json` | modificado (+10 entradas) |
| `src/missdetect/metadata/datasets_v2b_32.txt` | novo (lista de 32 datasets) |
| `data/real/processed_chunks/{MCAR,MAR,MNAR}/` | gerado (1.593 bootstraps) |
| `.env`, `src/missdetect/.env` | gerado (gitignored, copiado do projeto antigo) |
| `results/step12_ml_only_v2b_32datasets/` | gerado (saída do experimento) |
| `results/step12_flash_neutral_v2b_32datasets/` | gerado (em andamento) |

## 8. Verificação antes de re-executar a Fase 12 do zero

```bash
# 1. Paths corretos
uv run python -c "from missdetect.utils.paths import DATASET_PATHS, OUTPUT_BASE; \
  import os; \
  assert os.path.exists(OUTPUT_BASE); \
  for t,d in DATASET_PATHS.items(): \
    [print(t,m,os.path.exists(p),len(os.listdir(p)) if os.path.exists(p) else 0) for m,p in d.items()]"

# 2. Metadata cobre 32 datasets
uv run python -c "import json; \
  m = json.load(open('src/missdetect/metadata/real_datasets_metadata_neutral.json')); \
  assert len(m) == 32, f'esperado 32, got {len(m)}'"

# 3. Bootstraps gerados
ls data/real/processed_chunks/MCAR/ | wc -l   # esperado: ~298
ls data/real/processed_chunks/MAR/ | wc -l    # esperado: ~645
ls data/real/processed_chunks/MNAR/ | wc -l   # esperado: ~650

# 4. .env carregável
uv run python -c "from dotenv import load_dotenv; import os; \
  load_dotenv('src/missdetect/.env'); \
  assert os.getenv('GEMINI_API_KEY'), 'falta GEMINI_API_KEY'"
```
