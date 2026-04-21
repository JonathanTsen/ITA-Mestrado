"""
Self-Consistency Extractor — Multiple perspectives with CISC voting.

Five specialized perspectives analyze the same dataset independently:
  1. Statistical: purely data-driven, ignores domain
  2. Domain: domain expertise + collection process
  3. Process: data engineering / recording process
  4. Adversarial: argues AGAINST the default MAR hypothesis
  5. Censoring: censoring/truncation specialist

Aggregation via Confidence-Informed Self-Consistency (CISC):
  - Each perspective votes for a mechanism with a confidence weight
  - Final mechanism = weighted majority vote
  - Features capture vote distribution and agreement

References:
  - Wang et al. (2022) "Self-Consistency Improves CoT Reasoning" (NeurIPS)
  - "Confidence Improves Self-Consistency in LLMs" (ACL Findings 2025)
  - Du et al. (2023) "Multiagent Debate" (ICML 2024)

Extracted features (8):
  - llm_sc_domain_prior: Winning mechanism (MCAR=0, MAR=0.5, MNAR=1)
  - llm_sc_confidence: Aggregated CISC confidence
  - llm_sc_agreement: Fraction of perspectives agreeing with winner
  - llm_sc_vote_mcar: Proportion of MCAR-weighted votes
  - llm_sc_vote_mar: Proportion of MAR-weighted votes
  - llm_sc_vote_mnar: Proportion of MNAR-weighted votes
  - llm_sc_stats_consistency: Mean stats_consistency across perspectives
  - llm_sc_surprise: Mean surprise_factor across perspectives
"""

import hashlib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy import stats as sp_stats


# ======================================================
# SCHEMAS
# ======================================================

class PerspectiveResponse(BaseModel):
    """Output from a single perspective."""

    domain_mechanism_prior: str = Field(
        default="MCAR",
        description="Most likely mechanism: MCAR, MAR, or MNAR",
    )
    domain_confidence: float = Field(
        default=0.33, ge=0.0, le=1.0,
        description="Confidence in the classification",
    )
    stats_consistent_with_domain: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="1=statistics consistent with classification, 0=inconsistent",
    )
    surprise_factor: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="1=very surprising/unexpected, 0=expected",
    )
    reasoning: str = Field(
        default="",
        description="Short reasoning explanation",
    )


MECHANISM_TO_SCORE = {"MCAR": 0.0, "MAR": 0.5, "MNAR": 1.0}


# ======================================================
# MAIN CLASS
# ======================================================

class SelfConsistencyExtractor:
    """Self-Consistency extractor with 5 specialized perspectives and CISC voting."""

    PERSPECTIVE_NAMES = ["statistical", "domain", "process", "adversarial", "censoring"]

    def __init__(
        self,
        model_name: str,
        provider: str = "gemini",
        metadata_variant: str = "default",
        n_perspectives: int = 5,
        temperature: float = 0.3,
    ):
        self.model_name = model_name
        self.provider = provider
        self.metadata_variant = metadata_variant
        self.n_perspectives = min(n_perspectives, 5)
        self.temperature = temperature
        self.llm = self._init_llm()
        self._cache: dict = {}

        # Load metadata
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        real_metadata_file = {
            "default": "real_datasets_metadata.json",
            "neutral": "real_datasets_metadata_neutral.json",
        }.get(metadata_variant)
        if real_metadata_file is None:
            raise ValueError(
                f"metadata_variant desconhecido: {metadata_variant!r}. "
                f"Use 'default' ou 'neutral'."
            )
        self._real_metadata = self._load_json(
            os.path.join(data_dir, real_metadata_file)
        )
        self._synthetic_metadata = self._load_json(
            os.path.join(data_dir, "synthetic_variants_metadata.json")
        )
        print(f"  [SC] Metadata variant: {metadata_variant} ({real_metadata_file})")
        print(f"  [SC] Perspectives: {self.n_perspectives}, temperature: {self.temperature}")

        # Load original stats for real data
        stats_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "..", "Dataset", "real_data", "processado", "stats_originais.json",
        )
        self._original_stats = self._load_json(os.path.normpath(stats_path))

    def _load_json(self, path: str) -> dict:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        print(f"  Warning: metadata not found at {path}")
        return {}

    def _init_llm(self):
        if self.provider == "openai":
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(
                model_name=self.model_name,
                api_key=os.getenv("OPENAI_API_KEY"),
                temperature=self.temperature,
            )
        elif self.provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            return ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=os.getenv("GEMINI_API_KEY"),
                temperature=self.temperature,
            )
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    # --------------------------------------------------
    # PUBLIC API
    # --------------------------------------------------

    def extract_features(
        self,
        df: pd.DataFrame,
        filename: str = "",
        data_type: str = "sintetico",
        use_cache: bool = True,
    ) -> dict:
        """Extract 8 self-consistency features via multi-perspective voting."""
        stats = self._compute_stats(df)

        cache_key = hashlib.md5(
            json.dumps(
                {"stats": stats, "filename": filename, "data_type": data_type,
                 "_v": "sc1", "_n": self.n_perspectives},
                sort_keys=True,
            ).encode()
        ).hexdigest()

        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        if data_type == "real":
            result = self._extract_real(df, stats, filename)
        else:
            result = self._extract_synthetic(df, stats, filename)

        if use_cache:
            self._cache[cache_key] = result

        return result

    # --------------------------------------------------
    # REAL DATA EXTRACTION
    # --------------------------------------------------

    def _extract_real(self, df: pd.DataFrame, stats: dict, filename: str) -> dict:
        dataset_key = filename.replace(".txt", "")
        base_key = re.sub(r"_boot\d+$", "", dataset_key)
        metadata = self._real_metadata.get(base_key, {})
        orig_stats = self._original_stats.get(base_key, {})

        if not metadata:
            return get_sc_fallback_features()

        prompts = self._build_real_prompts(metadata, stats, orig_stats)
        responses = self._call_all_perspectives(prompts)
        return self._aggregate(responses)

    # --------------------------------------------------
    # SYNTHETIC DATA EXTRACTION
    # --------------------------------------------------

    def _extract_synthetic(self, df: pd.DataFrame, stats: dict, filename: str) -> dict:
        variant_info = self._parse_synthetic_filename(filename)
        dist_info = self._get_distribution_info(variant_info.get("variant_key", ""))

        prompts = self._build_synthetic_prompts(stats, variant_info, dist_info)
        responses = self._call_all_perspectives(prompts)
        return self._aggregate(responses)

    # --------------------------------------------------
    # PARALLEL LLM CALLING
    # --------------------------------------------------

    def _call_all_perspectives(self, prompts: dict[str, str]) -> list[PerspectiveResponse]:
        """Call all perspectives in parallel and return responses."""
        perspectives_to_call = self.PERSPECTIVE_NAMES[:self.n_perspectives]
        responses = [None] * len(perspectives_to_call)

        with ThreadPoolExecutor(max_workers=self.n_perspectives) as executor:
            future_to_idx = {}
            for i, name in enumerate(perspectives_to_call):
                if name in prompts:
                    future = executor.submit(self._call_perspective, prompts[name])
                    future_to_idx[future] = i

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    responses[idx] = future.result()
                except Exception:
                    responses[idx] = PerspectiveResponse()

        return [r if r is not None else PerspectiveResponse() for r in responses]

    def _call_perspective(self, prompt: str) -> PerspectiveResponse:
        """Call LLM for a single perspective and parse response."""
        raw = self._call_llm(prompt)
        if raw is None:
            return PerspectiveResponse()

        try:
            parsed = PerspectiveResponse.model_validate(raw)
            if parsed.domain_mechanism_prior not in ("MCAR", "MAR", "MNAR"):
                parsed.domain_mechanism_prior = "MCAR"
            return parsed
        except Exception:
            return PerspectiveResponse()

    def _call_llm(self, prompt: str, max_retries: int = 3) -> dict | None:
        for attempt in range(max_retries):
            try:
                response = self.llm.invoke(prompt)

                raw = response.content
                if isinstance(raw, list):
                    parts = []
                    for part in raw:
                        if isinstance(part, dict) and "text" in part:
                            parts.append(part["text"])
                        else:
                            parts.append(str(part))
                    raw = "".join(parts)
                elif isinstance(raw, dict) and "text" in raw:
                    raw = raw["text"]
                else:
                    raw = str(raw)

                json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    json_match = re.search(r"\{[\s\S]*\}", raw)
                    json_str = json_match.group(0) if json_match else raw

                return json.loads(json_str)

            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"  Warning: SC perspective failed after {max_retries} attempts: {e}")
                    return None

        return None

    # --------------------------------------------------
    # AGGREGATION (CISC)
    # --------------------------------------------------

    def _aggregate(self, responses: list[PerspectiveResponse]) -> dict:
        """Confidence-Informed Self-Consistency aggregation."""
        votes = {"MCAR": 0.0, "MAR": 0.0, "MNAR": 0.0}
        total_confidence = 0.0

        for resp in responses:
            mechanism = resp.domain_mechanism_prior
            confidence = resp.domain_confidence
            if mechanism in votes:
                votes[mechanism] += confidence
                total_confidence += confidence

        # Normalize
        if total_confidence > 0:
            for k in votes:
                votes[k] /= total_confidence
        else:
            votes = {"MCAR": 1 / 3, "MAR": 1 / 3, "MNAR": 1 / 3}

        winner = max(votes, key=votes.get)
        aggregated_confidence = votes[winner]
        n_agree = sum(
            1 for r in responses if r.domain_mechanism_prior == winner
        )
        agreement_ratio = n_agree / len(responses) if responses else 0.0

        # Mean stats_consistency and surprise across perspectives
        mean_stats_consistency = float(np.mean(
            [r.stats_consistent_with_domain for r in responses]
        ))
        mean_surprise = float(np.mean(
            [r.surprise_factor for r in responses]
        ))

        return {
            "llm_sc_domain_prior": MECHANISM_TO_SCORE.get(winner, 0.5),
            "llm_sc_confidence": round(aggregated_confidence, 4),
            "llm_sc_agreement": round(agreement_ratio, 4),
            "llm_sc_vote_mcar": round(votes["MCAR"], 4),
            "llm_sc_vote_mar": round(votes["MAR"], 4),
            "llm_sc_vote_mnar": round(votes["MNAR"], 4),
            "llm_sc_stats_consistency": round(mean_stats_consistency, 4),
            "llm_sc_surprise": round(mean_surprise, 4),
        }

    # --------------------------------------------------
    # PROMPT BUILDERS — REAL DATA
    # --------------------------------------------------

    def _build_real_header(self, metadata: dict, stats: dict, orig_stats: dict) -> str:
        """Shared dataset context for real data prompts."""
        domain = metadata.get("domain", "unknown")
        source = metadata.get("source", "unknown")
        x0_var = metadata.get("x0_variable", "X0")
        x0_units = metadata.get("x0_units", "")
        x0_desc = metadata.get("x0_description", "")
        x0_range = metadata.get("x0_typical_range", "")
        missing_ctx = metadata.get("missing_context", "")

        preds = metadata.get("predictors", {})
        pred_lines = []
        for xi, info in preds.items():
            name = info.get("name", xi)
            role = info.get("role", "")
            units = info.get("units", "")
            pred_lines.append(f"  - {xi} = {name} ({units}) — {role}")
        pred_text = "\n".join(pred_lines) if pred_lines else "  X1-X4 (numeric predictors)"

        if orig_stats.get("source") == "raw":
            mean_str = f"{orig_stats.get('X0_mean', '?')} {x0_units}"
            std_str = f"{orig_stats.get('X0_std', '?')} {x0_units}"
        else:
            mean_str = f"{stats.get('X0_obs_mean', '?')} (normalized [0,1])"
            std_str = f"{stats.get('X0_obs_std', '?')} (normalized [0,1])"

        r2 = stats.get("x0_imputation_r2", 0.0)
        r2_note = " (UNRELIABLE — R²<0.1, X0 nearly independent of predictors)" if r2 < 0.1 else f" (R²={r2:.2f})"

        return f"""Source: {source}
Variable with missing data: {x0_var} ({x0_units})
  Description: {x0_desc}
  Typical range: {x0_range}
Domain: {domain}
Predictors:
{pred_text}

Missing context: {missing_ctx}

Observed statistics:
- N={stats['n_total']}, missing={stats['n_missing']} ({stats['missing_rate']:.1%})
- {x0_var} observed: mean={mean_str}, std={std_str}
- Skewness={stats.get('X0_obs_skew', 0):.4f}, Kurtosis={stats.get('X0_obs_kurtosis', 0):.4f}
- Missing rate by estimated X0 quartile{r2_note}: Q1={stats.get('q1_rate', 0):.1%}, Q2={stats.get('q2_rate', 0):.1%}, Q3={stats.get('q3_rate', 0):.1%}, Q4={stats.get('q4_rate', 0):.1%}
- Missing rate by X1 quartile (always reliable): Q1={stats.get('x1_q1_miss_rate', 0):.1%}, Q2={stats.get('x1_q2_miss_rate', 0):.1%}, Q3={stats.get('x1_q3_miss_rate', 0):.1%}, Q4={stats.get('x1_q4_miss_rate', 0):.1%}
- Correlation mask-X1: {stats.get('corr_mask_X1', 0):.4f}
- Correlation mask-X2: {stats.get('corr_mask_X2', 0):.4f}"""

    def _build_real_prompts(
        self, metadata: dict, stats: dict, orig_stats: dict
    ) -> dict[str, str]:
        """Build all 5 perspective prompts for real data."""
        header = self._build_real_header(metadata, stats, orig_stats)
        x0_var = metadata.get("x0_variable", "X0")
        domain = metadata.get("domain", "unknown")

        json_schema = self._json_schema_block()

        prompts = {}

        # --- Perspective 1: Statistical ---
        prompts["statistical"] = f"""You are a statistician. Analyze ONLY the numerical patterns.
Completely ignore the domain and the variable name.
Base your classification EXCLUSIVELY on the observed statistics.

## DATASET
{header}

## KEY STATISTICAL CLUES
- X1 quartile miss rates are the MOST RELIABLE indicator:
  - Uniform across X1 quartiles AND |corr_mask_X1| < 0.05 → likely MCAR
  - Varying across X1 quartiles OR |corr_mask_X1| > 0.1 → likely MAR
- X0 quartile miss rates (only trust if R² > 0.1):
  - If X0 Q-rates vary strongly (one quartile has 2x+ rate) AND R² is decent → likely MNAR
  - If R² < 0.1, IGNORE X0 Q-rates — they are unreliable artifacts
- Skewness: high |skewness| with weak correlations may suggest MNAR truncation

Focus on the NUMBERS. What do the statistics tell you?

{json_schema}"""

        # --- Perspective 2: Domain Expert ---
        prompts["domain"] = f"""You are an expert in {domain}.
Focus on HOW {x0_var} is measured/collected in practice.

## DATASET
{header}

## REFERENCE EXAMPLES

### MCAR example
Domain: Manufacturing / printing equipment. Variable: Blade pressure (psi).
Cause: Random sensor malfunctions, data entry errors. The cause is INDEPENDENT of
the pressure value and of any other measured variable.

### MAR example
Domain: Oceanography. Variable: Air temperature (C).
Cause: Sensor fails more under high humidity (X1). The cause depends on an OBSERVED
predictor, NOT on the temperature value itself.

### MNAR example
Domain: Labor economics. Variable: Hourly wages (USD).
Cause: Non-participation when expected wage is too low. The cause depends on the
VALUE of X0 ITSELF.

## QUESTIONS TO ANSWER
1. What equipment or process generates {x0_var} data?
2. Under what circumstances would {x0_var} NOT be recorded?
3. Does the decision to collect depend on the expected value of {x0_var}?

{json_schema}"""

        # --- Perspective 3: Data Engineering Process ---
        prompts["process"] = f"""You are a data engineer.
Focus on the RECORDING and STORAGE process for {x0_var}.

## DATASET
{header}

## ANALYSIS FRAMEWORK
Consider three scenarios:
- **Technical failure (→ MCAR)**: Can {x0_var} be lost due to sensor failure, transmission
  error, database bug, or random data entry mistake? If so, it's MCAR — the loss is
  independent of the value.
- **Conditional recording (→ MAR)**: Is the recording of {x0_var} triggered by another
  observed variable (X1-X4)? For example, a test ordered based on another measurement.
  If so, it's MAR.
- **Value-dependent filtering (→ MNAR)**: Does the system filter, censor, or skip values
  of {x0_var} based on {x0_var}'s own value? For example, values below a detection limit,
  or patients who drop out because their condition (measured by X0) worsened.
  If so, it's MNAR.

Which scenario best matches this dataset?

{json_schema}"""

        # --- Perspective 4: Adversarial (Anti-MAR) ---
        prompts["adversarial"] = f"""You are a critical reviewer. A previous analyst classified this
dataset as MAR. Your task is to ARGUE AGAINST this classification.

## DATASET
{header}

## YOUR TASK: CHALLENGE THE MAR HYPOTHESIS

Consider carefully:
1. If it were MAR, WHICH specific variable (X1, X2, X3, or X4) would cause the missingness?
   Can you name it? If not, it's probably NOT MAR.
2. Is |corr_mask_X1| = {stats.get('corr_mask_X1', 0):.4f} really significant, or could it
   be random noise? Values below 0.05 are likely noise.
3. Missing rate by X1 quartile: Q1={stats.get('x1_q1_miss_rate', 0):.1%}, Q2={stats.get('x1_q2_miss_rate', 0):.1%},
   Q3={stats.get('x1_q3_miss_rate', 0):.1%}, Q4={stats.get('x1_q4_miss_rate', 0):.1%}.
   If these are UNIFORM, there is NO evidence for MAR.
4. If X1 Q-rates are uniform AND correlations are weak, it is likely MCAR.

WARNING: Do NOT default to MAR. You must find STRONG evidence for MAR.
If evidence is weak, classify as MCAR or MNAR based on the X1 quartile pattern and correlations.

{json_schema}"""

        # --- Perspective 5: Censoring/Truncation Expert ---
        prompts["censoring"] = f"""You are an expert in censored and truncated data.
Analyze specifically whether {x0_var} shows signs of MNAR due to censoring or selection.

## DATASET
{header}

## MNAR SUBTYPES TO CONSIDER

1. **Censoring MNAR**: Values below/above a detection limit are not recorded.
   Evidence: missing concentrated in one tail, distribution appears "cut off".
2. **Selection MNAR**: Decision to collect data depends on expected value.
   Evidence: missing correlated with X0's own latent value.
3. **Diffuse MNAR**: Missingness depends on X0 AND other variables jointly.
   Evidence: interaction between X0 and Xi in the missing pattern.

## KEY EVIDENCE
- Skewness of observed X0: {stats.get('X0_obs_skew', 0):.4f}
  (High |skewness| → possible truncation/censoring → MNAR)
- X0 Q-rates (R²={stats.get('x0_imputation_r2', 0):.2f}): Q1={stats.get('q1_rate', 0):.1%}, Q2={stats.get('q2_rate', 0):.1%}, Q3={stats.get('q3_rate', 0):.1%}, Q4={stats.get('q4_rate', 0):.1%}
  (Only trust if R² > 0.1. Large Q1 vs Q4 difference → one-tail censoring)
- X1 Q-rates (reliable): Q1={stats.get('x1_q1_miss_rate', 0):.1%}, Q2={stats.get('x1_q2_miss_rate', 0):.1%}, Q3={stats.get('x1_q3_miss_rate', 0):.1%}, Q4={stats.get('x1_q4_miss_rate', 0):.1%}
- Does the {domain} domain typically involve detection limits for {x0_var}?

If you find NO evidence of censoring or truncation, classify as MCAR or MAR
based on the correlation evidence.

{json_schema}"""

        return prompts

    # --------------------------------------------------
    # PROMPT BUILDERS — SYNTHETIC DATA
    # --------------------------------------------------

    def _build_synthetic_header(
        self, stats: dict, variant_info: dict, dist_info: dict
    ) -> str:
        dist_name = dist_info.get("description", "unknown")
        mr = variant_info.get("missing_rate", "?")

        r2 = stats.get("x0_imputation_r2", 0.0)
        r2_note = " (UNRELIABLE — R²<0.1, X0 nearly independent of predictors)" if r2 < 0.1 else f" (R²={r2:.2f})"

        return f"""Synthetic dataset:
- 1000 observations, 5 variables (X0-X4)
- All variables from {dist_name} distribution
- X0 has {mr}% missing values, X1-X4 are complete

Statistics:
- N={stats['n_total']}, missing={stats['n_missing']} ({stats['missing_rate']:.1%})
- X0 observed: mean={stats.get('X0_obs_mean', 0):.4f}, std={stats.get('X0_obs_std', 0):.4f}
- Skewness={stats.get('X0_obs_skew', 0):.4f}, Kurtosis={stats.get('X0_obs_kurtosis', 0):.4f}
- Missing rate by estimated X0 quartile{r2_note}: Q1={stats.get('q1_rate', 0):.1%}, Q2={stats.get('q2_rate', 0):.1%}, Q3={stats.get('q3_rate', 0):.1%}, Q4={stats.get('q4_rate', 0):.1%}
- Missing rate by X1 quartile (always reliable): Q1={stats.get('x1_q1_miss_rate', 0):.1%}, Q2={stats.get('x1_q2_miss_rate', 0):.1%}, Q3={stats.get('x1_q3_miss_rate', 0):.1%}, Q4={stats.get('x1_q4_miss_rate', 0):.1%}
- Correlation mask-X1: {stats.get('corr_mask_X1', 0):.4f}
- Correlation mask-X2: {stats.get('corr_mask_X2', 0):.4f}
- X1 mean diff (missing vs observed): {stats.get('X1_mean_diff', 0):.4f}"""

    def _build_synthetic_prompts(
        self, stats: dict, variant_info: dict, dist_info: dict
    ) -> dict[str, str]:
        """Build all 5 perspective prompts for synthetic data."""
        header = self._build_synthetic_header(stats, variant_info, dist_info)
        json_schema = self._json_schema_block()

        prompts = {}

        # --- Perspective 1: Statistical ---
        prompts["statistical"] = f"""You are a statistician analyzing a synthetic dataset.
Base your classification EXCLUSIVELY on the observed statistics.

## DATASET
{header}

## KEY STATISTICAL CLUES (use X1 quartile miss rates as PRIMARY evidence)
- X1 Q-rates uniform AND |corr_mask_X1| < 0.05 → MCAR
- X1 Q-rates varying OR |corr_mask_X1| > 0.1 OR significant X1 mean diff → MAR
- X0 Q-rates very unequal AND R² > 0.1 → MNAR (ignore X0 Q-rates if R² < 0.1)

{json_schema}"""

        # --- Perspective 2: Domain (structural for synthetic) ---
        prompts["domain"] = f"""You are an expert in missing data mechanisms.
Analyze this synthetic dataset to determine the mechanism.

## DATASET
{header}

## MECHANISM DEFINITIONS
- **MCAR**: P(missing) is constant — a random coin flip per row.
  Signature: uniform Q1-Q4 rates, no correlations with any Xi.
- **MAR**: P(missing|X1) varies — missingness depends on observed X1.
  Signature: significant |corr_mask_X1|, X1 mean diff between missing/observed groups.
- **MNAR**: P(missing|X0) varies — missingness depends on X0 itself.
  Signature: unequal Q1-Q4 rates, possibly skewed observed distribution.

Which mechanism best explains the observed patterns?

{json_schema}"""

        # --- Perspective 3: Process ---
        prompts["process"] = f"""You are a data engineer analyzing a synthetic dataset.
Think about what DATA GENERATING PROCESS could produce these patterns.

## DATASET
{header}

## ANALYSIS (prioritize X1 quartile miss rates over X0 Q-rates)
- If X1 Q-rates are roughly equal AND |corr_mask_X1| is weak: no MAR evidence.
  Then if X0 Q-rates are also equal (or R² is low so unreliable): MCAR.
- If X1 Q-rates vary OR |corr_mask_X1| is strong OR X1 mean diff is large: MAR evidence.
- If X0 Q-rates are very unequal AND R² > 0.1: MNAR evidence.
- If R² < 0.1: X0 Q-rates are UNRELIABLE — do not use them for MNAR conclusions.

{json_schema}"""

        # --- Perspective 4: Adversarial ---
        prompts["adversarial"] = f"""You are a critical reviewer. Challenge the obvious interpretation.

## DATASET
{header}

## YOUR TASK
Look at the data and argue AGAINST the most obvious classification:
- If correlations are weak and X1 Q-rates look uniform, someone might say MCAR.
  Check: is the skewness unusual? Could there be subtle MNAR truncation?
- If |corr_mask_X1| looks moderate, someone might say MAR.
  But could X1 be correlated with X0, making this actually MNAR?
- If X0 Q-rates look unequal, someone might say MNAR.
  But check R² first — if R² < 0.1, those Q-rates are artifacts, not evidence.

After considering all alternatives, give your BEST classification.
Do NOT default to MAR without strong X1 Q-rate variation or |corr_mask_X1| > 0.1.

{json_schema}"""

        # --- Perspective 5: Censoring ---
        prompts["censoring"] = f"""You are a censoring/truncation specialist.

## DATASET
{header}

## CENSORING ANALYSIS
Look for signs of value-dependent missingness:
- X0 Q-rates (R²={stats.get('x0_imputation_r2', 0):.2f}): Q1={stats.get('q1_rate', 0):.1%} vs Q4={stats.get('q4_rate', 0):.1%}
  Only trust these if R² > 0.1. Large Q1/Q4 difference → one-tail censoring.
- High |skewness| ({stats.get('X0_obs_skew', 0):.4f}) → possible truncation even if Q-rates are unreliable
- X1 Q-rates (reliable): Q1={stats.get('x1_q1_miss_rate', 0):.1%}, Q2={stats.get('x1_q2_miss_rate', 0):.1%}, Q3={stats.get('x1_q3_miss_rate', 0):.1%}, Q4={stats.get('x1_q4_miss_rate', 0):.1%}
  If uniform → NOT MAR → could be MCAR or MNAR

{json_schema}"""

        return prompts

    # --------------------------------------------------
    # SHARED PROMPT COMPONENTS
    # --------------------------------------------------

    @staticmethod
    def _json_schema_block() -> str:
        return """Return ONLY a valid JSON:

```json
{
  "domain_mechanism_prior": "MCAR|MAR|MNAR",
  "domain_confidence": 0.5,
  "stats_consistent_with_domain": 0.5,
  "surprise_factor": 0.0,
  "reasoning": "short explanation"
}
```"""

    # --------------------------------------------------
    # STATISTICS COMPUTATION
    # --------------------------------------------------

    def _compute_stats(self, df: pd.DataFrame) -> dict:
        mask = df["X0"].isna().astype(int).values
        n_total = len(mask)
        n_missing = int(mask.sum())
        X0_obs = df["X0"].dropna().values

        stats = {
            "n_total": n_total,
            "n_missing": n_missing,
            "missing_rate": n_missing / n_total if n_total > 0 else 0,
        }

        if len(X0_obs) > 5:
            stats["X0_obs_mean"] = round(float(np.mean(X0_obs)), 4)
            stats["X0_obs_std"] = round(float(np.std(X0_obs)), 4)
            stats["X0_obs_skew"] = round(float(sp_stats.skew(X0_obs)), 4)
            stats["X0_obs_kurtosis"] = round(
                float(sp_stats.kurtosis(X0_obs, fisher=True)), 4
            )

            # Estimate X0 for missing rows via regression on X1-X4.
            # When R² is too low (< 0.1), regression predictions collapse
            # near the mean, creating the same artifact as median imputation.
            # In that case, report the overall missing rate for all quartiles
            # (= null hypothesis: MCAR, uniform missing).
            X0_estimated, r2 = self._estimate_x0(df)
            stats["x0_imputation_r2"] = r2

            if r2 >= 0.1:
                quartiles = np.percentile(X0_estimated, [25, 50, 75])
                bins = [-np.inf, quartiles[0], quartiles[1], quartiles[2], np.inf]
                bin_idx = np.digitize(X0_estimated, bins[1:-1])
                for q in range(4):
                    q_mask_bool = bin_idx == q
                    if q_mask_bool.sum() > 0:
                        stats[f"q{q + 1}_rate"] = float(mask[q_mask_bool].mean())
                    else:
                        stats[f"q{q + 1}_rate"] = 0.0
            else:
                # R² too low: X0 predictions are unreliable.
                # Report uniform rates (= "no evidence of MNAR from Q-rates").
                overall_rate = stats["missing_rate"]
                for q in range(4):
                    stats[f"q{q + 1}_rate"] = overall_rate

            # Missing rate by X1 quartile (X1 is always observed, no
            # imputation needed — reliable MAR/MCAR signal).
            if "X1" in df.columns:
                X1_vals = df["X1"].values
                x1_quartiles = np.percentile(X1_vals, [25, 50, 75])
                x1_bins = [-np.inf, x1_quartiles[0], x1_quartiles[1], x1_quartiles[2], np.inf]
                x1_bin_idx = np.digitize(X1_vals, x1_bins[1:-1])
                for q in range(4):
                    q_mask_bool = x1_bin_idx == q
                    if q_mask_bool.sum() > 0:
                        stats[f"x1_q{q + 1}_miss_rate"] = float(mask[q_mask_bool].mean())
                    else:
                        stats[f"x1_q{q + 1}_miss_rate"] = 0.0

        for col in ["X1", "X2"]:
            if col in df.columns:
                xi = df[col].values
                if np.std(xi) > 0 and np.std(mask) > 0:
                    corr = np.corrcoef(mask, xi)[0, 1]
                    stats[f"corr_mask_{col}"] = (
                        round(float(corr), 4) if not np.isnan(corr) else 0.0
                    )
                else:
                    stats[f"corr_mask_{col}"] = 0.0

        if "X1" in df.columns:
            X1 = df["X1"].values
            X1_miss = X1[mask == 1]
            X1_obs = X1[mask == 0]
            if len(X1_miss) > 0 and len(X1_obs) > 0:
                stats["X1_mean_diff"] = round(
                    float(np.mean(X1_miss) - np.mean(X1_obs)), 4
                )
            else:
                stats["X1_mean_diff"] = 0.0

        return stats

    @staticmethod
    def _estimate_x0(df: pd.DataFrame) -> tuple[np.ndarray, float]:
        """Estimate full X0 using regression on X1-X4 for missing rows.

        Returns (estimated_X0, r_squared).  R² indicates how reliable
        the quartile-rate estimates are — low R² means X0 is nearly
        independent of predictors and Q-rates should be discounted.
        """
        from sklearn.linear_model import LinearRegression

        pred_cols = [c for c in ["X1", "X2", "X3", "X4"] if c in df.columns]
        obs_mask = df["X0"].notna()
        n_obs = obs_mask.sum()
        n_miss = (~obs_mask).sum()

        if n_miss == 0 or not pred_cols or n_obs < 10:
            return df["X0"].fillna(df["X0"].median()).values, 0.0

        X_obs = df.loc[obs_mask, pred_cols].values
        y_obs = df.loc[obs_mask, "X0"].values
        X_miss = df.loc[~obs_mask, pred_cols].values

        try:
            reg = LinearRegression().fit(X_obs, y_obs)
            r2 = max(reg.score(X_obs, y_obs), 0.0)
            y_pred = reg.predict(X_miss)
            result = df["X0"].copy().values.astype(float)
            result[~obs_mask.values] = y_pred
            return result, round(r2, 4)
        except Exception:
            return df["X0"].fillna(df["X0"].median()).values, 0.0

    # --------------------------------------------------
    # FILENAME PARSING
    # --------------------------------------------------

    def _parse_synthetic_filename(self, filename: str) -> dict:
        name = filename.replace(".txt", "")
        parts = name.split("_")

        result = {"filename": filename, "variant_key": ""}

        if len(parts) >= 2:
            mechanism = parts[0]
            variant_parts = []
            for p in parts[1:]:
                if p.startswith("seed") or p.startswith("mr"):
                    break
                variant_parts.append(p)
            variant = "_".join(variant_parts)
            result["variant_key"] = f"{mechanism}_{variant}"

        for p in parts:
            if p.startswith("mr"):
                try:
                    result["missing_rate"] = int(p[2:])
                except ValueError:
                    pass

        return result

    def _get_distribution_info(self, variant_key: str) -> dict:
        variant_meta = self._synthetic_metadata.get(variant_key, {})
        result = {
            "description": "one of: Uniform[0,1], Normal(0.5,0.15), Exponential(0.3), or Beta(2,5)",
            "expected_mean": "0.26-0.5",
            "expected_skew": "0.0-1.7",
        }
        if variant_meta:
            result["expected_statistics"] = variant_meta.get("expected_statistics", "")
        return result


def get_sc_fallback_features() -> dict:
    """Return default features when self-consistency is not available."""
    return {
        "llm_sc_domain_prior": float("nan"),
        "llm_sc_confidence": float("nan"),
        "llm_sc_agreement": float("nan"),
        "llm_sc_vote_mcar": float("nan"),
        "llm_sc_vote_mar": float("nan"),
        "llm_sc_vote_mnar": float("nan"),
        "llm_sc_stats_consistency": float("nan"),
        "llm_sc_surprise": float("nan"),
    }
