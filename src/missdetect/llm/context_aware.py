"""
LLM Context-Aware Extractor — Domain-knowledge-based features.

Three-step approach:
  Step 1: Causal DAG elicitation (identify causes of missingness)
  Step 2: DAG-informed classification with cross-validation against statistics
  Step 3: Counter-argumentation to calibrate confidence

Extracted features (9):
  - llm_ctx_domain_prior: Mechanism prior (MCAR=0, MAR=0.5, MNAR=1)
  - llm_ctx_domain_confidence: Domain-based confidence
  - llm_ctx_stats_consistency: Are statistics consistent with domain expectation?
  - llm_ctx_surprise: Surprise factor in data
  - llm_ctx_confidence_delta: Confidence change after counter-argument
  - llm_ctx_counter_strength: Strength of counter-argument
  - llm_ctx_cause_type: Type of most plausible cause (A=0, B=0.5, C=1.0)
  - llm_ctx_n_causes: Causal complexity (n_causes/5, normalized)
  - llm_ctx_stats_agreement: Stats agree with causal DAG (1=agree, 0.5=inconclusive, 0=contradict)
"""

import contextlib
import hashlib
import json
import os
import re

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy import stats as sp_stats

# ======================================================
# SCHEMAS
# ======================================================


class ContextAnalysis(BaseModel):
    """Step 1 output: analysis with context."""

    domain_mechanism_prior: str = Field(
        default="MCAR",
        description="Most likely mechanism given the domain: MCAR, MAR, or MNAR",
    )
    domain_confidence: float = Field(
        default=0.33,
        ge=0.0,
        le=1.0,
        description="Confidence in the domain-based classification",
    )
    stats_consistent_with_domain: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="1=statistics consistent with domain expectation, 0=inconsistent",
    )
    surprise_factor: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="1=very surprising/unexpected data, 0=within expectations",
    )
    stats_dag_agreement: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="1=statistics agree with causal DAG, 0.5=inconclusive, 0=contradict",
    )
    reasoning: str = Field(
        default="",
        description="Short reasoning explanation",
    )


class CounterAnalysis(BaseModel):
    """Step 2 output: counter-argumentation."""

    revised_mechanism: str = Field(
        default="MCAR",
        description="Revised mechanism after counter-argument",
    )
    revised_confidence: float = Field(
        default=0.33,
        ge=0.0,
        le=1.0,
        description="Revised confidence",
    )
    counter_argument_strength: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="1=very strong counter-argument, 0=weak",
    )
    mechanism_changed: bool = Field(
        default=False,
        description="True if mechanism changed after counter-argument",
    )


class CausalCause(BaseModel):
    """A single cause of missingness identified by the LLM."""

    description: str = Field(default="", description="Short description of the cause")
    cause_type: str = Field(
        default="A",
        description="A=independent of all variables (MCAR), B=depends on observed Xi (MAR), C=depends on X0 itself (MNAR)",
    )
    plausibility: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="How plausible is this cause (0=implausible, 1=very plausible)",
    )


class CausalDAGAnalysis(BaseModel):
    """Step 1 output: causal DAG elicitation."""

    causes: list[CausalCause] = Field(
        default_factory=lambda: [CausalCause()],
        description="List of possible causes for missingness",
    )
    most_plausible_cause_type: str = Field(
        default="A",
        description="Type of the most plausible cause: A, B, or C",
    )
    reasoning: str = Field(
        default="",
        description="Short explanation of the causal reasoning",
    )


MECHANISM_TO_SCORE = {"MCAR": 0.0, "MAR": 0.5, "MNAR": 1.0}
CAUSE_TYPE_TO_SCORE = {"A": 0.0, "B": 0.5, "C": 1.0}


# ======================================================
# MAIN CLASS
# ======================================================


class LLMContextAwareExtractor:
    """LLM feature extractor with domain context and counter-argumentation."""

    def __init__(
        self,
        model_name: str,
        provider: str = "gemini",
        metadata_variant: str = "default",
    ):
        self.model_name = model_name
        self.provider = provider
        self.metadata_variant = metadata_variant
        self.llm = self._init_llm()
        self._cache: dict = {}

        # Load metadata (variant controla qual arquivo carregar)
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "metadata")
        real_metadata_file = {
            "default": "real_datasets_metadata.json",
            "neutral": "real_datasets_metadata_neutral.json",
            "anonymous": "real_datasets_metadata_anonymous.json",
        }.get(metadata_variant)
        if real_metadata_file is None:
            raise ValueError(f"metadata_variant desconhecido: {metadata_variant!r}. " f"Use 'default', 'neutral' ou 'anonymous'.")
        self._real_metadata = self._load_json(os.path.join(data_dir, real_metadata_file))
        self._synthetic_metadata = self._load_json(os.path.join(data_dir, "synthetic_variants_metadata.json"))
        print(f"  📖 Metadata variant: {metadata_variant} ({real_metadata_file})")

        # Load original stats for real data
        stats_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            "..",
            "Dataset",
            "real_data",
            "processado",
            "stats_originais.json",
        )
        self._original_stats = self._load_json(os.path.normpath(stats_path))

    def _load_json(self, path: str) -> dict:
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        print(f"  Warning: metadata not found at {path}")
        return {}

    def _init_llm(self):
        if self.provider == "openai":
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model_name=self.model_name,
                api_key=os.getenv("OPENAI_API_KEY"),
                temperature=0.1,
            )
        elif self.provider == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI

            return ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=os.getenv("GEMINI_API_KEY"),
                temperature=0.1,
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
        """Extract 6 context-aware features.

        Args:
            df: DataFrame with X0 (missing) and X1-X4
            filename: file name (e.g., "MNAR_pima_insulin.txt")
            data_type: "sintetico" or "real"
            use_cache: use response cache
        """
        # Compute dataset statistics
        stats = self._compute_stats(df)

        # Build cache key (v2 = 3-step causal DAG pipeline)
        cache_key = hashlib.md5(
            json.dumps(
                {"stats": stats, "filename": filename, "data_type": data_type, "_v": 2},
                sort_keys=True,
            ).encode()
        ).hexdigest()

        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        # Build and run prompts
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
        # Handle bootstrap chunk filenames: MNAR_pima_insulin_boot001 → MNAR_pima_insulin
        base_key = re.sub(r"_boot\d+$", "", dataset_key)
        metadata = self._real_metadata.get(base_key, {})
        orig_stats = self._original_stats.get(base_key, {})

        if not metadata:
            return self._extract_generic(stats)

        # Step 1: Causal DAG elicitation
        dag = self._call_dag_step(self._build_real_dag_prompt(metadata, stats, orig_stats))

        # Step 2: DAG-informed classification
        step2 = self._call_step1(self._build_real_classification_prompt(metadata, stats, orig_stats, dag))

        # Step 3: Counter-argumentation with DAG context
        step3 = self._call_step2(self._build_counter_prompt(step2, stats, dag))

        return self._combine_features(dag, step2, step3)

    # --------------------------------------------------
    # SYNTHETIC DATA EXTRACTION
    # --------------------------------------------------

    def _extract_synthetic(self, df: pd.DataFrame, stats: dict, filename: str) -> dict:
        # Parse filename: e.g., "MAR_logistic_seed1234_mr5.txt"
        variant_info = self._parse_synthetic_filename(filename)

        # Get distribution context
        dist_info = self._get_distribution_info(variant_info.get("variant_key", ""))

        # Step 1: Causal DAG elicitation (NO mechanism leak)
        dag = self._call_dag_step(self._build_synthetic_dag_prompt(stats, variant_info, dist_info))

        # Step 2: DAG-informed classification
        step2 = self._call_step1(self._build_synthetic_classification_prompt(stats, variant_info, dist_info, dag))

        # Step 3: Counter-argumentation with DAG context
        step3 = self._call_step2(self._build_counter_prompt(step2, stats, dag))

        return self._combine_features(dag, step2, step3)

    # --------------------------------------------------
    # PROMPT BUILDERS — SHARED HEADER
    # --------------------------------------------------

    def _build_real_header(self, metadata: dict, stats: dict, orig_stats: dict) -> str:
        """Build the shared domain context header for real data prompts."""
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
            stats_source = "original values (pre-normalization)"
        else:
            mean_str = f"{stats.get('X0_obs_mean', '?')} (normalized [0,1])"
            std_str = f"{stats.get('X0_obs_std', '?')} (normalized [0,1])"
            stats_source = "normalized values [0,1]"

        r2 = stats.get("x0_imputation_r2", 0.0)
        r2_note = " (UNRELIABLE — R²<0.1)" if r2 < 0.1 else f" (R²={r2:.2f})"

        return f"""You are an expert in statistics and {domain}.

## DATASET
Source: {source}
Variable with missing data: {x0_var} ({x0_units})
  Description: {x0_desc}
  Typical range: {x0_range}
Predictors:
{pred_text}

## MISSING CONTEXT
{missing_ctx}

## OBSERVED STATISTICS ({stats_source})
- N={stats['n_total']}, missing={stats['n_missing']} ({stats['missing_rate']:.1%})
- {x0_var} observed: mean={mean_str}, std={std_str}
- Skewness={stats.get('X0_obs_skew', 0):.4f}, Kurtosis={stats.get('X0_obs_kurtosis', 0):.4f}
- Missing rate by estimated X0 quartile{r2_note}: Q1={stats.get('q1_rate', 0):.1%}, Q2={stats.get('q2_rate', 0):.1%}, Q3={stats.get('q3_rate', 0):.1%}, Q4={stats.get('q4_rate', 0):.1%}
- Correlation mask-X1: {stats.get('corr_mask_X1', 0):.4f}
- Correlation mask-X2: {stats.get('corr_mask_X2', 0):.4f}"""

    # --------------------------------------------------
    # PROMPT BUILDERS — DAG ELICITATION (STEP 1)
    # --------------------------------------------------

    def _build_real_dag_prompt(self, metadata: dict, stats: dict, orig_stats: dict) -> str:
        header = self._build_real_header(metadata, stats, orig_stats)
        x0_var = metadata.get("x0_variable", "X0")
        domain = metadata.get("domain", "unknown")

        return f"""{header}

## REFERENCE EXAMPLES

Below are three canonical examples of each missing mechanism. Use them to calibrate
your causal reasoning — pay attention to the CAUSES, not just the domain.

### Example: MCAR causes (Type A)
Domain: Manufacturing / printing equipment
Variable: Blade pressure setting (psi)
Typical causes: Random sensor malfunctions, data entry errors, transmission failures.
Why Type A: These causes are INDEPENDENT of the pressure value itself and of any
  other measured variable. The cause is purely technical/random.

### Example: MAR causes (Type B)
Domain: Oceanography / environmental monitoring
Variable: Air temperature (°C)
Typical causes: Sensor fails more under high humidity (X1), maintenance scheduled
  based on wind speed readings (X2).
Why Type B: The cause depends on an OBSERVED predictor (humidity, wind speed), NOT
  on the temperature value itself.

### Example: MNAR causes (Type C)
Domain: Labor economics / wage studies
Variable: Hourly wages (USD)
Typical causes: Non-participation in labor force (expected wage too low), self-censoring
  of very high incomes in surveys, test not ordered when doctor expects normal result.
Why Type C: The cause depends on the VALUE of X0 ITSELF — low wages lead to
  non-participation, high incomes lead to non-reporting.

## MNAR SUBTYPES

1. **Censoring MNAR**: Values below/above a detection limit are not recorded.
2. **Selection MNAR**: The decision to collect data depends on the expected value.
3. **Diffuse MNAR**: Missingness depends on X0 AND other variables jointly.

## CALIBRATION

WARNING: Do NOT default to Type B (MAR) causes.
Before listing a Type B cause, you MUST identify which SPECIFIC predictor (X1, X2, X3, or X4)
drives the missingness. If you cannot name the variable, it is NOT Type B.

## TASK

List ALL plausible causes why {x0_var} might have missing values in the {domain} domain.
For each cause, classify it:
- **Type A (MCAR)**: Cause is independent of all variables (random failures, errors)
- **Type B (MAR)**: Cause depends on a SPECIFIC observed predictor Xi — name which one
- **Type C (MNAR)**: Cause depends on the VALUE of {x0_var} itself

Return ONLY a valid JSON:

```json
{{
  "causes": [
    {{"description": "cause description", "cause_type": "A|B|C", "plausibility": 0.0-1.0}}
  ],
  "most_plausible_cause_type": "A|B|C",
  "reasoning": "short explanation of why this cause type dominates"
}}
```"""

    def _build_synthetic_dag_prompt(self, stats: dict, variant_info: dict, dist_info: dict) -> str:
        dist_name = dist_info.get("description", "unknown")
        mr = variant_info.get("missing_rate", "?")

        return f"""You are an expert in missing data mechanisms.

## SYNTHETIC DATASET
- 1000 observations, 5 variables (X0-X4)
- All variables generated from the {dist_name} distribution
- X0 has {mr}% missing values
- X1-X4 are complete

## OBSERVED STATISTICS
- N={stats['n_total']}, missing={stats['n_missing']} ({stats['missing_rate']:.1%})
- X0 observed: mean={stats.get('X0_obs_mean', 0):.4f}, std={stats.get('X0_obs_std', 0):.4f}
- Skewness={stats.get('X0_obs_skew', 0):.4f}, Kurtosis={stats.get('X0_obs_kurtosis', 0):.4f}
- Missing rate by estimated X0 quartile (R²={stats.get('x0_imputation_r2', 0):.2f}): Q1={stats.get('q1_rate', 0):.1%}, Q2={stats.get('q2_rate', 0):.1%}, Q3={stats.get('q3_rate', 0):.1%}, Q4={stats.get('q4_rate', 0):.1%}
- Correlation mask-X1: {stats.get('corr_mask_X1', 0):.4f}
- Correlation mask-X2: {stats.get('corr_mask_X2', 0):.4f}
- X1 mean diff (missing vs observed): {stats.get('X1_mean_diff', 0):.4f}

## TASK

Based ONLY on the statistical patterns above, identify what structural mechanisms
could produce this pattern of missingness. List possible causes:

- **Type A (MCAR)**: Missing is a random coin flip per row — no relationship with any variable
- **Type B (MAR)**: Missing probability depends on the value of X1 or X2 (observed predictors)
- **Type C (MNAR)**: Missing probability depends on the value of X0 itself

Key statistical clues:
- Strong |corr_mask_X1| > 0.05 → evidence for Type B
- Unequal Q1-Q4 missing rates (one quartile has 2x+ the rate) → evidence for Type C
- Neither of the above → evidence for Type A

Return ONLY a valid JSON:

```json
{{
  "causes": [
    {{"description": "cause description", "cause_type": "A|B|C", "plausibility": 0.0-1.0}}
  ],
  "most_plausible_cause_type": "A|B|C",
  "reasoning": "short explanation based on statistical evidence"
}}
```"""

    # --------------------------------------------------
    # PROMPT BUILDERS — CLASSIFICATION (STEP 2)
    # --------------------------------------------------

    def _build_real_classification_prompt(
        self, metadata: dict, stats: dict, orig_stats: dict, dag: CausalDAGAnalysis
    ) -> str:
        header = self._build_real_header(metadata, stats, orig_stats)
        metadata.get("x0_variable", "X0")
        metadata.get("domain", "unknown")
        dag_section = self._format_dag_section(dag)

        return f"""{header}

## CAUSAL ANALYSIS (from prior step)

{dag_section}

## IMPORTANT CALIBRATION INSTRUCTION

WARNING: Do NOT assume MAR is the default mechanism.
In real datasets, the approximate distribution of mechanisms is:
- ~30% MCAR (data missing due to technical failures, no pattern)
- ~40% MAR (missingness depends on another observed variable)
- ~30% MNAR (missingness depends on the unobserved value itself)

Also consider:
- If corr_mask_X1 is weak (< 0.05) AND Q1-Q4 rates are roughly uniform → likely MCAR
- If corr_mask_X1 is strong (> 0.1) → likely MAR
- If Q1-Q4 rates are very unequal (one quartile has 2x+ the rate) → likely MNAR

## TASK

Using the causal analysis above AND the observed statistics, classify the mechanism.

1. Which missing mechanism (MCAR, MAR, MNAR) is most plausible? Why?
2. Are the observed statistics CONSISTENT with the most plausible cause from the DAG?
   - Type A cause (MCAR): missing rate should be uniform across Q1-Q4, no strong correlations
   - Type B cause (MAR): there should be significant |corr_mask_Xi| > 0.05
   - Type C cause (MNAR): Q1-Q4 rates should be very unequal
3. If statistics CONTRADICT the most plausible cause, reconsider your classification.
4. Do the statistics AGREE, are INCONCLUSIVE, or CONTRADICT the causal DAG?

Return ONLY a valid JSON:

```json
{{
  "domain_mechanism_prior": "MCAR|MAR|MNAR",
  "domain_confidence": 0.5,
  "stats_consistent_with_domain": 0.5,
  "surprise_factor": 0.0,
  "stats_dag_agreement": 0.5,
  "reasoning": "short explanation including how stats relate to the causal DAG"
}}
```"""

    def _build_synthetic_classification_prompt(
        self, stats: dict, variant_info: dict, dist_info: dict, dag: CausalDAGAnalysis
    ) -> str:
        dist_name = dist_info.get("description", "unknown")
        expected_mean = dist_info.get("expected_mean", "?")
        expected_skew = dist_info.get("expected_skew", "?")
        mr = variant_info.get("missing_rate", "?")
        dag_section = self._format_dag_section(dag)

        return f"""You are an expert in missing data mechanisms.

## SYNTHETIC DATASET
- 1000 observations, 5 variables (X0-X4)
- All variables generated from the {dist_name} distribution
  (expected mean ~{expected_mean}, expected skewness ~{expected_skew})
- X0 has {mr}% missing values
- X1-X4 are complete

## OBSERVED STATISTICS
- N={stats['n_total']}, missing={stats['n_missing']} ({stats['missing_rate']:.1%})
- X0 observed: mean={stats.get('X0_obs_mean', 0):.4f}, std={stats.get('X0_obs_std', 0):.4f}
- Skewness={stats.get('X0_obs_skew', 0):.4f}, Kurtosis={stats.get('X0_obs_kurtosis', 0):.4f}
- Missing rate by estimated X0 quartile (R²={stats.get('x0_imputation_r2', 0):.2f}): Q1={stats.get('q1_rate', 0):.1%}, Q2={stats.get('q2_rate', 0):.1%}, Q3={stats.get('q3_rate', 0):.1%}, Q4={stats.get('q4_rate', 0):.1%}
- Correlation mask-X1: {stats.get('corr_mask_X1', 0):.4f}
- Correlation mask-X2: {stats.get('corr_mask_X2', 0):.4f}
- X1 mean diff (missing vs observed): {stats.get('X1_mean_diff', 0):.4f}

## CAUSAL ANALYSIS (from prior step)

{dag_section}

## TASK

Using the causal analysis above AND the observed statistics, classify the mechanism.

1. Which mechanism is most plausible? Why?
2. Do the statistics AGREE with the most plausible cause type?
   - Type A (MCAR): uniform Q1-Q4 rates, weak correlations
   - Type B (MAR): strong |corr_mask_X1|, significant X1 mean diff
   - Type C (MNAR): unequal Q1-Q4 rates
3. If statistics CONTRADICT the causal analysis, reconsider.

Return ONLY a valid JSON:

```json
{{
  "domain_mechanism_prior": "MCAR|MAR|MNAR",
  "domain_confidence": 0.5,
  "stats_consistent_with_domain": 0.5,
  "surprise_factor": 0.0,
  "stats_dag_agreement": 0.5,
  "reasoning": "short explanation"
}}
```"""

    def _format_dag_section(self, dag: CausalDAGAnalysis) -> str:
        """Format DAG analysis as text for injection into subsequent prompts."""
        type_labels = {"A": "MCAR", "B": "MAR", "C": "MNAR"}
        lines = [f"Identified {len(dag.causes)} possible cause(s) of missingness:"]
        for i, cause in enumerate(dag.causes, 1):
            label = type_labels.get(cause.cause_type, "?")
            lines.append(
                f"  {i}. [{cause.cause_type}/{label}] {cause.description} " f"(plausibility={cause.plausibility:.2f})"
            )
        most_label = type_labels.get(dag.most_plausible_cause_type, "?")
        lines.append(f"\nMost plausible cause type: {dag.most_plausible_cause_type} ({most_label})")
        if dag.reasoning:
            lines.append(f"Reasoning: {dag.reasoning}")
        return "\n".join(lines)

    def _build_counter_prompt(self, step2: ContextAnalysis, stats: dict, dag: CausalDAGAnalysis) -> str:
        mech = step2.domain_mechanism_prior
        conf = step2.domain_confidence
        reasoning = step2.reasoning

        # Generate alternative mechanisms
        alternatives = [m for m in ["MCAR", "MAR", "MNAR"] if m != mech]
        alt_text = " or ".join(alternatives)

        # Build evidence for/against
        r2 = stats.get("x0_imputation_r2", 0.0)
        q_rates = f"Q1={stats.get('q1_rate', 0):.1%}, Q2={stats.get('q2_rate', 0):.1%}, Q3={stats.get('q3_rate', 0):.1%}, Q4={stats.get('q4_rate', 0):.1%}"
        r2_note = " (UNRELIABLE — R²<0.1)" if r2 < 0.1 else f" (R²={r2:.2f})"
        corr_x1 = stats.get("corr_mask_X1", 0)
        skew = stats.get("X0_obs_skew", 0)
        dag_section = self._format_dag_section(dag)

        return f"""You analyzed a dataset and concluded:
- Most likely mechanism: {mech}
- Confidence: {conf:.2f}
- Reasoning: {reasoning}

## CAUSAL DAG CONTEXT
{dag_section}

## COUNTER-ARGUMENT
Consider the ALTERNATIVE hypothesis: what if it were {alt_text}?
Also consider whether the causal DAG above might have missed important causes.

Evidence to consider:
- Missing rate by estimated X0 quartile{r2_note}: {q_rates}
  (If unequal AND R²>0.1 → suggests MNAR; if uniform → suggests MCAR; if R²<0.1 → unreliable)
- Correlation mask-X1: {corr_x1:.4f}
  (If strong → suggests MAR; if weak → MCAR or MNAR)
- Skewness of observed X0: {skew:.4f}
  (If high → possible MNAR due to truncation)

After considering the counter-argument, update your analysis:

Return ONLY a valid JSON:

```json
{{
  "revised_mechanism": "MCAR|MAR|MNAR",
  "revised_confidence": 0.5,
  "counter_argument_strength": 0.5,
  "mechanism_changed": false
}}
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
            stats["X0_obs_kurtosis"] = round(float(sp_stats.kurtosis(X0_obs, fisher=True)), 4)

            # Missing rate by quartile of X0 — uses regression imputation.
            # When R² is too low (< 0.1), predictions collapse near the mean,
            # so we report uniform rates (null hypothesis: MCAR).
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
                overall_rate = stats["missing_rate"]
                for q in range(4):
                    stats[f"q{q + 1}_rate"] = overall_rate

        # Correlations mask-Xi
        for col in ["X1", "X2"]:
            if col in df.columns:
                xi = df[col].values
                if np.std(xi) > 0 and np.std(mask) > 0:
                    corr = np.corrcoef(mask, xi)[0, 1]
                    stats[f"corr_mask_{col}"] = round(float(corr), 4) if not np.isnan(corr) else 0.0
                else:
                    stats[f"corr_mask_{col}"] = 0.0

        # X1 mean diff
        if "X1" in df.columns:
            X1 = df["X1"].values
            X1_miss = X1[mask == 1]
            X1_obs = X1[mask == 0]
            if len(X1_miss) > 0 and len(X1_obs) > 0:
                stats["X1_mean_diff"] = round(float(np.mean(X1_miss) - np.mean(X1_obs)), 4)
            else:
                stats["X1_mean_diff"] = 0.0

        return stats

    @staticmethod
    def _estimate_x0(df: pd.DataFrame) -> tuple[np.ndarray, float]:
        """Estimate full X0 using regression on X1-X4 for missing rows.

        Returns (estimated_X0, r_squared).  R² indicates how reliable
        the quartile-rate estimates are.
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
        """Parse synthetic filename: MAR_logistic_seed1234_mr5.txt"""
        name = filename.replace(".txt", "")
        parts = name.split("_")

        result = {"filename": filename, "variant_key": ""}

        if len(parts) >= 2:
            mechanism = parts[0]
            # Find variant name (everything between mechanism and seed/mr)
            variant_parts = []
            for p in parts[1:]:
                if p.startswith("seed") or p.startswith("mr"):
                    break
                variant_parts.append(p)
            variant = "_".join(variant_parts)
            result["variant_key"] = f"{mechanism}_{variant}"

        # Extract missing rate
        for p in parts:
            if p.startswith("mr"):
                with contextlib.suppress(ValueError):
                    result["missing_rate"] = int(p[2:])

        return result

    def _get_distribution_info(self, variant_key: str) -> dict:
        """Get expected distribution and variant info for synthetic data.

        Returns both distribution info (generic since k is unknown from filename)
        and variant-specific structural context (mechanism description, expected stats).
        """
        # Variant-specific metadata (e.g., MAR_logistic description)
        variant_meta = self._synthetic_metadata.get(variant_key, {})

        # Distribution info is generic since we can't determine k from filename
        self._synthetic_metadata.get("_distribution_cycle", {})

        result = {
            "description": "one of: Uniform[0,1], Normal(0.5,0.15), Exponential(0.3), or Beta(2,5)",
            "expected_mean": "0.26-0.5",
            "expected_skew": "0.0-1.7",
        }

        # Add variant-specific structural info if available
        if variant_meta:
            result["expected_statistics"] = variant_meta.get("expected_statistics", "")

        return result

    # --------------------------------------------------
    # LLM CALLING
    # --------------------------------------------------

    def _call_step1(self, prompt: str) -> ContextAnalysis:
        raw = self._call_llm(prompt)
        if raw is None:
            return ContextAnalysis()

        try:
            parsed = ContextAnalysis.model_validate(raw)
            # Validate mechanism
            if parsed.domain_mechanism_prior not in ("MCAR", "MAR", "MNAR"):
                parsed.domain_mechanism_prior = "MCAR"
            return parsed
        except Exception:
            return ContextAnalysis()

    def _call_step2(self, prompt: str) -> CounterAnalysis:
        raw = self._call_llm(prompt)
        if raw is None:
            return CounterAnalysis()

        try:
            parsed = CounterAnalysis.model_validate(raw)
            if parsed.revised_mechanism not in ("MCAR", "MAR", "MNAR"):
                parsed.revised_mechanism = "MCAR"
            return parsed
        except Exception:
            return CounterAnalysis()

    def _call_dag_step(self, prompt: str) -> CausalDAGAnalysis:
        raw = self._call_llm(prompt)
        if raw is None:
            return CausalDAGAnalysis()

        try:
            # Normalize cause fields before validation
            if isinstance(raw, dict):
                causes = raw.get("causes") or raw.get("possible_causes") or []
                for cause in causes:
                    if isinstance(cause, dict) and "cause_type" in cause:
                        cause["cause_type"] = self._normalize_cause_type(cause["cause_type"])
                raw["causes"] = causes
                if "most_plausible_cause_type" in raw:
                    raw["most_plausible_cause_type"] = self._normalize_cause_type(raw["most_plausible_cause_type"])

            parsed = CausalDAGAnalysis.model_validate(raw)
            if parsed.most_plausible_cause_type not in ("A", "B", "C"):
                parsed.most_plausible_cause_type = "A"
            if not parsed.causes:
                parsed.causes = [CausalCause()]
            return parsed
        except Exception:
            return CausalDAGAnalysis()

    @staticmethod
    def _normalize_cause_type(ct: str) -> str:
        ct = ct.strip().upper()
        if ct in ("A", "B", "C"):
            return ct
        for letter in ("A", "B", "C"):
            if letter in ct:
                return letter
        return "A"

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

                # Extract JSON
                json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    json_match = re.search(r"\{[\s\S]*\}", raw)
                    json_str = json_match.group(0) if json_match else raw

                return json.loads(json_str)

            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"  Warning: LLM context-aware failed after {max_retries} attempts: {e}")
                    return None

        return None

    # --------------------------------------------------
    # FEATURE COMBINATION
    # --------------------------------------------------

    def _combine_features(self, dag: CausalDAGAnalysis, step2: ContextAnalysis, step3: CounterAnalysis) -> dict:
        mech_score = MECHANISM_TO_SCORE.get(step2.domain_mechanism_prior, 0.5)
        confidence_delta = abs(step3.revised_confidence - step2.domain_confidence)
        cause_type_score = CAUSE_TYPE_TO_SCORE.get(dag.most_plausible_cause_type, 0.5)
        n_causes_normalized = min(len(dag.causes) / 5.0, 1.0)

        return {
            "llm_ctx_domain_prior": mech_score,
            "llm_ctx_domain_confidence": step2.domain_confidence,
            "llm_ctx_stats_consistency": step2.stats_consistent_with_domain,
            "llm_ctx_surprise": step2.surprise_factor,
            "llm_ctx_confidence_delta": round(confidence_delta, 4),
            "llm_ctx_counter_strength": step3.counter_argument_strength,
            "llm_ctx_cause_type": cause_type_score,
            "llm_ctx_n_causes": round(n_causes_normalized, 4),
            "llm_ctx_stats_agreement": step2.stats_dag_agreement,
        }

    def _extract_generic(self, stats: dict) -> dict:
        """Fallback when no metadata is available."""
        return get_context_fallback_features()


def get_context_fallback_features() -> dict:
    """Return default features when LLM is not available."""
    return {
        "llm_ctx_domain_prior": float("nan"),
        "llm_ctx_domain_confidence": float("nan"),
        "llm_ctx_stats_consistency": float("nan"),
        "llm_ctx_surprise": float("nan"),
        "llm_ctx_confidence_delta": float("nan"),
        "llm_ctx_counter_strength": float("nan"),
        "llm_ctx_cause_type": float("nan"),
        "llm_ctx_n_causes": float("nan"),
        "llm_ctx_stats_agreement": float("nan"),
    }
