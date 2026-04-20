"""
LLM Context-Aware Extractor — Domain-knowledge-based features.

Two-step approach:
  Step 1: Analysis with domain context (real) or structural context (synthetic)
  Step 2: Counter-argumentation to calibrate confidence

Extracted features (6):
  - llm_ctx_domain_prior: Mechanism prior (MCAR=0, MAR=0.5, MNAR=1)
  - llm_ctx_domain_confidence: Domain-based confidence
  - llm_ctx_stats_consistency: Are statistics consistent with domain expectation?
  - llm_ctx_surprise: Surprise factor in data
  - llm_ctx_confidence_delta: Confidence change after counter-argument
  - llm_ctx_counter_strength: Strength of counter-argument
"""

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
        default=0.33, ge=0.0, le=1.0,
        description="Confidence in the domain-based classification",
    )
    stats_consistent_with_domain: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="1=statistics consistent with domain expectation, 0=inconsistent",
    )
    surprise_factor: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="1=very surprising/unexpected data, 0=within expectations",
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
        default=0.33, ge=0.0, le=1.0,
        description="Revised confidence",
    )
    counter_argument_strength: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="1=very strong counter-argument, 0=weak",
    )
    mechanism_changed: bool = Field(
        default=False,
        description="True if mechanism changed after counter-argument",
    )


MECHANISM_TO_SCORE = {"MCAR": 0.0, "MAR": 0.5, "MNAR": 1.0}


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
        print(f"  📖 Metadata variant: {metadata_variant} ({real_metadata_file})")

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

        # Build cache key
        cache_key = hashlib.md5(
            json.dumps(
                {"stats": stats, "filename": filename, "data_type": data_type},
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

        # Step 1: Analysis with domain context
        prompt1 = self._build_real_prompt(metadata, stats, orig_stats)
        step1 = self._call_step1(prompt1)

        # Step 2: Counter-argumentation
        prompt2 = self._build_counter_prompt(step1, stats)
        step2 = self._call_step2(prompt2)

        return self._combine_features(step1, step2)

    # --------------------------------------------------
    # SYNTHETIC DATA EXTRACTION
    # --------------------------------------------------

    def _extract_synthetic(self, df: pd.DataFrame, stats: dict, filename: str) -> dict:
        # Parse filename: e.g., "MAR_logistic_seed1234_mr5.txt"
        variant_info = self._parse_synthetic_filename(filename)

        # Get distribution context
        dist_info = self._get_distribution_info(variant_info.get("variant_key", ""))

        # Step 1: Analysis with structural context (NO mechanism leak)
        prompt1 = self._build_synthetic_prompt(stats, variant_info, dist_info)
        step1 = self._call_step1(prompt1)

        # Step 2: Counter-argumentation
        prompt2 = self._build_counter_prompt(step1, stats)
        step2 = self._call_step2(prompt2)

        return self._combine_features(step1, step2)

    # --------------------------------------------------
    # PROMPT BUILDERS
    # --------------------------------------------------

    def _build_real_prompt(self, metadata: dict, stats: dict, orig_stats: dict) -> str:
        domain = metadata.get("domain", "unknown")
        source = metadata.get("source", "unknown")
        x0_var = metadata.get("x0_variable", "X0")
        x0_units = metadata.get("x0_units", "")
        x0_desc = metadata.get("x0_description", "")
        x0_range = metadata.get("x0_typical_range", "")
        missing_ctx = metadata.get("missing_context", "")

        # Build predictor description
        preds = metadata.get("predictors", {})
        pred_lines = []
        for xi, info in preds.items():
            name = info.get("name", xi)
            role = info.get("role", "")
            units = info.get("units", "")
            pred_lines.append(f"  - {xi} = {name} ({units}) — {role}")
        pred_text = "\n".join(pred_lines) if pred_lines else "  X1-X4 (numeric predictors)"

        # Use original stats if available, else normalized
        if orig_stats.get("source") == "raw":
            mean_str = f"{orig_stats.get('X0_mean', '?')} {x0_units}"
            std_str = f"{orig_stats.get('X0_std', '?')} {x0_units}"
            stats_source = "original values (pre-normalization)"
        else:
            mean_str = f"{stats.get('X0_obs_mean', '?')} (normalized [0,1])"
            std_str = f"{stats.get('X0_obs_std', '?')} (normalized [0,1])"
            stats_source = "normalized values [0,1]"

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
- Missing rate by X0 quartile: Q1={stats.get('q1_rate', 0):.1%}, Q2={stats.get('q2_rate', 0):.1%}, Q3={stats.get('q3_rate', 0):.1%}, Q4={stats.get('q4_rate', 0):.1%}
- Correlation mask-X1: {stats.get('corr_mask_X1', 0):.4f}
- Correlation mask-X2: {stats.get('corr_mask_X2', 0):.4f}

## TASK
Considering your knowledge about {domain} and how {x0_var} is measured/collected:

1. Which missing mechanism (MCAR, MAR, MNAR) is most plausible given the DOMAIN? Why?
2. Are the observed statistics CONSISTENT with your domain expectation?
3. Is there anything SURPRISING in the data that contradicts the domain explanation?

Return ONLY a valid JSON:

```json
{{
  "domain_mechanism_prior": "MCAR|MAR|MNAR",
  "domain_confidence": 0.5,
  "stats_consistent_with_domain": 0.5,
  "surprise_factor": 0.0,
  "reasoning": "short explanation"
}}
```"""

    def _build_synthetic_prompt(
        self, stats: dict, variant_info: dict, dist_info: dict
    ) -> str:
        dist_name = dist_info.get("description", "unknown")
        expected_mean = dist_info.get("expected_mean", "?")
        expected_skew = dist_info.get("expected_skew", "?")
        mr = variant_info.get("missing_rate", "?")

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
- Missing rate by X0 quartile: Q1={stats.get('q1_rate', 0):.1%}, Q2={stats.get('q2_rate', 0):.1%}, Q3={stats.get('q3_rate', 0):.1%}, Q4={stats.get('q4_rate', 0):.1%}
- Correlation mask-X1: {stats.get('corr_mask_X1', 0):.4f}
- Correlation mask-X2: {stats.get('corr_mask_X2', 0):.4f}
- X1 mean diff (missing vs observed): {stats.get('X1_mean_diff', 0):.4f}

## CONTEXT
Artificially generated data with no real domain. Analyze ONLY the statistical patterns.
Consider: If the variables are {dist_name}, what is the EXPECTED distribution of observed
values if the mechanism were MCAR? Deviations from that expectation suggest MAR or MNAR.

Key clues:
- Strong mask-X1 correlation → MAR (missing depends on X1)
- Unequal missing rate across X0 quartiles → MNAR (missing depends on X0)
- Neither of the above → MCAR (random)

## TASK
Which mechanism is most plausible and why?

Return ONLY a valid JSON:

```json
{{
  "domain_mechanism_prior": "MCAR|MAR|MNAR",
  "domain_confidence": 0.5,
  "stats_consistent_with_domain": 0.5,
  "surprise_factor": 0.0,
  "reasoning": "short explanation"
}}
```"""

    def _build_counter_prompt(self, step1: ContextAnalysis, stats: dict) -> str:
        mech = step1.domain_mechanism_prior
        conf = step1.domain_confidence
        reasoning = step1.reasoning

        # Generate alternative mechanisms
        alternatives = [m for m in ["MCAR", "MAR", "MNAR"] if m != mech]
        alt_text = " or ".join(alternatives)

        # Build evidence for/against
        q_rates = f"Q1={stats.get('q1_rate', 0):.1%}, Q2={stats.get('q2_rate', 0):.1%}, Q3={stats.get('q3_rate', 0):.1%}, Q4={stats.get('q4_rate', 0):.1%}"
        corr_x1 = stats.get("corr_mask_X1", 0)
        skew = stats.get("X0_obs_skew", 0)

        return f"""You analyzed a dataset and concluded:
- Most likely mechanism: {mech}
- Confidence: {conf:.2f}
- Reasoning: {reasoning}

## COUNTER-ARGUMENT
Consider the ALTERNATIVE hypothesis: what if it were {alt_text}?

Evidence to consider:
- Missing rate by X0 quartile: {q_rates}
  (If unequal → suggests MNAR; if uniform → suggests MCAR)
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
        X0_imputed = df["X0"].fillna(df["X0"].median()).values

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

            # Missing rate by quartile of X0 (using imputed values)
            quartiles = np.percentile(X0_imputed, [25, 50, 75])
            bins = [-np.inf, quartiles[0], quartiles[1], quartiles[2], np.inf]
            bin_idx = np.digitize(X0_imputed, bins[1:-1])
            for q in range(4):
                q_mask_bool = bin_idx == q
                if q_mask_bool.sum() > 0:
                    stats[f"q{q + 1}_rate"] = float(mask[q_mask_bool].mean())
                else:
                    stats[f"q{q + 1}_rate"] = 0.0

        # Correlations mask-Xi
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

        # X1 mean diff
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
                try:
                    result["missing_rate"] = int(p[2:])
                except ValueError:
                    pass

        return result

    def _get_distribution_info(self, variant_key: str) -> dict:
        """Get expected distribution and variant info for synthetic data.

        Returns both distribution info (generic since k is unknown from filename)
        and variant-specific structural context (mechanism description, expected stats).
        """
        # Variant-specific metadata (e.g., MAR_logistic description)
        variant_meta = self._synthetic_metadata.get(variant_key, {})

        # Distribution info is generic since we can't determine k from filename
        dist_cycle = self._synthetic_metadata.get("_distribution_cycle", {})

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

    def _combine_features(
        self, step1: ContextAnalysis, step2: CounterAnalysis
    ) -> dict:
        mech_score = MECHANISM_TO_SCORE.get(step1.domain_mechanism_prior, 0.5)
        confidence_delta = abs(step2.revised_confidence - step1.domain_confidence)

        return {
            "llm_ctx_domain_prior": mech_score,
            "llm_ctx_domain_confidence": step1.domain_confidence,
            "llm_ctx_stats_consistency": step1.stats_consistent_with_domain,
            "llm_ctx_surprise": step1.surprise_factor,
            "llm_ctx_confidence_delta": round(confidence_delta, 4),
            "llm_ctx_counter_strength": step2.counter_argument_strength,
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
    }
