"""
LLM Judge: Desambiguação binária MCAR vs MNAR.

Em vez de pedir classificação 3-way (MCAR/MAR/MNAR), pede ao LLM
para analisar especificamente se há evidência de censura/truncamento
em X0 que indicaria MNAR disfarçado de MCAR.

Motivação (STEP03): MCAR e MNAR são quase indistinguíveis por features
estatísticas porque MNAR depende de X0, mas X0 está faltante justamente
onde precisamos medi-lo — um problema circular.
"""
import os
import re
import json
import hashlib
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy import stats as sp_stats


class MNARJudgment(BaseModel):
    """Schema para julgamento binário MCAR vs MNAR."""

    mnar_probability: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="0=claramente MCAR, 1=claramente MNAR"
    )
    censoring_evidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Força da evidência de censura/truncamento em X0"
    )
    distribution_anomaly: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Grau de anomalia na distribuição de X0 observado"
    )
    pattern_structured: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="0=missing parece aleatório, 1=missing parece estruturado"
    )

    def to_feature_dict(self) -> dict:
        return {
            "llm_judge_mnar_prob": self.mnar_probability,
            "llm_judge_censoring": self.censoring_evidence,
            "llm_judge_dist_anomaly": self.distribution_anomaly,
            "llm_judge_structured": self.pattern_structured,
        }


class LLMJudgeMNAR:
    """
    LLM como juiz binário para MCAR vs MNAR.

    Diferente do extractor_v2 (classificação 3-way com 8 features),
    este faz desambiguação focada em 4 features complementares.
    """

    def __init__(self, model_name: str, provider: str = "gemini"):
        self.model_name = model_name
        self.provider = provider
        self.llm = self._init_llm()
        self._cache: dict = {}

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
            raise ValueError(f"Provider não suportado: {self.provider}")

    def judge(self, df: pd.DataFrame, use_cache: bool = True) -> dict:
        """Julga se o dataset é mais MCAR ou MNAR."""
        profile = self._build_dataset_profile(df)

        cache_key = hashlib.md5(
            json.dumps(profile, sort_keys=True).encode()
        ).hexdigest()

        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        prompt = self._build_judge_prompt(profile)
        result = self._call_llm(prompt)

        if use_cache:
            self._cache[cache_key] = result

        return result

    def _build_dataset_profile(self, df: pd.DataFrame) -> dict:
        """Constrói perfil completo do dataset para o juiz."""
        mask = df["X0"].isna().astype(int).values
        n_total = len(mask)
        n_missing = int(mask.sum())
        X0_obs = df["X0"].dropna().values
        X0_imputed = df["X0"].fillna(df["X0"].median()).values

        profile = {
            "n_total": n_total,
            "n_missing": n_missing,
            "missing_rate": round(n_missing / n_total, 4),
        }

        if len(X0_obs) > 5:
            profile["X0_obs_mean"] = round(float(np.mean(X0_obs)), 4)
            profile["X0_obs_std"] = round(float(np.std(X0_obs)), 4)
            profile["X0_obs_skew"] = round(float(sp_stats.skew(X0_obs)), 4)
            profile["X0_obs_kurtosis"] = round(float(sp_stats.kurtosis(X0_obs, fisher=True)), 4)
            profile["X0_obs_min"] = round(float(np.min(X0_obs)), 4)
            profile["X0_obs_max"] = round(float(np.max(X0_obs)), 4)

            # Percentis de X0 observado
            p5, p25, p50, p75, p95 = np.percentile(X0_obs, [5, 25, 50, 75, 95])
            profile["X0_obs_percentiles"] = {
                "p5": round(float(p5), 4),
                "p25": round(float(p25), 4),
                "p50": round(float(p50), 4),
                "p75": round(float(p75), 4),
                "p95": round(float(p95), 4),
            }

            # Missing rate por faixa de X0 imputado
            quartiles = np.percentile(X0_imputed, [25, 50, 75])
            bins = [-np.inf, quartiles[0], quartiles[1], quartiles[2], np.inf]
            bin_idx = np.digitize(X0_imputed, bins[1:-1])
            rates_by_q = {}
            for q in range(4):
                q_mask = bin_idx == q
                if q_mask.sum() > 0:
                    rates_by_q[f"Q{q+1}"] = round(float(mask[q_mask].mean()), 4)
            profile["missing_rate_by_quartile"] = rates_by_q

            # KS test X0_obs vs distribuição teórica normal
            if len(X0_obs) > 20:
                ks_stat, ks_pval = sp_stats.kstest(
                    (X0_obs - np.mean(X0_obs)) / max(np.std(X0_obs), 1e-10),
                    'norm'
                )
                profile["X0_normality_ks"] = round(float(ks_stat), 4)
                profile["X0_normality_pval"] = round(float(ks_pval), 4)

        # Correlações mask-Xi
        for col in ["X1", "X2", "X3", "X4"]:
            if col in df.columns:
                xi = df[col].values
                if np.std(xi) > 0 and np.std(mask) > 0:
                    corr = np.corrcoef(mask, xi)[0, 1]
                    if not np.isnan(corr):
                        profile[f"corr_mask_{col}"] = round(float(corr), 4)

        return profile

    def _build_judge_prompt(self, profile: dict) -> str:
        return f"""Você é um estatístico especialista em dados faltantes.

## TAREFA ESPECÍFICA
Dado que Little's test NÃO rejeitou MCAR e não há correlação forte mask-Xi,
examine a distribuição de X0_observado. Há evidência de CENSURA, TRUNCAMENTO,
ou padrão NÃO-ALEATÓRIO que indicaria MNAR disfarçado de MCAR?

## CONTEXTO
Em MNAR, os dados faltam PORQUE seu valor é extremo (ex: insulina alta em
diabéticos não é medida). Isso causa:
- Truncamento: uma cauda de X0 é "cortada"
- Missing rate desigual por faixa de X0
- Curtose anormal (distribuição achatada ou pontiaguda)
- Assimetria na distribuição observada

Em MCAR, os dados faltam por acaso. X0_observado deve parecer uma
amostra aleatória da distribuição completa.

## PERFIL DO DATASET

```json
{json.dumps(profile, indent=2)}
```

## ANÁLISE PEDIDA

Examine especificamente:
1. A taxa de missing varia entre quartis de X0? (missing_rate_by_quartile)
2. X0 tem assimetria ou curtose anormal?
3. Os percentis de X0 sugerem truncamento em alguma cauda?
4. O padrão de missing parece estruturado ou aleatório?

## RESPOSTA

Retorne SOMENTE um JSON válido:

```json
{{
  "mnar_probability": 0.5,
  "censoring_evidence": 0.0,
  "distribution_anomaly": 0.0,
  "pattern_structured": 0.0
}}
```

Onde:
- mnar_probability: 0=claramente MCAR, 1=claramente MNAR disfarçado
- censoring_evidence: força da evidência de censura/truncamento
- distribution_anomaly: grau de anomalia na distribuição de X0
- pattern_structured: 0=missing aleatório, 1=missing estruturado
"""

    def _call_llm(self, prompt: str, max_retries: int = 3) -> dict:
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

                parsed = MNARJudgment.model_validate_json(json_str)
                return parsed.to_feature_dict()

            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"⚠️ LLM Judge falhou após {max_retries} tentativas: {e}")
                    return {k: float("nan") for k in MNARJudgment().to_feature_dict()}

        return {k: float("nan") for k in MNARJudgment().to_feature_dict()}


def get_judge_fallback_features() -> dict:
    """Retorna features padrão quando LLM não está disponível."""
    return MNARJudgment().to_feature_dict()
