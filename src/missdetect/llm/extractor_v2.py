"""
Extrator de features LLM v2 - Foco em Análise de Segunda Ordem.

Estratégia: A LLM analisa CONSISTÊNCIA entre evidências estatísticas
para detectar padrões que combinações simples de features não capturam.

Principais melhorias:
1. Fornecer estatísticas de segunda ordem já calculadas
2. Pedir análise de CONSISTÊNCIA e CONFLITO entre evidências
3. Gerar features de raciocínio, não apenas scores diretos
4. Foco específico na confusão MCAR↔MNAR
"""

import hashlib
import json
import os
import re

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from scipy import stats as sp_stats


class LLMAnalysisV2(BaseModel):
    """Schema v2: Features de raciocínio e análise de segunda ordem.

    NOTA: Versão otimizada com apenas 8 features relevantes identificadas
    pela análise de relevância (analyze_feature_relevance.py).
    Removidas: mar_evidence_strength, mnar_evidence_strength (redundantes).
    """

    # Análise de consistência
    evidence_consistency: float = Field(
        default=0.5, ge=0.0, le=1.0, description="1=evidências consistentes, 0=evidências conflitantes"
    )

    # Análise de anomalia
    anomaly_detected: float = Field(
        default=0.0, ge=0.0, le=1.0, description="1=padrão anômalo detectado, 0=padrão normal"
    )
    distribution_shift: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Magnitude do desvio da distribuição esperada"
    )

    # Classificação com confiança calibrada
    mcar_confidence: float = Field(default=0.33, ge=0.0, le=1.0)
    mar_confidence: float = Field(default=0.33, ge=0.0, le=1.0)
    mnar_confidence: float = Field(default=0.34, ge=0.0, le=1.0)

    # Reasoning features
    reasoning_mcar_vs_mnar: float = Field(
        default=0.5, ge=0.0, le=1.0, description="0=claramente MCAR, 1=claramente MNAR, 0.5=incerto"
    )
    pattern_clarity: float = Field(
        default=0.5, ge=0.0, le=1.0, description="1=padrão muito claro, 0=nenhum padrão detectável"
    )

    def to_feature_dict(self) -> dict:
        """Converte para dict com prefixo 'llm_'."""
        return {
            "llm_evidence_consistency": self.evidence_consistency,
            "llm_anomaly": self.anomaly_detected,
            "llm_dist_shift": self.distribution_shift,
            "llm_mcar_conf": self.mcar_confidence,
            "llm_mar_conf": self.mar_confidence,
            "llm_mnar_conf": self.mnar_confidence,
            "llm_mcar_vs_mnar": self.reasoning_mcar_vs_mnar,
            "llm_pattern_clarity": self.pattern_clarity,
        }


class LLMFeatureExtractorV2:
    """
    Extrator v2: Análise de segunda ordem com foco em MCAR↔MNAR.

    Diferenças da v1:
    - Estatísticas de segunda ordem pré-calculadas
    - Prompt focado em análise de consistência
    - Features de raciocínio, não apenas classificação
    """

    def __init__(self, model_name: str, provider: str = "gemini"):
        self.model_name = model_name
        self.provider = provider
        self.llm = self._init_llm()
        self._cache: dict = {}

    def _init_llm(self):
        """Inicializa o cliente LLM."""
        if self.provider == "openai":
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model_name=self.model_name,
                api_key=os.getenv("OPENAI_API_KEY"),
                temperature=0.1,  # Pequena variação para exploração
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

    def extract_features(self, df: pd.DataFrame, use_cache: bool = True) -> dict:
        """Extrai features usando análise de segunda ordem."""
        # Calcula estatísticas de primeira E segunda ordem
        stats_summary = self._compute_advanced_statistics(df)

        # Cache key
        cache_key = hashlib.md5(json.dumps(stats_summary, sort_keys=True).encode()).hexdigest()

        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        # Gera prompt focado em análise de consistência
        prompt = self._build_reasoning_prompt(stats_summary)

        # Chama LLM
        result = self._call_llm_with_retry(prompt, max_retries=3)

        if use_cache:
            self._cache[cache_key] = result

        return result

    def _compute_advanced_statistics(self, df: pd.DataFrame) -> dict:
        """
        Calcula estatísticas de primeira e SEGUNDA ordem.

        Segunda ordem = combinações e consistência entre features.
        """
        mask = df["X0"].isna().astype(int).values
        n_total = len(mask)
        n_missing = int(mask.sum())

        X0_obs = df["X0"].dropna().values
        X1 = df["X1"].values

        stats_dict = {
            "n_total": int(n_total),
            "n_missing": n_missing,
            "missing_rate": round(n_missing / n_total, 4),
        }

        # ============ PRIMEIRA ORDEM ============

        # Estatísticas de X0 observado
        if len(X0_obs) > 1:
            stats_dict["X0_obs_mean"] = round(float(np.mean(X0_obs)), 4)
            stats_dict["X0_obs_std"] = round(float(np.std(X0_obs)), 4)
            stats_dict["X0_obs_skew"] = round(float(sp_stats.skew(X0_obs)), 4)
            stats_dict["X0_obs_median"] = round(float(np.median(X0_obs)), 4)

        # Evidências MAR (correlação X1-mask)
        corr_X1_mask = 0.0
        X1_mean_diff = 0.0
        if np.std(X1) > 0 and np.std(mask) > 0:
            corr = np.corrcoef(mask, X1)[0, 1]
            corr_X1_mask = round(float(corr), 4) if not np.isnan(corr) else 0.0
            stats_dict["corr_X1_mask"] = corr_X1_mask

            X1_miss = X1[mask == 1]
            X1_obs = X1[mask == 0]
            if len(X1_miss) > 0 and len(X1_obs) > 0:
                X1_mean_diff = round(float(np.mean(X1_miss) - np.mean(X1_obs)), 4)
                stats_dict["X1_mean_diff"] = X1_mean_diff

        # Evidências MNAR (desvio relativo ao range dos dados, não a 0.5 fixo)
        X0_mean_dev = 0.0
        if len(X0_obs) > 10:
            X0_range = float(np.max(X0_obs) - np.min(X0_obs)) if len(X0_obs) > 1 else 1.0
            X0_center = float(np.median(X0_obs))
            X0_mean_dev = round((X0_center - float(np.mean(X0_obs))) / max(X0_range, 0.01), 4)
            stats_dict["X0_mean_deviation"] = X0_mean_dev
            stats_dict["X0_median_deviation"] = round((X0_center - float(np.median(X0_obs))) / max(X0_range, 0.01), 4)
            stats_dict["X0_obs_range"] = round(X0_range, 4)

        # ============ SEGUNDA ORDEM ============

        # 1. Força combinada de evidências MAR
        mar_evidence = abs(corr_X1_mask) * 5 + abs(X1_mean_diff) * 5
        stats_dict["mar_combined_evidence"] = round(min(mar_evidence, 1.0), 4)

        # 2. Força combinada de evidências MNAR
        X0_skew = stats_dict.get("X0_obs_skew", 0)
        mnar_evidence = abs(X0_mean_dev) * 5 + abs(X0_skew) * 2
        stats_dict["mnar_combined_evidence"] = round(min(mnar_evidence, 1.0), 4)

        # 3. Consistência de evidências
        # Se MAR alto e MNAR baixo = consistente com MAR
        # Se MAR baixo e MNAR alto = consistente com MNAR
        # Se ambos baixos = consistente com MCAR
        # Se ambos altos = CONFLITO (anomalia)
        evidence_conflict = mar_evidence * mnar_evidence  # Alto se ambos altos
        stats_dict["evidence_conflict_score"] = round(evidence_conflict, 4)

        # 4. Score de anomalia (quanto desvia do esperado)
        anomaly = abs(X0_mean_dev) * 10 + abs(X0_skew)
        stats_dict["anomaly_score"] = round(anomaly, 4)

        # 5. Consistência interna MNAR (skew e mean_dev mesmo sinal?)
        if X0_mean_dev != 0 and X0_skew != 0:
            # MNAR com valores altos missing: mean_dev > 0, skew > 0
            mnar_internal_consistency = 1 if np.sign(X0_mean_dev) == np.sign(X0_skew) else 0
        else:
            mnar_internal_consistency = 0.5
        stats_dict["mnar_internal_consistency"] = mnar_internal_consistency

        # 6. Padrão temporal (burst analysis)
        if n_missing > 0 and n_missing < n_total:
            bursts = self._get_burst_sizes(mask)
            stats_dict["avg_burst_size"] = round(float(np.mean(bursts)), 2) if bursts else 1.0
            stats_dict["n_bursts"] = len(bursts)

            # Runs test para aleatoriedade
            n_runs = self._count_runs(mask)
            expected_runs = (2 * n_missing * (n_total - n_missing) / n_total) + 1
            stats_dict["runs_ratio"] = round(n_runs / expected_runs, 4) if expected_runs > 0 else 1.0

        return stats_dict

    def _get_burst_sizes(self, mask):
        """Retorna tamanhos dos bursts de missing."""
        bursts = []
        current = 0
        for v in mask:
            if v == 1:
                current += 1
            else:
                if current > 0:
                    bursts.append(current)
                current = 0
        if current > 0:
            bursts.append(current)
        return bursts

    def _count_runs(self, mask):
        """Conta número de runs (sequências consecutivas de 0s ou 1s)."""
        if len(mask) == 0:
            return 0
        runs = 1
        for i in range(1, len(mask)):
            if mask[i] != mask[i - 1]:
                runs += 1
        return runs

    def _build_reasoning_prompt(self, stats: dict) -> str:
        """
        Prompt focado em análise de CONSISTÊNCIA e RACIOCÍNIO.

        Pede à LLM para analisar se as evidências são consistentes
        e fazer inferência de segunda ordem.
        """
        prompt = f"""Você é um especialista em mecanismos de dados faltantes.

## TAREFA
Analise as EVIDÊNCIAS ESTATÍSTICAS abaixo e determine:
1. Se as evidências são CONSISTENTES entre si
2. Qual mecanismo (MCAR, MAR, MNAR) é mais provável
3. O grau de CONFIANÇA na classificação

## TEORIA RESUMIDA

- **MCAR**: Missing aleatório. Evidências: corr_X1_mask ≈ 0, X0_mean_deviation ≈ 0
- **MAR**: Missing depende de X1. Evidências: |corr_X1_mask| > 0.05, |X1_mean_diff| > 0.05
- **MNAR**: Missing depende de X0. Evidências: X0_mean_deviation > 0.005, X0_obs_skew > 0.02

## ESTATÍSTICAS CALCULADAS

```json
{json.dumps(stats, indent=2)}
```

## ANÁLISE DE CONSISTÊNCIA

Analise:
1. **mar_combined_evidence**: {stats.get('mar_combined_evidence', 0):.3f} - força das evidências MAR
2. **mnar_combined_evidence**: {stats.get('mnar_combined_evidence', 0):.3f} - força das evidências MNAR
3. **evidence_conflict_score**: {stats.get('evidence_conflict_score', 0):.3f} - se alto, evidências conflitantes
4. **mnar_internal_consistency**: {stats.get('mnar_internal_consistency', 0.5)} - se 1, skew e mean_dev concordam

## INSTRUÇÕES DE RACIOCÍNIO

1. Compare a MAGNITUDE RELATIVA de mar_combined_evidence vs mnar_combined_evidence
2. Se ambas são baixas relativas uma à outra → provavelmente MCAR
3. Se mar_combined_evidence >> mnar_combined_evidence → provavelmente MAR
4. Se mnar_combined_evidence >> mar_combined_evidence E mnar_internal_consistency = 1 → provavelmente MNAR
5. Se ambas são altas e próximas → caso ambíguo, reduza confiança e pattern_clarity
6. IMPORTANTE: A distribuição de X0 pode NÃO ser uniforme [0,1]. Avalie os desvios
   relativamente ao X0_obs_std e X0_obs_range, não em termos absolutos.

## RESPOSTA

Retorne SOMENTE um JSON válido:

```json
{{
  "evidence_consistency": 0.5,
  "anomaly_detected": 0.0,
  "distribution_shift": 0.0,
  "mcar_confidence": 0.33,
  "mar_confidence": 0.33,
  "mnar_confidence": 0.34,
  "reasoning_mcar_vs_mnar": 0.5,
  "pattern_clarity": 0.5
}}
```

Onde:
- evidence_consistency: 1=todas evidências apontam mesma direção, 0=conflitantes
- anomaly_detected: 1=padrão incomum, 0=padrão normal
- distribution_shift: magnitude do desvio da distribuição esperada
- mcar/mar/mnar_confidence: devem somar ~1.0
- reasoning_mcar_vs_mnar: 0=claramente MCAR, 1=claramente MNAR (chave para distinguir!)
- pattern_clarity: 1=classificação muito confiante, 0=incerto
"""
        return prompt

    def _call_llm_with_retry(self, prompt: str, max_retries: int = 3) -> dict:
        """Chama LLM com retry e parse robusto."""

        for attempt in range(max_retries):
            try:
                response = self.llm.invoke(prompt)

                # Extrai texto
                raw = response.content
                if isinstance(raw, list):
                    texts = []
                    for part in raw:
                        if isinstance(part, dict) and "text" in part:
                            texts.append(part["text"])
                        else:
                            texts.append(str(part))
                    raw = "".join(texts)
                elif isinstance(raw, dict) and "text" in raw:
                    raw = raw["text"]
                else:
                    raw = str(raw)

                # Extrai JSON
                json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    json_match = re.search(r"\{[\s\S]*\}", raw)
                    json_str = json_match.group(0) if json_match else raw

                # Valida com Pydantic
                parsed = LLMAnalysisV2.model_validate_json(json_str)
                return parsed.to_feature_dict()

            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"⚠️ LLM v2 falhou após {max_retries} tentativas: {e}")
                    # Retorna NaN em vez de defaults para evitar viés sistemático
                    return {k: float("nan") for k in LLMAnalysisV2().to_feature_dict()}

        # Fallback final: NaN em vez de defaults
        return {k: float("nan") for k in LLMAnalysisV2().to_feature_dict()}


def get_llm_fallback_features_v2() -> dict:
    """Retorna features padrão v2 quando LLM não está disponível."""
    return LLMAnalysisV2().to_feature_dict()
