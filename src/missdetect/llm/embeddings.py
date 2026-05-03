"""
LLM Embeddings: serializa estatísticas como texto → gera embedding → PCA.

Baseado em "Enriching Tabular Data with LLM Embeddings" (Kasneci, 2024).

Em vez de pedir scores ao LLM, usa a representação interna (embedding)
das estatísticas serializadas como features. O embedding captura relações
não-lineares entre as estatísticas que não são explícitas nos scores.

Usa sentence-transformers local (all-MiniLM-L6-v2) por ser:
- Gratuito (sem API)
- Determinístico (cacheável)
- Rápido (CPU)
"""

import hashlib
import json
import os

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


class EmbeddingFeatureExtractor:
    """
    Extrai features via embedding de texto serializado.

    Pipeline:
    1. Serializa estatísticas do dataset como texto descritivo
    2. Gera embedding com sentence-transformers
    3. Retorna componentes do embedding (PCA é feito no treinamento)
    """

    def __init__(self, n_components: int = 10, cache_dir: str | None = None):
        self.n_components = n_components
        self._model = None
        self._cache: dict = {}
        self._cache_dir = cache_dir
        self._disk_cache: dict = {}

        if cache_dir:
            self._load_disk_cache()

    def _get_model(self):
        """Lazy load do modelo de embeddings."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._model

    def _load_disk_cache(self):
        """Carrega cache de embeddings do disco (formato JSON)."""
        if self._cache_dir:
            cache_path = os.path.join(self._cache_dir, ".embedding_cache.json")
            if os.path.exists(cache_path):
                try:
                    with open(cache_path) as f:
                        self._disk_cache = json.load(f)
                except Exception:
                    self._disk_cache = {}

    def _save_disk_cache(self):
        """Salva cache de embeddings em disco (formato JSON)."""
        if self._cache_dir:
            os.makedirs(self._cache_dir, exist_ok=True)
            cache_path = os.path.join(self._cache_dir, ".embedding_cache.json")
            try:
                with open(cache_path, "w") as f:
                    json.dump(self._disk_cache, f)
            except Exception:
                pass

    def extract_features(self, df: pd.DataFrame, use_cache: bool = True) -> dict:
        """
        Extrai features de embedding de um DataFrame.

        Returns:
            Dict com 'emb_0', 'emb_1', ..., 'emb_{n_components-1}'
        """
        text = self._serialize_dataset(df)

        cache_key = hashlib.md5(text.encode()).hexdigest()

        # Check memory cache
        if use_cache and cache_key in self._cache:
            return self._cache[cache_key]

        # Check disk cache
        if use_cache and cache_key in self._disk_cache:
            result = self._disk_cache[cache_key]
            self._cache[cache_key] = result
            return result

        # Generate embedding
        model = self._get_model()
        embedding = model.encode(text, show_progress_bar=False)

        # Truncate to n_components (PCA será aplicado no treinamento se necessário)
        # O embedding do MiniLM tem 384 dims — usamos os primeiros n_components
        emb_values = embedding[: self.n_components].astype(float)

        result = {f"emb_{i}": float(emb_values[i]) for i in range(len(emb_values))}

        if use_cache:
            self._cache[cache_key] = result
            self._disk_cache[cache_key] = result
            if len(self._disk_cache) % 50 == 0:
                self._save_disk_cache()

        return result

    def flush_cache(self):
        """Salva cache pendente em disco."""
        self._save_disk_cache()

    def _serialize_dataset(self, df: pd.DataFrame) -> str:
        """
        Serializa estatísticas do dataset como texto descritivo.

        Formato projetado para que o modelo de linguagem capture
        relações semânticas entre as estatísticas.
        """
        mask = df["X0"].isna().astype(int).values
        n_total = len(mask)
        n_missing = int(mask.sum())
        missing_rate = n_missing / n_total

        X0_obs = df["X0"].dropna().values

        parts = [
            f"Dataset with {n_total} observations and {missing_rate:.1%} missing values in X0.",
        ]

        if len(X0_obs) > 5:
            parts.append(
                f"X0 observed: mean={np.mean(X0_obs):.3f}, "
                f"std={np.std(X0_obs):.3f}, "
                f"skew={sp_stats.skew(X0_obs):.3f}, "
                f"kurtosis={sp_stats.kurtosis(X0_obs, fisher=True):.3f}, "
                f"min={np.min(X0_obs):.3f}, max={np.max(X0_obs):.3f}."
            )

            p5, p25, p50, p75, p95 = np.percentile(X0_obs, [5, 25, 50, 75, 95])
            parts.append(
                f"X0 percentiles: p5={p5:.3f}, p25={p25:.3f}, " f"p50={p50:.3f}, p75={p75:.3f}, p95={p95:.3f}."
            )

        # Correlações mask-Xi
        corrs = []
        for col in ["X1", "X2", "X3", "X4"]:
            if col in df.columns:
                xi = df[col].values
                if np.std(xi) > 0 and np.std(mask) > 0:
                    corr = np.corrcoef(mask, xi)[0, 1]
                    if not np.isnan(corr):
                        corrs.append(f"mask-{col}={corr:.3f}")
        if corrs:
            parts.append(f"Correlations: {', '.join(corrs)}.")

        # Missing rate por quartil
        if len(X0_obs) > 10:
            X0_imputed = df["X0"].fillna(df["X0"].median()).values
            quartiles = np.percentile(X0_imputed, [25, 50, 75])
            bins = [-np.inf, quartiles[0], quartiles[1], quartiles[2], np.inf]
            bin_idx = np.digitize(X0_imputed, bins[1:-1])
            rates = []
            for q in range(4):
                q_mask = bin_idx == q
                if q_mask.sum() > 0:
                    rates.append(f"Q{q+1}={mask[q_mask].mean():.3f}")
            if rates:
                parts.append(f"Missing rate by X0 quartile: {', '.join(rates)}.")

        # Mean diffs X1 por grupo
        X1 = df["X1"].values
        X1_miss = X1[mask == 1]
        X1_obs = X1[mask == 0]
        if len(X1_miss) > 0 and len(X1_obs) > 0:
            diff = np.mean(X1_miss) - np.mean(X1_obs)
            parts.append(f"X1 mean difference (missing-observed): {diff:.3f}.")

        return " ".join(parts)


def get_embedding_fallback_features(n_components: int = 10) -> dict:
    """Retorna features padrão quando embeddings não estão disponíveis."""
    return {f"emb_{i}": 0.0 for i in range(n_components)}
