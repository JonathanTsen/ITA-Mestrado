"""missdetect — Hierarchical classification of missing-data mechanisms (MCAR / MAR / MNAR).

Companion code for the master's thesis "Hierarchical Classification of Missing
Data Mechanisms: A Statistical Feature Engineering Approach with LLM Augmentation
and Real-World Validation" (ITA, 2026).

Top-level subpackages:
    features   — statistical and discriminative feature extractors
    llm        — LLM-augmented feature extraction (judge, context-aware, self-consistency, embeddings, CAAFE)
    baselines  — reference implementations (MechDetect, PKLM)
    utils      — argument parsing and path helpers
    data_generation — scripts to (re)generate synthetic and real datasets
    metadata   — JSON/text metadata used by LLM prompts (not raw data)
"""

__version__ = "0.1.0"
