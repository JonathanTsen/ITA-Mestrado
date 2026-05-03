"""
Gera amostras bootstrap dos dados reais processados.

Para cada arquivo original (ex: 736 linhas), gera N_BOOTSTRAP amostras
de CHUNK_SIZE linhas via reamostragem com reposicao. Shuffle automatico.

Isso resolve dois problemas do chunking sequencial:
1. Chunks sem missing (clustering temporal) -> bootstrap distribui uniformemente
2. Poucas amostras (n=43) -> bootstrap gera ~300 amostras (razao 16:1 amostras/features)

IMPORTANTE: Bootstraps do mesmo arquivo compartilham linhas. O train_model.py
deve usar GroupShuffleSplit para evitar data leakage no split treino/teste.
"""
import os
import pandas as pd
import shutil

CHUNK_SIZE = 100       # linhas por amostra
N_BOOTSTRAP = 50       # amostras por arquivo original
MIN_MISSING_RATE = 0.01  # descarta amostras com <1% missing

BASE = os.path.dirname(os.path.abspath(__file__))
PROCESSADO = os.path.join(BASE, "..", "Dataset", "real_data", "processado")
OUTPUT = os.path.join(BASE, "..", "Dataset", "real_data", "processado_chunks")
MECANISMOS = ["MCAR", "MAR", "MNAR"]


def gerar_bootstrap():
    if os.path.exists(OUTPUT):
        shutil.rmtree(OUTPUT)

    total = 0
    for mec in MECANISMOS:
        input_dir = os.path.join(PROCESSADO, mec)
        output_dir = os.path.join(OUTPUT, mec)
        os.makedirs(output_dir, exist_ok=True)

        arquivos = sorted([f for f in os.listdir(input_dir) if f.endswith(".txt")])

        for arq in arquivos:
            df = pd.read_csv(os.path.join(input_dir, arq), sep="\t")
            base_name = arq.replace(".txt", "")
            gerados = 0

            for i in range(N_BOOTSTRAP):
                # Reamostragem com reposicao (bootstrap)
                amostra = df.sample(n=min(CHUNK_SIZE, len(df)),
                                    replace=True,
                                    random_state=42 + i)

                # Verifica taxa de missing minima
                missing_rate = amostra["X0"].isna().mean()
                if missing_rate < MIN_MISSING_RATE:
                    continue

                out_name = f"{base_name}_boot{i+1:03d}.txt"
                amostra.to_csv(os.path.join(output_dir, out_name),
                               sep="\t", index=False)
                gerados += 1
                total += 1

            print(f"  {mec}/{arq}: {len(df)} linhas -> {gerados} bootstraps")

    print(f"\nTotal: {total} arquivos em {OUTPUT}")
    for mec in MECANISMOS:
        d = os.path.join(OUTPUT, mec)
        n = len([f for f in os.listdir(d) if f.endswith(".txt")])
        print(f"   {mec}: {n} arquivos")


if __name__ == "__main__":
    gerar_bootstrap()
