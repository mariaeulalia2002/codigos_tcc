"""
Compara tres procedimentos de suavizacao das velocidades do poco 137.

Entrada esperada: CSV gerado por ``extracao_dados_vsp_poco_137.py``, contendo
profundidade, velocidade intervalar original e TWT.

Aplicacoes no TCC:
- media movel ponderada (0,25; 0,50; 0,25): conversao morfometrica dos
  MTDs/MTCs;
- filtro Savitzky-Golay (janela 5, polinomio de ordem 2): valores rasos usados
  na estimativa de velocidades da analise HVSR;
- media movel simples: comparacao entre alternativas de suavizacao.

Exemplo:
    python suavizacao_poco_137.py velocidades_originais_poco_137.csv \
        --saida resultados/velocidades_suavizadas_poco_137.csv \
        --figura resultados/curvas_velocidade_poco_137.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def suavizar(velocidade: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if velocidade.size < 5:
        raise ValueError("Sao necessarias pelo menos cinco amostras de velocidade.")

    media_movel = velocidade.copy()
    media_movel[1:-1] = (
        velocidade[:-2] + velocidade[1:-1] + velocidade[2:]
    ) / 3.0

    media_ponderada = velocidade.copy()
    media_ponderada[1:-1] = (
        0.25 * velocidade[:-2]
        + 0.50 * velocidade[1:-1]
        + 0.25 * velocidade[2:]
    )

    savitzky_golay = savgol_filter(velocidade, window_length=5, polyorder=2)
    return media_movel, savitzky_golay, media_ponderada


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Gera as curvas suavizadas de velocidade do poco 137."
    )
    parser.add_argument("csv", type=Path, help="CSV de velocidades originais.")
    parser.add_argument(
        "--saida",
        type=Path,
        default=Path("velocidades_suavizadas_poco_137.csv"),
        help="Tabela CSV de saida.",
    )
    parser.add_argument("--figura", type=Path, help="Arquivo opcional da figura.")
    parser.add_argument(
        "--mostrar", action="store_true", help="Exibe a figura em uma janela."
    )
    argumentos = parser.parse_args()

    if not argumentos.csv.is_file():
        parser.error(f"Arquivo CSV nao encontrado: {argumentos.csv}")

    dados = pd.read_csv(argumentos.csv)
    obrigatorias = {
        "profundidade_m",
        "velocidade_original_m_s",
        "twt_ms",
    }
    ausentes = obrigatorias.difference(dados.columns)
    if ausentes:
        parser.error("Colunas ausentes no CSV: " + ", ".join(sorted(ausentes)))

    profundidade = dados["profundidade_m"].to_numpy(dtype=float)
    velocidade = dados["velocidade_original_m_s"].to_numpy(dtype=float)
    twt = dados["twt_ms"].to_numpy(dtype=float)
    media, savgol, ponderada = suavizar(velocidade)

    resultado = pd.DataFrame(
        {
            "profundidade_m": profundidade,
            "velocidade_original_m_s": velocidade,
            "twt_ms": twt,
            "media_movel_m_s": media,
            "savitzky_golay_m_s": savgol,
            "media_movel_ponderada_m_s": ponderada,
        }
    )
    argumentos.saida.parent.mkdir(parents=True, exist_ok=True)
    resultado.to_csv(argumentos.saida, index=False, float_format="%.3f")
    print(f"Tabela salva em: {argumentos.saida.resolve()}")

    figura, eixo = plt.subplots(figsize=(6, 9))
    eixo.plot(velocidade, profundidade, "--", alpha=0.45, label="Original")
    eixo.plot(media, profundidade, label="Media movel")
    eixo.plot(savgol, profundidade, label="Savitzky-Golay")
    eixo.plot(ponderada, profundidade, label="Media movel ponderada")
    eixo.invert_yaxis()
    eixo.set_xlabel("Velocidade Vp (m/s)")
    eixo.set_ylabel("Profundidade (m)")
    eixo.set_title("Curvas de velocidade do poco 1-BRSA-137-RJS")
    eixo.grid(alpha=0.35)
    eixo.legend()
    figura.tight_layout()

    if argumentos.figura:
        argumentos.figura.parent.mkdir(parents=True, exist_ok=True)
        figura.savefig(argumentos.figura, dpi=300, bbox_inches="tight")
        print(f"Figura salva em: {argumentos.figura.resolve()}")
    if argumentos.mostrar:
        plt.show()
    plt.close(figura)


if __name__ == "__main__":
    main()
