"""
Identifica possiveis picos secundarios em uma curva HVSR ja processada.

A curva de entrada deve ter sido calculada previamente pelo hvsrweb/hvsrpy.
Este script nao calcula o HVSR, nao determina a frequencia fundamental f0 e
nao rejeita janelas temporais. O valor de f0 e fornecido como entrada, e a
rotina adicional e aplicada somente a identificacao de possiveis f1 e f2.

Exemplo:
    python identificacao_picos_secundarios.py curva_hvsr.csv --f0 0.857 \
        --figura resultados/picos_secundarios.png \
        --saida resultados/picos_secundarios.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks


def ler_curva(caminho: Path) -> tuple[np.ndarray, np.ndarray]:
    dados = pd.read_csv(caminho, comment="#", header=None)
    if dados.shape[1] < 2:
        raise ValueError("O CSV deve conter frequencia e razao H/V nas duas primeiras colunas.")
    frequencia = pd.to_numeric(dados.iloc[:, 0], errors="coerce").to_numpy()
    hvsr = pd.to_numeric(dados.iloc[:, 1], errors="coerce").to_numpy()
    validos = np.isfinite(frequencia) & np.isfinite(hvsr)
    return frequencia[validos], hvsr[validos]


def identificar_picos(
    frequencia: np.ndarray,
    hvsr: np.ndarray,
    f0: float,
    largura_bin_hz: float,
    altura_minima: float,
    proeminencia_minima: float,
    separacao_relativa: float,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    selecionados = hvsr >= altura_minima
    tabela = pd.DataFrame(
        {"frequencia_hz": frequencia[selecionados], "hvsr": hvsr[selecionados]}
    )
    if tabela.empty:
        raise ValueError("Nenhuma amostra atende ao limite minimo de H/V.")

    tabela["bin"] = (tabela["frequencia_hz"] / largura_bin_hz).astype(int)
    agrupada = tabela.groupby("bin", as_index=False).agg(
        frequencia_hz=("frequencia_hz", "mean"), hvsr=("hvsr", "mean")
    )

    picos, propriedades = find_peaks(
        agrupada["hvsr"].to_numpy(),
        height=altura_minima,
        prominence=proeminencia_minima,
    )
    frequencias = agrupada["frequencia_hz"].to_numpy()[picos]
    amplitudes = propriedades["peak_heights"]

    # Os picos retornados seguem a ordem crescente dos indices/frequencias.
    frequencias_limpas: list[float] = []
    amplitudes_limpas: list[float] = []
    for frequencia_pico, amplitude in zip(frequencias, amplitudes):
        if not frequencias_limpas or (
            abs(frequencia_pico - frequencias_limpas[-1])
            / frequencias_limpas[-1]
            > separacao_relativa
        ):
            frequencias_limpas.append(float(frequencia_pico))
            amplitudes_limpas.append(float(amplitude))

    limite_regiao_f0 = f0 * (1.0 + separacao_relativa)
    secundarios = [
        (freq, amp)
        for freq, amp in zip(frequencias_limpas, amplitudes_limpas)
        if freq > limite_regiao_f0
    ]

    linhas = [{"pico": "f0", "frequencia_hz": f0, "amplitude_hv": np.nan}]
    for indice, (freq, amp) in enumerate(secundarios[:2], start=1):
        linhas.append(
            {"pico": f"f{indice}", "frequencia_hz": freq, "amplitude_hv": amp}
        )

    return pd.DataFrame(linhas), agrupada["frequencia_hz"].to_numpy(), agrupada[
        "hvsr"
    ].to_numpy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Identifica f1 e f2 em uma curva exportada pelo hvsrweb/hvsrpy."
    )
    parser.add_argument("csv", type=Path, help="Curva HVSR em CSV.")
    parser.add_argument("--f0", type=float, required=True, help="Frequencia fundamental obtida pelo hvsrpy.")
    parser.add_argument("--bin", type=float, default=0.2, dest="largura_bin", help="Largura dos intervalos de frequencia em Hz (padrao: 0.2).")
    parser.add_argument("--altura", type=float, default=1.5, help="Altura H/V minima (padrao: 1.5).")
    parser.add_argument("--proeminencia", type=float, default=0.4, help="Proeminencia minima (padrao: 0.4).")
    parser.add_argument("--separacao", type=float, default=0.15, help="Separacao relativa minima (padrao: 0.15).")
    parser.add_argument("--saida", type=Path, help="CSV opcional dos picos identificados.")
    parser.add_argument("--figura", type=Path, help="Figura opcional em PNG/PDF.")
    parser.add_argument("--mostrar", action="store_true", help="Exibe a figura em uma janela.")
    argumentos = parser.parse_args()

    if not argumentos.csv.is_file():
        parser.error(f"Arquivo CSV nao encontrado: {argumentos.csv}")
    if argumentos.f0 <= 0 or argumentos.largura_bin <= 0:
        parser.error("f0 e a largura do bin devem ser maiores que zero")

    frequencia, hvsr = ler_curva(argumentos.csv)
    picos, frequencia_agrupada, hvsr_agrupado = identificar_picos(
        frequencia,
        hvsr,
        argumentos.f0,
        argumentos.largura_bin,
        argumentos.altura,
        argumentos.proeminencia,
        argumentos.separacao,
    )
    print(picos.to_string(index=False))

    if argumentos.saida:
        argumentos.saida.parent.mkdir(parents=True, exist_ok=True)
        picos.to_csv(argumentos.saida, index=False, float_format="%.6f")
        print(f"Tabela salva em: {argumentos.saida.resolve()}")

    figura, eixo = plt.subplots(figsize=(9, 5))
    eixo.semilogx(frequencia, hvsr, alpha=0.30, label="HVSR original")
    eixo.semilogx(frequencia_agrupada, hvsr_agrupado, linewidth=2, label="HVSR agrupado")
    cores = {"f0": "green", "f1": "orange", "f2": "purple"}
    for linha in picos.itertuples(index=False):
        eixo.axvline(
            linha.frequencia_hz,
            color=cores[linha.pico],
            linestyle="--",
            label=f"{linha.pico} = {linha.frequencia_hz:.3f} Hz",
        )
    eixo.set_xlabel("Frequencia (Hz)")
    eixo.set_ylabel("H/V")
    eixo.set_title("Identificacao de picos secundarios")
    eixo.grid(True, which="both", alpha=0.4)
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
