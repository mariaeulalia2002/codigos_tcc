"""
Verifica o intervalo amostral nos cabecalhos binario e de tracos de um SEG-Y.

O numero padrao de tracos inspecionados e 200, conforme descrito no TCC.

Exemplo:
    python verificacao_headers_segy.py levantamento.sgy --tracos 200
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import segyio


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compara o intervalo amostral dos headers binario e de tracos."
    )
    parser.add_argument("segy", type=Path, help="Arquivo SEG-Y a verificar.")
    parser.add_argument(
        "--tracos",
        type=int,
        default=200,
        help="Numero de trace headers a inspecionar (padrao: 200).",
    )
    argumentos = parser.parse_args()

    if not argumentos.segy.is_file():
        parser.error(f"Arquivo SEG-Y nao encontrado: {argumentos.segy}")
    if argumentos.tracos <= 0:
        parser.error("--tracos deve ser maior que zero")

    with segyio.open(argumentos.segy, "r", ignore_geometry=True) as arquivo:
        print("Informacoes do arquivo")
        print(f"Total de tracos: {arquivo.tracecount}")
        print(f"Numero de amostras por traco: {len(arquivo.samples)}")

        dt_bin_micro = arquivo.bin[segyio.BinField.Interval]
        dt_bin_s = dt_bin_micro * 1e-6
        frequencia_amostragem = 1.0 / dt_bin_s
        print(f"dt do binary header: {dt_bin_micro} us ({dt_bin_s:.6f} s)")
        print(f"Frequencia de amostragem: {frequencia_amostragem:.2f} Hz")

        quantidade = min(argumentos.tracos, arquivo.tracecount)
        intervalos = np.asarray(
            [
                arquivo.header[indice][segyio.TraceField.TRACE_SAMPLE_INTERVAL]
                for indice in range(quantidade)
            ]
        )
        print(f"Trace headers analisados: {quantidade}")
        print(f"dt minimo: {intervalos.min()} us")
        print(f"dt maximo: {intervalos.max()} us")
        print(f"dt medio: {intervalos.mean():.2f} us")
        print(
            "Todos os trace headers possuem o mesmo dt: "
            + ("sim" if np.all(intervalos == intervalos[0]) else "nao")
        )
        print(
            "O dt dos trace headers coincide com o binary header: "
            + ("sim" if np.all(intervalos == dt_bin_micro) else "nao")
        )


if __name__ == "__main__":
    main()
