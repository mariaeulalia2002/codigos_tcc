"""
Extrai profundidade, tempo e velocidade intervalar do VSP do poco 137.

Esta e uma etapa preparatoria. O script le o arquivo DLIS, identifica os
canais de profundidade e tempo de percurso, calcula velocidades intervalares
entre registros consecutivos e exporta uma tabela CSV.

O script nao seleciona intervalos de MTDs/MTCs, nao calcula velocidades
equivalentes dos eventos e nao realiza ponderacao pela distancia aos pocos.
Esses procedimentos sao executados posteriormente por
``conversao_tempo_profundidade.py``.

Exemplo:
    python extracao_dados_vsp_poco_137.py dados/poco_137.dlis \
        --saida resultados/velocidades_originais_poco_137.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from dlisio import dlis


CANAIS_PROFUNDIDADE = ("U-SEIS_TVDSRD1", "DEP1")
CANAIS_TEMPO = ("CTT1", "TOFS")


def obter_canal(frame, nomes: tuple[str, ...]) -> tuple[np.ndarray | None, str | None]:
    for nome in nomes:
        for canal in frame.channels:
            if canal.name != nome:
                continue
            try:
                valores = np.asarray(canal.curves(), dtype=float)
            except Exception as erro:
                print(f"Nao foi possivel ler o canal {nome}: {erro}")
                continue
            if valores.size:
                return valores, nome
    return None, None


def extrair_modelo(caminho_dlis: Path) -> np.ndarray:
    with dlis.load(caminho_dlis) as arquivos:
        for arquivo in arquivos:
            for frame in arquivo.frames:
                profundidade, canal_profundidade = obter_canal(
                    frame, CANAIS_PROFUNDIDADE
                )
                tempo, canal_tempo = obter_canal(frame, CANAIS_TEMPO)

                if profundidade is None or tempo is None:
                    continue

                print(
                    f"Frame {frame.name}: profundidade={canal_profundidade}; "
                    f"tempo={canal_tempo}"
                )

                n = min(profundidade.size, tempo.size)
                profundidade = profundidade[:n]
                tempo = tempo[:n]
                validos = np.isfinite(profundidade) & np.isfinite(tempo)
                profundidade = profundidade[validos]
                tempo = tempo[validos]

                # Mantem as conversoes de unidade utilizadas na analise original.
                if np.max(profundidade) < 1000:
                    profundidade = profundidade * 0.3048
                if np.max(tempo) > 10:
                    tempo = tempo / 1000.0

                profundidade, indices_unicos = np.unique(
                    profundidade, return_index=True
                )
                tempo = tempo[indices_unicos]
                ordem = np.argsort(profundidade)
                profundidade = profundidade[ordem]
                tempo = tempo[ordem]

                delta_z = np.diff(profundidade)
                delta_t = np.diff(tempo)
                validos = (delta_z != 0) & (delta_t != 0)
                velocidade = delta_z[validos] / delta_t[validos]
                profundidade_media = (
                    (profundidade[:-1] + profundidade[1:]) / 2.0
                )[validos]
                tempo_inicial = tempo[:-1][validos]

                # Faixa fisicamente plausivel adotada na preparacao original.
                fisicos = (velocidade > 1000) & (velocidade < 6000)
                velocidade = velocidade[fisicos]
                profundidade_media = profundidade_media[fisicos]
                twt_ms = tempo_inicial[fisicos] * 2.0 * 1000.0

                return np.column_stack(
                    [profundidade_media, velocidade, twt_ms]
                )

    raise RuntimeError(
        "Nenhum frame contendo canais compativeis de profundidade e tempo foi encontrado."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extrai velocidades intervalares do VSP do poco 137."
    )
    parser.add_argument("dlis", type=Path, help="Arquivo DLIS do poco 137.")
    parser.add_argument(
        "--saida",
        type=Path,
        default=Path("velocidades_originais_poco_137.csv"),
        help="CSV de saida (padrao: velocidades_originais_poco_137.csv).",
    )
    argumentos = parser.parse_args()

    if not argumentos.dlis.is_file():
        parser.error(f"Arquivo DLIS nao encontrado: {argumentos.dlis}")

    tabela = extrair_modelo(argumentos.dlis)
    argumentos.saida.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        argumentos.saida,
        tabela,
        delimiter=",",
        fmt="%.3f",
        header="profundidade_m,velocidade_original_m_s,twt_ms",
        comments="",
    )
    print(f"Tabela salva em: {argumentos.saida.resolve()}")


if __name__ == "__main__":
    main()
