"""
Reproduz as velocidades equivalentes adotadas na conversao tempo-profundidade.

O codigo utiliza os controles publicados nas Tabelas D.1 e D.2 do TCC e os
intervalos medios de TWT abaixo do fundo marinho apresentados na Tabela D.3.

Etapas:
1. constroi um modelo cumulativo de TWT abaixo do fundo para cada poco;
2. converte os tempos medios de topo e base dos eventos em profundidades;
3. calcula a velocidade equivalente de cada intervalo;
4. combina as velocidades dos dois pocos pelo inverso da distancia.

Este codigo nao le o arquivo DLIS original. A extracao do DLIS constitui uma
etapa anterior e independente. Os dados brutos de poco nao sao distribuidos
com este script.

Uso:
    python conversao_tempo_profundidade.py
    python conversao_tempo_profundidade.py --saida resultados

O argumento --saida e opcional. Quando informado, o programa grava uma tabela
CSV com os resultados intermediarios e finais.
"""

from __future__ import annotations

import argparse
import csv
from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable


LAMINA_DAGUA_137_M = 1672.0
LAMINA_DAGUA_127_M = 1831.0

# Tabela D.2: profundidade total (m) e velocidade da media movel ponderada
# (m/s). Esta foi a curva utilizada na conversao morfometrica dos MTDs/MTCs.
CONTROLES_137 = [
    (1849.0, 1567.280),
    (2073.0, 1855.715),
    (2251.0, 2094.440),
    (2397.0, 2254.315),
    (2472.0, 2319.271),
    (2530.0, 2374.271),
    (2578.0, 2407.406),
    (2639.0, 2412.433),
    (2737.0, 2416.881),
    (2845.0, 2474.852),
    (2915.0, 2530.723),
    (2945.0, 2572.784),
    (2995.0, 2684.996),
    (3066.0, 2769.583),
    (3141.0, 2867.208),
    (3327.0, 2978.149),
    (3372.0, 2999.916),
]

# Tabela D.1: profundidade total (m) e TWT registrado no relatorio (ms).
CONTROLES_127 = [
    (1854.0, 2425.68),
    (2059.0, 2693.94),
    (2479.0, 3136.78),
    (2732.0, 3362.78),
    (3065.0, 3642.54),
    (3300.0, 3822.20),
    (3520.0, 3982.00),
    (3682.0, 4091.40),
    (3894.0, 4228.92),
]


@dataclass(frozen=True)
class Evento:
    nome: str
    twt_topo_ms: float
    twt_base_ms: float
    distancia_137_km: float
    distancia_127_km: float


# Intervalos da Tabela D.3 e distancias apresentadas na Tabela D.4.
EVENTOS = [
    Evento("MTD 1", 64.332, 121.752, 48.810, 295.520),
    Evento("MTD 2", 56.275, 104.715, 32.300, 69.220),
    Evento("MTD 3", 265.065, 339.015, 44.080, 9.990),
    Evento("MTC 1-A", 262.787, 359.340, 38.480, 39.807),
    Evento("MTC 1-B", 317.655, 426.879, 20.105, 48.471),
    Evento("MTC 1-C", 111.102, 223.102, 31.032, 51.170),
    Evento("MTC 2-A", 324.596, 383.191, 21.019, 27.910),
    Evento("MTC 2-B", 214.101, 301.062, 21.019, 27.910),
    Evento("MTC 2-C", 195.182, 257.786, 6.004, 40.198),
    Evento("MTC 2-D", 119.202, 222.432, 6.705, 36.022),
]


def interpolar(x: float, xs: list[float], ys: list[float]) -> float:
    """Interpolacao linear de x em uma serie monotonicamente crescente."""
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]

    indice = bisect_right(xs, x) - 1
    fracao = (x - xs[indice]) / (xs[indice + 1] - xs[indice])
    return ys[indice] + fracao * (ys[indice + 1] - ys[indice])


def modelo_velocidade_137() -> tuple[Callable[[float], float], float]:
    profundidades = [p - LAMINA_DAGUA_137_M for p, _ in CONTROLES_137]
    velocidades = [v for _, v in CONTROLES_137]

    def velocidade(profundidade: float) -> float:
        # Extensao constante da velocidade mais rasa ate o fundo marinho.
        if profundidade <= profundidades[0]:
            return velocidades[0]
        return interpolar(profundidade, profundidades, velocidades)

    return velocidade, profundidades[-1]


def construir_relacao_127() -> tuple[list[float], list[float]]:
    """Constroi exatamente o modelo por camadas do poco 127."""
    profundidades = [p - LAMINA_DAGUA_127_M for p, _ in CONTROLES_127]
    tempos = [t for _, t in CONTROLES_127]

    velocidades_intervalares = [
        2000.0 * (CONTROLES_127[i + 1][0] - CONTROLES_127[i][0])
        / (tempos[i + 1] - tempos[i])
        for i in range(len(CONTROLES_127) - 1)
    ]

    # A primeira velocidade intervalar e estendida desde o primeiro controle
    # ate o fundo marinho. Os limites seguintes coincidem com os controles de
    # profundidade do relatorio.
    limites_profundidade = [0.0] + profundidades[1:]
    tempos_cumulativos = [0.0]
    for indice, velocidade in enumerate(velocidades_intervalares):
        delta_z = limites_profundidade[indice + 1] - limites_profundidade[indice]
        tempos_cumulativos.append(
            tempos_cumulativos[-1] + 2000.0 * delta_z / velocidade
        )

    return limites_profundidade, tempos_cumulativos


def construir_relacao_cumulativa(
    velocidade: Callable[[float], float],
    profundidade_maxima: float,
    passo_m: float,
) -> tuple[list[float], list[float]]:
    """Acumula dT = 2000 dz/v desde o fundo marinho."""
    profundidades = [0.0]
    tempos = [0.0]
    z = 0.0

    while z < profundidade_maxima:
        proximo_z = min(z + passo_m, profundidade_maxima)
        # Media das velocidades nos extremos do pequeno intervalo.
        velocidade_media = (velocidade(z) + velocidade(proximo_z)) / 2.0
        incremento_twt = 2000.0 * (proximo_z - z) / velocidade_media
        profundidades.append(proximo_z)
        tempos.append(tempos[-1] + incremento_twt)
        z = proximo_z

    return profundidades, tempos


def velocidade_equivalente(
    twt_topo_ms: float,
    twt_base_ms: float,
    profundidades: list[float],
    tempos: list[float],
) -> tuple[float, float, float]:
    profundidade_topo = interpolar(twt_topo_ms, tempos, profundidades)
    profundidade_base = interpolar(twt_base_ms, tempos, profundidades)
    velocidade = (
        2000.0
        * (profundidade_base - profundidade_topo)
        / (twt_base_ms - twt_topo_ms)
    )
    return profundidade_topo, profundidade_base, velocidade


def peso_inverso_distancia(distancia_alvo: float, distancia_outro: float) -> float:
    """Peso normalizado do controle-alvo para expoente IDW igual a 1."""
    inverso_alvo = 1.0 / distancia_alvo
    inverso_outro = 1.0 / distancia_outro
    return inverso_alvo / (inverso_alvo + inverso_outro)


def calcular_resultados(passo_m: float) -> list[dict[str, float | str]]:
    velocidade_137, maximo_137 = modelo_velocidade_137()
    z137, t137 = construir_relacao_cumulativa(velocidade_137, maximo_137, passo_m)
    z127, t127 = construir_relacao_127()

    resultados: list[dict[str, float | str]] = []
    for evento in EVENTOS:
        topo_137, base_137, veq_137 = velocidade_equivalente(
            evento.twt_topo_ms, evento.twt_base_ms, z137, t137
        )
        topo_127, base_127, veq_127 = velocidade_equivalente(
            evento.twt_topo_ms, evento.twt_base_ms, z127, t127
        )

        peso_137 = peso_inverso_distancia(
            evento.distancia_137_km, evento.distancia_127_km
        )
        peso_127 = 1.0 - peso_137
        velocidade_ponderada = peso_137 * veq_137 + peso_127 * veq_127

        resultados.append(
            {
                "evento": evento.nome,
                "twt_topo_ms": evento.twt_topo_ms,
                "twt_base_ms": evento.twt_base_ms,
                "profundidade_topo_137_m": topo_137,
                "profundidade_base_137_m": base_137,
                "velocidade_equivalente_137_m_s": veq_137,
                "profundidade_topo_127_m": topo_127,
                "profundidade_base_127_m": base_127,
                "velocidade_equivalente_127_m_s": veq_127,
                "distancia_137_km": evento.distancia_137_km,
                "distancia_127_km": evento.distancia_127_km,
                "peso_137_percentual": 100.0 * peso_137,
                "peso_127_percentual": 100.0 * peso_127,
                "velocidade_ponderada_m_s": velocidade_ponderada,
            }
        )

    return resultados


def imprimir_resultados(resultados: Iterable[dict[str, float | str]]) -> None:
    cabecalho = (
        f"{'Evento':<9} {'Veq 137':>10} {'Veq 127':>10} "
        f"{'Peso 137':>10} {'Peso 127':>10} {'V ponderada':>12}"
    )
    print(cabecalho)
    print("-" * len(cabecalho))
    for linha in resultados:
        print(
            f"{linha['evento']:<9} "
            f"{linha['velocidade_equivalente_137_m_s']:>10.2f} "
            f"{linha['velocidade_equivalente_127_m_s']:>10.2f} "
            f"{linha['peso_137_percentual']:>9.2f}% "
            f"{linha['peso_127_percentual']:>9.2f}% "
            f"{linha['velocidade_ponderada_m_s']:>12.2f}"
        )


def salvar_csv(resultados: list[dict[str, float | str]], pasta: Path) -> Path:
    pasta.mkdir(parents=True, exist_ok=True)
    destino = pasta / "velocidades_equivalentes_eventos.csv"
    with destino.open("w", newline="", encoding="utf-8-sig") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=list(resultados[0].keys()))
        escritor.writeheader()
        escritor.writerows(resultados)
    return destino


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduz as velocidades equivalentes das Tabelas D.3 e D.4."
    )
    parser.add_argument(
        "--saida",
        type=Path,
        help="Pasta opcional para gravacao do arquivo CSV de resultados.",
    )
    parser.add_argument(
        "--passo",
        type=float,
        default=0.01,
        help="Passo vertical da integracao numerica, em metros (padrao: 0.01).",
    )
    argumentos = parser.parse_args()

    if argumentos.passo <= 0:
        parser.error("--passo deve ser maior que zero")

    resultados = calcular_resultados(argumentos.passo)
    imprimir_resultados(resultados)

    if argumentos.saida:
        destino = salvar_csv(resultados, argumentos.saida)
        print(f"\nArquivo salvo em: {destino.resolve()}")


if __name__ == "__main__":
    main()
