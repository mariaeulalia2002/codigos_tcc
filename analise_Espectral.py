"""
Estima a frequencia dominante em pontos selecionados de um volume SEG-Y.

Para cada ponto, o programa extrai uma vizinhanca de 3 x 3 tracos, remove a
tendencia linear, aplica uma janela de Hann e calcula o espectro de potencia.
A frequencia de pico e refinada por interpolacao parabolica local em escala
logaritmica. Tambem sao geradas tabelas por ponto, setor e evento, alem dos
graficos dos espectros.

Os pontos, tempos de referencia e bytes dos trace headers correspondem ao
levantamento analisado no TCC. O arquivo SEG-Y e a pasta de resultados sao
informados na linha de comando; nenhum caminho pessoal e armazenado aqui.

Exemplo:
    python analise_Espectral.py levantamento.sgy \
        --saida resultados/frequencias --janela-ms 256
"""

from pathlib import Path
import argparse
import os
import struct
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import segyio
from tqdm import tqdm


# ============================================================
# ARGUMENTOS E CONFIGURACOES
# ============================================================

parser = argparse.ArgumentParser(
    description=(
        "Calcula frequencias dominantes em pontos selecionados de um volume SEG-Y."
    )
)
parser.add_argument("segy", type=Path, help="Arquivo SEG-Y de entrada.")
parser.add_argument(
    "--saida",
    type=Path,
    default=Path("resultados_freq_mtd"),
    help="Diretorio dos CSVs e graficos (padrao: resultados_freq_mtd).",
)
parser.add_argument(
    "--janela-ms",
    type=float,
    default=256.0,
    help="Duracao da janela espectral em milissegundos (padrao: 256).",
)
argumentos = parser.parse_args()

SEGY_PATH = argumentos.segy
OUT_DIR = argumentos.saida
WINDOW_MS = argumentos.janela_ms

if not SEGY_PATH.is_file():
    parser.error(f"Arquivo SEG-Y nao encontrado: {SEGY_PATH}")
if WINDOW_MS <= 0:
    parser.error("--janela-ms deve ser maior que zero")

OUT_DIR.mkdir(parents=True, exist_ok=True)

PLOT_DIR = OUT_DIR / "espectros"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Geometria identificada no seu SEG-Y
IL_STEP = 1
XL_STEP = 2

# Vizinhança 3 x 3:
# IL-1 / IL / IL+1
# XL-2 / XL / XL+2
IL_OFFSETS = [-1, 0, 1]
XL_OFFSETS = [-2, 0, 2]

# Campos locais do seu SEG-Y (posição inicial, 1-based)
BYTE_X = 181
BYTE_Y = 185
BYTE_IL = 221
BYTE_XL = 225

# Delay recording time
BYTE_DELRT = 109


# ============================================================
# PONTOS SELECIONADOS NO PETREL
# ============================================================

POINTS = [

    # --------------------------------------------------------
    # SANTOS
    # --------------------------------------------------------
    dict(evento="Santos", setor="geral", ponto="S1",
         il=6011, xl=6905, x=778838.57, y=7325120.72,
         twt_ms=1904.90, referencia="topo"),

    dict(evento="Santos", setor="geral", ponto="S2",
         il=6011, xl=6741, x=779597.86, y=7323216.91,
         twt_ms=2098.65, referencia="base"),

    dict(evento="Santos", setor="geral", ponto="S3",
         il=6011, xl=6575, x=780368.73, y=7321284.03,
         twt_ms=2162.84, referencia="topo"),

    dict(evento="Santos", setor="geral", ponto="S4",
         il=6011, xl=6443, x=780973.62, y=7319767.34,
         twt_ms=2141.29, referencia="topo"),

    dict(evento="Santos", setor="geral", ponto="S5",
         il=6027, xl=6759, x=779884.18, y=7323578.75,
         twt_ms=2048.68, referencia="topo"),

    dict(evento="Santos", setor="geral", ponto="S6",
         il=6043, xl=6573, x=781113.97, y=7321575.00,
         twt_ms=2169.00, referencia="topo"),


    # --------------------------------------------------------
    # CONTINUAÇÃO SANTOS
    # --------------------------------------------------------
    dict(evento="Continuacao_Santos", setor="geral", ponto="CS1",
         il=5227, xl=6757, x=761315.77, y=7316147.74,
         twt_ms=2017.36, referencia="topo"),

    dict(evento="Continuacao_Santos", setor="geral", ponto="CS2",
         il=5243, xl=6661, x=762131.23, y=7315182.88,
         twt_ms=2133.02, referencia="topo"),

    dict(evento="Continuacao_Santos", setor="geral", ponto="CS3",
     il=5259, xl=6591, x=762830.14, y=7314510.21,
     twt_ms=2130.24, referencia="topo"),

    dict(evento="Continuacao_Santos", setor="geral", ponto="CS4",
         il=5275, xl=6725, x=762581.64, y=7316213.08,
         twt_ms=2077.48, referencia="topo"),


    # --------------------------------------------------------
    # REGIONAL MAIS RECENTE - RASO
    # --------------------------------------------------------
    dict(evento="Regional_recente", setor="raso", ponto="RR-r1",
         il=3819, xl=8385, x=721079.33, y=7322015.05,
         twt_ms=1440.09, referencia="topo"),

    dict(evento="Regional_recente", setor="raso", ponto="RR-r2",
         il=3835, xl=8279, x=721948.87, y=7320914.56,
         twt_ms=1482.13, referencia="topo"),

    dict(evento="Regional_recente", setor="raso", ponto="RR-r3",
         il=3851, xl=8671, x=720499.78, y=7325627.75,
         twt_ms=1267.03, referencia="topo"),


    # --------------------------------------------------------
    # REGIONAL MAIS RECENTE - INTERMEDIÁRIO
    # --------------------------------------------------------
    dict(evento="Regional_recente", setor="intermediario", ponto="RR-i1",
         il=3867, xl=7361, x=726936.03, y=7310569.40,
         twt_ms=1799.74, referencia="topo"),

    dict(evento="Regional_recente", setor="intermediario", ponto="RR-i2",
         il=3883, xl=7231, x=727912.64, y=7309200.46,
         twt_ms=1841.64, referencia="topo"),

    dict(evento="Regional_recente", setor="intermediario", ponto="RR-i3",
         il=3899, xl=7123, x=728781.88, y=7308100.74,
         twt_ms=1872.98, referencia="topo"),


    # --------------------------------------------------------
    # REGIONAL MAIS RECENTE - DISTAL
    # --------------------------------------------------------
    dict(evento="Regional_recente", setor="distal", ponto="RR-d1",
         il=3915, xl=6549, x=731810.10, y=7301587.61,
         twt_ms=2135.91, referencia="topo"),

    dict(evento="Regional_recente", setor="distal", ponto="RR-d2",
         il=3931, xl=6395, x=732899.14, y=7299936.75,
         twt_ms=2103.54, referencia="topo"),

    dict(evento="Regional_recente", setor="distal", ponto="RR-d3",
         il=3947, xl=6293, x=733742.58, y=7298901.72,
         twt_ms=2099.62, referencia="topo"),


    # --------------------------------------------------------
    # REGIONAL - EVENTO INTERMEDIÁRIO
    # --------------------------------------------------------
    dict(evento="Regional_intermediario", setor="geral", ponto="RI1",
         il=3947, xl=6781, x=731479.57, y=7304575.93,
         twt_ms=2160.41, referencia="base"),

    dict(evento="Regional_intermediario", setor="geral", ponto="RI2",
         il=3963, xl=6621, x=732599.53, y=7302847.56,
         twt_ms=2296.75, referencia="base"),

    dict(evento="Regional_intermediario", setor="geral", ponto="RI3",
         il=3963, xl=6341, x=733889.11, y=7299614.09,
         twt_ms=2287.50, referencia="cursor"),


    # --------------------------------------------------------
    # REGIONAL MAIS ANTIGO - RASO
    # --------------------------------------------------------
    dict(evento="Regional_antigo", setor="raso", ponto="RA-r1",
         il=3099, xl=8541, x=703637.82, y=7317157.37,
         twt_ms=1361.98, referencia="cursor"),

    dict(evento="Regional_antigo", setor="raso", ponto="RA-r2",
         il=3115, xl=8417, x=704586.03, y=7315859.63,
         twt_ms=1428.13, referencia="cursor"),

    dict(evento="Regional_antigo", setor="raso", ponto="RA-r3",
         il=3163, xl=8657, x=704590.21, y=7319088.48,
         twt_ms=1303.54, referencia="cursor"),


    # --------------------------------------------------------
    # REGIONAL MAIS ANTIGO - INTERMEDIÁRIO
    # --------------------------------------------------------
    dict(evento="Regional_antigo", setor="intermediario", ponto="RA-i1",
         il=3003, xl=7345, x=706946.89, y=7302381.57,
         twt_ms=1922.26, referencia="cursor"),

    dict(evento="Regional_antigo", setor="intermediario", ponto="RA-i2",
         il=3019, xl=7203, x=707974.74, y=7300884.16,
         twt_ms=2015.75, referencia="cursor"),

    dict(evento="Regional_antigo", setor="intermediario", ponto="RA-i3",
         il=3051, xl=7547, x=707129.47, y=7305163.14,
         twt_ms=1842.04, referencia="topo"),


    # --------------------------------------------------------
    # REGIONAL MAIS ANTIGO - DISTAL
    # --------------------------------------------------------
    dict(evento="Regional_antigo", setor="distal", ponto="RA-d1",
         il=3083, xl=6755, x=711537.64, y=7296269.74,
         twt_ms=2346.72, referencia="cursor"),

    dict(evento="Regional_antigo", setor="distal", ponto="RA-d2",
         il=3083, xl=6593, x=712294.81, y=7294371.21,
         twt_ms=2385.52, referencia="cursor"),

    dict(evento="Regional_antigo", setor="distal", ponto="RA-d3",
         il=3083, xl=6431, x=713037.70, y=7292508.52,
         twt_ms=2405.15, referencia="cursor"),

    dict(evento="Regional_antigo", setor="distal", ponto="RA-d4",
         il=4107, xl=6427, x=736837.69, y=7301938.90,
         twt_ms=2424.34, referencia="topo"),


    # --------------------------------------------------------
    # BÚZIOS
    # --------------------------------------------------------
    dict(evento="Buzios", setor="geral", ponto="BZ1",
         il=4123, xl=5185, x=742957.37, y=7287674.34,
         twt_ms=2328.54, referencia="topo"),

    dict(evento="Buzios", setor="geral", ponto="BZ2",
         il=4155, xl=5217, x=743552.99, y=7288340.44,
         twt_ms=2267.96, referencia="topo"),

    dict(evento="Buzios", setor="geral", ponto="BZ3",
         il=4171, xl=4553, x=746999.18, y=7280779.31,
         twt_ms=2529.89, referencia="topo"),

    dict(evento="Buzios", setor="geral", ponto="BZ4",
         il=4187, xl=4297, x=748560.40, y=7277944.53,
         twt_ms=2670.85, referencia="topo"),

    dict(evento="Buzios", setor="geral", ponto="BZ5",
         il=4203, xl=4155, x=749590.17, y=7276442.27,
         twt_ms=2650.87, referencia="topo"),


    # --------------------------------------------------------
    # RECORRÊNCIA
    # --------------------------------------------------------
    dict(evento="Recorrencia", setor="geral", ponto="REC1",
         il=2507, xl=5769, x=702727.75, y=7279487.35,
         twt_ms=2395.63, referencia="topo"),

    dict(evento="Recorrencia", setor="geral", ponto="REC2",
         il=2523, xl=5423, x=704705.24, y=7275608.82,
         twt_ms=2687.82, referencia="topo"),

    dict(evento="Recorrencia", setor="geral", ponto="REC3",
         il=2523, xl=5185, x=705808.30, y=7272843.02,
         twt_ms=2784.96, referencia="topo"),

    dict(evento="Recorrencia", setor="geral", ponto="REC4",
         il=2539, xl=5039, x=706852.87, y=7271303.66,
         twt_ms=2787.73, referencia="topo"),

    dict(evento="Recorrencia", setor="geral", ponto="REC5",
         il=2555, xl=4991, x=707451.08, y=7270833.52,
         twt_ms=2658.53, referencia="topo"),


    # --------------------------------------------------------
    # REGIONAL ANTIGO ABAIXO DA RECORRÊNCIA
    # --------------------------------------------------------
    dict(evento="Regional_antigo", setor="sub_recorrencia", ponto="RA-REC1",
         il=2571, xl=5783, x=704151.28, y=7280237.15,
         twt_ms=2451.45, referencia="topo"),

    dict(evento="Regional_antigo", setor="sub_recorrencia", ponto="RA-REC2",
         il=2587, xl=5467, x=705983.22, y=7276723.55,
         twt_ms=2709.25, referencia="topo"),

    dict(evento="Regional_antigo", setor="sub_recorrencia", ponto="RA-REC3",
         il=2587, xl=5191, x=707261.21, y=7273519.14,
         twt_ms=2824.68, referencia="topo"),
]


# ============================================================
# LEITURA DOS TRACE HEADERS BRUTOS
# ============================================================

def i32(header_bytes, byte_position):
    """Lê inteiro signed de 4 bytes, big endian.
    byte_position é 1-based, como na especificação SEG-Y.
    """
    i = byte_position - 1
    return struct.unpack(">i", header_bytes[i:i + 4])[0]


def i16(header_bytes, byte_position):
    """Lê inteiro signed de 2 bytes, big endian."""
    i = byte_position - 1
    return struct.unpack(">h", header_bytes[i:i + 2])[0]


class HeaderReader:
    def __init__(self, path, tracecount, ns, bytes_per_sample=4):
        self.path = Path(path)
        self.tracecount = tracecount
        self.ns = ns
        self.bytes_per_sample = bytes_per_sample

        self.trace_bytes = 240 + ns * bytes_per_sample
        self.file_size = os.path.getsize(path)

        # Calcula automaticamente onde começa o primeiro trace.
        # Isso evita assumir que o arquivo possui exatamente 3600 bytes
        # de cabeçalhos antes dos traces.
        self.first_trace_offset = (
            self.file_size - self.tracecount * self.trace_bytes
        )

        self.fh = open(path, "rb")

    def close(self):
        self.fh.close()

    def header(self, trace_index):
        offset = (
            self.first_trace_offset
            + trace_index * self.trace_bytes
        )

        self.fh.seek(offset)
        b = self.fh.read(240)

        if len(b) != 240:
            raise IOError(
                f"Não foi possível ler o header do trace {trace_index}"
            )

        return b

    def values(self, trace_index):
        h = self.header(trace_index)

        return {
            "il": i32(h, BYTE_IL),
            "xl": i32(h, BYTE_XL),
            "x": i32(h, BYTE_X),
            "y": i32(h, BYTE_Y),
            "delrt_ms": i16(h, BYTE_DELRT),
        }


# ============================================================
# LOCALIZAÇÃO DOS TRAÇOS
# ============================================================

def trace_key(hr, idx):
    h = hr.header(idx)
    return (
        i32(h, BYTE_IL),
        i32(h, BYTE_XL),
    )


def check_sorting(hr, n=200):
    """
    Confere se o arquivo está ordenado por (IL, XL),
    usando amostragem ao longo do arquivo.
    """
    indices = np.linspace(
        0,
        hr.tracecount - 1,
        min(n, hr.tracecount),
        dtype=int
    )

    previous = None

    for idx in indices:
        key = trace_key(hr, idx)

        if previous is not None and key < previous:
            return False

        previous = key

    return True


def find_exact_trace(hr, target_il, target_xl):
    """
    Busca binária pelo par exato (IL, XL).
    Muito mais rápido do que varrer 14,7 milhões de traces.
    """
    target = (int(target_il), int(target_xl))

    lo = 0
    hi = hr.tracecount - 1

    while lo <= hi:
        mid = (lo + hi) // 2
        key = trace_key(hr, mid)

        if key == target:
            return mid

        if key < target:
            lo = mid + 1
        else:
            hi = mid - 1

    return None


# ============================================================
# ANÁLISE ESPECTRAL
# ============================================================

def remove_linear_trend(x):
    x = np.asarray(x, dtype=np.float64)

    n = len(x)
    t = np.arange(n, dtype=np.float64)

    p = np.polyfit(t, x, 1)
    trend = np.polyval(p, t)

    return x - trend


def normalized_power_spectrum(x, dt_s):
    """
    FFT usando janela Hann.

    Retorna potência normalizada.
    Nenhum zero-padding é utilizado para não criar
    falsa impressão de resolução espectral.
    """

    x = np.asarray(x, dtype=np.float64)

    x = remove_linear_trend(x)

    window = np.hanning(len(x))
    xw = x * window

    spectrum = np.fft.rfft(xw)
    power = np.abs(spectrum) ** 2

    freq = np.fft.rfftfreq(
        len(x),
        d=dt_s
    )

    # Remove componente DC da análise
    power[0] = 0.0

    total = power.sum()

    if not np.isfinite(total) or total <= 0:
        return freq, None

    power = power / total

    return freq, power


def parabolic_peak_frequency(freq, power):
    """
    Localiza o máximo do espectro e faz interpolação
    parabólica local em log-potência.

    Isso fornece uma estimativa mais suave da posição
    do pico, mas NÃO altera a resolução física da FFT.
    """

    if power is None:
        return np.nan

    idx = np.argmax(power)

    # Se o pico estiver numa extremidade
    if idx <= 0 or idx >= len(power) - 1:
        return float(freq[idx])

    eps = np.finfo(float).tiny

    y1 = np.log(power[idx - 1] + eps)
    y2 = np.log(power[idx] + eps)
    y3 = np.log(power[idx + 1] + eps)

    denominator = y1 - 2.0 * y2 + y3

    if abs(denominator) < 1e-15:
        return float(freq[idx])

    delta = 0.5 * (y1 - y3) / denominator

    df = freq[1] - freq[0]

    return float(freq[idx] + delta * df)


def spectral_metrics(freq, power):

    if power is None:
        return {
            "f_peak_hz": np.nan,
            "f_centroid_hz": np.nan,
            "f05_hz": np.nan,
            "f50_hz": np.nan,
            "f95_hz": np.nan,
        }

    f_peak = parabolic_peak_frequency(freq, power)

    f_centroid = np.sum(freq * power)

    cumulative = np.cumsum(power)

    def percentile_frequency(q):
        idx = np.searchsorted(cumulative, q)
        idx = min(idx, len(freq) - 1)
        return float(freq[idx])

    return {
        "f_peak_hz": f_peak,
        "f_centroid_hz": float(f_centroid),
        "f05_hz": percentile_frequency(0.05),
        "f50_hz": percentile_frequency(0.50),
        "f95_hz": percentile_frequency(0.95),
    }


# ============================================================
# EXECUÇÃO
# ============================================================

print("=" * 70)
print("ANÁLISE DE FREQUÊNCIA DOMINANTE DOS MTDs")
print("=" * 70)

with segyio.open(
    str(SEGY_PATH),
    mode="r",
    strict=False,
    ignore_geometry=True
) as segy:

    ntraces = segy.tracecount

    dt_us = int(segy.bin[segyio.BinField.Interval])
    ns = int(segy.bin[segyio.BinField.Samples])

    dt_s = dt_us / 1_000_000.0
    dt_ms = dt_us / 1000.0

    fs = 1.0 / dt_s
    nyquist = fs / 2.0

    n_window = int(round(WINDOW_MS / dt_ms))

    # força número par
    if n_window % 2 != 0:
        n_window += 1

    actual_window_ms = n_window * dt_ms

    # resolução nominal dos bins espectrais
    df_hz = fs / n_window

    print(f"\nTraços: {ntraces:,}")
    print(f"Amostras/traço: {ns}")
    print(f"dt: {dt_ms:.3f} ms")
    print(f"fs: {fs:.2f} Hz")
    print(f"Nyquist: {nyquist:.2f} Hz")

    print("\nJanela espectral:")
    print(f"  {n_window} amostras")
    print(f"  {actual_window_ms:.1f} ms")
    print(f"  Δf nominal = {df_hz:.4f} Hz")

    hr = HeaderReader(
        SEGY_PATH,
        tracecount=ntraces,
        ns=ns,
        bytes_per_sample=4
    )

    try:

        print("\nVerificando ordenação IL/XL...")

        if not check_sorting(hr):
            raise RuntimeError(
                "A amostragem dos headers não confirmou ordenação "
                "monótona por IL/XL. Pare aqui e me envie a saída."
            )

        print("✅ Ordenação compatível com busca binária.")

        point_results = []
        trace_results = []

        spectra_by_point = {}

        print("\nProcessando pontos...\n")

        for p in tqdm(POINTS):

            center_idx = find_exact_trace(
                hr,
                p["il"],
                p["xl"]
            )

            if center_idx is None:
                print(
                    f"\n⚠ Não encontrei exatamente "
                    f"{p['ponto']} IL={p['il']} XL={p['xl']}"
                )
                continue

            center_header = hr.values(center_idx)

            coord_error_m = math.hypot(
                center_header["x"] - p["x"],
                center_header["y"] - p["y"]
            )

            local_spectra = []
            local_metrics = []

            for dil in IL_OFFSETS:
                for dxl in XL_OFFSETS:

                    il2 = p["il"] + dil
                    xl2 = p["xl"] + dxl

                    idx = find_exact_trace(
                        hr,
                        il2,
                        xl2
                    )

                    if idx is None:
                        continue

                    hdr = hr.values(idx)

                    trace = np.asarray(
                        segy.trace[idx],
                        dtype=np.float64
                    )

                    center_sample = int(
                        round(
                            (
                                p["twt_ms"]
                                - hdr["delrt_ms"]
                            )
                            / dt_ms
                        )
                    )

                    start = center_sample - n_window // 2
                    end = start + n_window

                    if start < 0 or end > len(trace):
                        continue

                    segment = trace[start:end]

                    freq, power = normalized_power_spectrum(
                        segment,
                        dt_s
                    )

                    if power is None:
                        continue

                    metrics = spectral_metrics(
                        freq,
                        power
                    )

                    local_spectra.append(power)
                    local_metrics.append(
                        metrics["f_peak_hz"]
                    )

                    trace_results.append({
                        "evento": p["evento"],
                        "setor": p["setor"],
                        "ponto": p["ponto"],

                        "il": il2,
                        "xl": xl2,

                        "trace_index": idx,

                        "x_segy": hdr["x"],
                        "y_segy": hdr["y"],

                        "twt_referencia_ms": p["twt_ms"],

                        "inicio_janela_ms":
                            hdr["delrt_ms"]
                            + start * dt_ms,

                        "fim_janela_ms":
                            hdr["delrt_ms"]
                            + (end - 1) * dt_ms,

                        **metrics
                    })

            if len(local_spectra) == 0:
                print(
                    f"\n⚠ Nenhum espectro válido para {p['ponto']}"
                )
                continue

            local_spectra = np.asarray(
                local_spectra
            )

            # Mediana espectral da vizinhança.
            # Importante: não empilhamos amplitudes dos traces
            # antes da FFT, evitando cancelamentos de fase.
            median_power = np.median(
                local_spectra,
                axis=0
            )

            median_power = (
                median_power
                / median_power.sum()
            )

            metrics_point = spectral_metrics(
                freq,
                median_power
            )

            local_metrics = np.asarray(
                local_metrics,
                dtype=float
            )

            if len(local_metrics) > 1:
                local_sd = np.std(
                    local_metrics,
                    ddof=1
                )
            else:
                local_sd = np.nan

            point_results.append({

                "evento": p["evento"],
                "setor": p["setor"],
                "ponto": p["ponto"],
                "referencia": p["referencia"],

                "il_petrel": p["il"],
                "xl_petrel": p["xl"],

                "x_petrel": p["x"],
                "y_petrel": p["y"],

                "x_segy": center_header["x"],
                "y_segy": center_header["y"],

                "erro_localizacao_m": coord_error_m,

                "twt_ms": p["twt_ms"],

                "n_tracos_vizinhanca":
                    len(local_spectra),

                "janela_ms":
                    actual_window_ms,

                "df_fft_hz":
                    df_hz,

                "f_peak_hz":
                    metrics_point["f_peak_hz"],

                "f_centroid_hz":
                    metrics_point["f_centroid_hz"],

                "f05_hz":
                    metrics_point["f05_hz"],

                "f50_hz":
                    metrics_point["f50_hz"],

                "f95_hz":
                    metrics_point["f95_hz"],

                "sd_f_peak_vizinhanca_hz":
                    local_sd,
            })

            spectra_by_point[p["ponto"]] = {
                "evento": p["evento"],
                "setor": p["setor"],
                "freq": freq.copy(),
                "power": median_power.copy(),
            }

    finally:
        hr.close()


# ============================================================
# TABELAS
# ============================================================

df_points = pd.DataFrame(point_results)
df_traces = pd.DataFrame(trace_results)

df_points.to_csv(
    OUT_DIR / "frequencia_por_ponto.csv",
    index=False,
    sep=";",
    decimal=","
)

df_traces.to_csv(
    OUT_DIR / "frequencia_tracos_vizinhanca.csv",
    index=False,
    sep=";",
    decimal=","
)


def group_summary(df, columns):

    rows = []

    for keys, g in df.groupby(columns):

        if not isinstance(keys, tuple):
            keys = (keys,)

        f = g["f_peak_hz"].dropna().values

        row = dict(zip(columns, keys))

        row["n_pontos"] = len(f)

        if len(f) == 0:
            continue

        row["f_peak_media_hz"] = np.mean(f)
        row["f_peak_mediana_hz"] = np.median(f)

        if len(f) > 1:
            sd = np.std(f, ddof=1)
            sem = sd / np.sqrt(len(f))
        else:
            sd = np.nan
            sem = np.nan

        # Mantemos SD e SEM separados.
        # Depois decidiremos qual representa a incerteza
        # apropriada ao parâmetro, conforme Taylor.
        row["sd_f_peak_hz"] = sd
        row["sem_f_peak_hz"] = sem

        row["f_peak_min_hz"] = np.min(f)
        row["f_peak_max_hz"] = np.max(f)

        row["centroide_medio_hz"] = (
            g["f_centroid_hz"].mean()
        )

        row["f05_medio_hz"] = (
            g["f05_hz"].mean()
        )

        row["f95_medio_hz"] = (
            g["f95_hz"].mean()
        )

        rows.append(row)

    return pd.DataFrame(rows)


summary_sector = group_summary(
    df_points,
    ["evento", "setor"]
)

summary_event = group_summary(
    df_points,
    ["evento"]
)

summary_sector.to_csv(
    OUT_DIR / "resumo_por_setor.csv",
    index=False,
    sep=";",
    decimal=","
)

summary_event.to_csv(
    OUT_DIR / "resumo_por_evento.csv",
    index=False,
    sep=";",
    decimal=","
)


# ============================================================
# GRÁFICOS
# ============================================================

for (evento, setor), group in df_points.groupby(
    ["evento", "setor"]
):

    point_names = group["ponto"].tolist()

    spectra = []

    plt.figure(figsize=(10, 6))

    for point_name in point_names:

        s = spectra_by_point.get(point_name)

        if s is None:
            continue

        plt.plot(
            s["freq"],
            s["power"],
            alpha=0.6,
            label=point_name
        )

        spectra.append(s["power"])

    if len(spectra) == 0:
        plt.close()
        continue

    spectra = np.asarray(spectra)

    group_median = np.median(
        spectra,
        axis=0
    )

    group_median = (
        group_median
        / group_median.sum()
    )

    freq = spectra_by_point[point_names[0]]["freq"]

    group_peak = parabolic_peak_frequency(
        freq,
        group_median
    )

    plt.plot(
        freq,
        group_median,
        linewidth=3,
        label=f"Mediana do grupo ({group_peak:.2f} Hz)"
    )

    plt.axvline(
        group_peak,
        linestyle="--"
    )

    plt.xlim(0, 125)

    plt.xlabel("Frequência (Hz)")
    plt.ylabel("Potência espectral normalizada")

    plt.title(
        f"{evento} - {setor}\n"
        f"Janela = {WINDOW_MS:.0f} ms"
    )

    plt.grid(alpha=0.25)
    plt.legend()

    plt.tight_layout()

    filename = (
        f"{evento}_{setor}"
        .replace(" ", "_")
        .replace("/", "_")
        + ".png"
    )

    plt.savefig(
        PLOT_DIR / filename,
        dpi=200
    )

    plt.close()


# ============================================================
# RESUMO NA TELA
# ============================================================

print("\n" + "=" * 70)
print("RESULTADOS POR SETOR")
print("=" * 70)

if not summary_sector.empty:
    print(
        summary_sector[
            [
                "evento",
                "setor",
                "n_pontos",
                "f_peak_media_hz",
                "f_peak_mediana_hz",
                "sd_f_peak_hz",
                "sem_f_peak_hz",
                "centroide_medio_hz",
            ]
        ].to_string(index=False)
    )

print("\nArquivos salvos em:")
print(OUT_DIR.resolve())

print("\nFinalizado.")
