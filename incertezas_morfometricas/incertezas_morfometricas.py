#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Componentes e propagação das incertezas morfométricas — BS-500.

Python >= 3.10, somente biblioteca padrão. Executar:
    python incertezas_morfometricas.py
    python incertezas_morfometricas.py --saida resultados --autoteste

Implementa as equações 20–30 da Seção 4.3.9 do TCC. Unidades de entrada:
TWT em ms, K em ms², velocidades em m/s, área em km², integral em ms.m²,
distâncias em m. Saídas: m, graus e km³. Não incorpora resolução sísmica
ou sensibilidade da grade à propagação; não estima incerteza de área,
perímetro, comprimento nem das medidas de cicatrizes Hs e Ss.

As funções são independentes das entradas do caso de estudo. Os dados
incorporados correspondem às medidas e aos parâmetros consolidados nas
tabelas e nos apêndices do TCC. Os cálculos são realizados antes do
arredondamento adotado para a apresentação dos resultados.

Referências metodológicas: Taylor (1997), An Introduction to Error Analysis,
2nd ed.; Isaaks & Srivastava (1989), An Introduction to Applied Geostatistics;
Shepard (1968), A two-dimensional interpolation function for irregularly-spaced
data; Illowsky & Dean (2023), Introductory Statistics 2e; JCGM 100:2008.
As escolhas de componentes representativas, expoente IDW=1 e uI=A.ut são
as escolhas operacionais descritas no TCC, não prescrições desses autores.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev

SD_DIFERENCAS_MS = 0.54
RMSE_FUNDO_MS = 0.09826
VELOCIDADE_AGUA_M_S = 1500.0


def nao_negativo(nome: str, valor: float) -> float:
    valor = float(valor)
    if not math.isfinite(valor) or valor < 0:
        raise ValueError(f"{nome}: esperado número finito não negativo; recebido {valor}")
    return valor


def positivo(nome: str, valor: float) -> float:
    valor = nao_negativo(nome, valor)
    if valor == 0:
        raise ValueError(f"{nome}: esperado valor positivo")
    return valor


def combinacao_quadratica(*componentes: float) -> float:
    """Eq. 23, após multiplicar cada u pela respectiva derivada parcial."""
    return math.sqrt(math.fsum(nao_negativo("componente", x)**2 for x in componentes))


def propagacao_primeira_ordem(derivadas: list[float], incertezas: list[float]) -> float:
    """Eq. 23; entradas independentes, sem covariâncias."""
    if len(derivadas) != len(incertezas) or not derivadas:
        raise ValueError("Forneça uma derivada para cada componente de incerteza")
    if not all(math.isfinite(x) for x in derivadas):
        raise ValueError("Derivadas não finitas")
    return combinacao_quadratica(*(abs(d)*nao_negativo("u", u)
                                  for d, u in zip(derivadas, incertezas)))


def repetibilidade(sd_diferencas_ms: float = SD_DIFERENCAS_MS,
                   diferencas_ms: list[float] | None = None) -> dict:
    """Eq. 20. Se houver leituras individuais, usa desvio amostral (ddof=1).

    Na execução padrão, utiliza s_d=0,54 ms, conforme a metodologia.
    As diferenças individuais podem ser fornecidas opcionalmente.
    """
    media = None
    n = None
    if diferencas_ms is not None:
        if len(diferencas_ms) < 2 or not all(math.isfinite(x) for x in diferencas_ms):
            raise ValueError("São necessárias pelo menos duas diferenças finitas")
        sd_diferencas_ms = stdev(diferencas_ms)
        media, n = mean(diferencas_ms), len(diferencas_ms)
    sd = nao_negativo("s_d", sd_diferencas_ms)
    return dict(s_d_ms=sd, u_pick_ms=sd/math.sqrt(2),
                media_diferencas_ms=media, n_diferencas_fornecidas=n)


def componente_interpolacao(variancia_media_ms2: float) -> float:
    """Eq. 21: raiz da média das variâncias, não média das raízes."""
    return math.sqrt(nao_negativo("K", variancia_media_ms2))


def velocidade_idw(velocidades_m_s: list[float], distancias_m: list[float],
                   expoente: float = 1.0) -> dict:
    """Eqs. 16 e 22: média IDW e dispersão ponderada, sem divisão por sqrt(N)."""
    if not velocidades_m_s or len(velocidades_m_s) != len(distancias_m):
        raise ValueError("Forneça uma distância por velocidade")
    vv = [positivo("velocidade", v) for v in velocidades_m_s]
    dd = [positivo("distância", d) for d in distancias_m]
    positivo("expoente IDW", expoente)
    pesos_brutos = [d**(-expoente) for d in dd]
    soma = math.fsum(pesos_brutos)
    pesos = [p/soma for p in pesos_brutos]
    v = math.fsum(p*x for p, x in zip(pesos, vv))
    uv = math.sqrt(math.fsum(p*(x-v)**2 for p, x in zip(pesos, vv)))
    return dict(v_idw_m_s=v, u_v_idw_m_s=uv, pesos=pesos,
                u_v_relativa_pct=100*uv/v)


def componentes_temporais(k_topo_ms2: float, k_base_ms2: float,
                          u_pick_ms: float, u_fundo_ms: float) -> dict:
    """Eqs. 21, 24 e 25; componentes representativas por superfície."""
    kt = nao_negativo("K_topo", k_topo_ms2)
    kb = nao_negativo("K_base", k_base_ms2)
    up = nao_negativo("u_pick", u_pick_ms)
    uf = nao_negativo("u_fundo", u_fundo_ms)
    return dict(u_interp_topo_ms=math.sqrt(kt), u_interp_base_ms=math.sqrt(kb),
                u_pick_diferenca_ms=math.sqrt(2)*up,
                u_krig_esp_ms=math.sqrt(kt+kb),
                u_t_esp_ms=math.sqrt(2*up**2+kt+kb),
                u_t_sot_ms=math.sqrt(up**2+kt+uf**2))


def medida_convertida(t_ms: float, u_t_ms: float, v_m_s: float,
                       u_v_m_s: float) -> dict:
    """Eq. 26. Incerteza relativa usa o valor não arredondado."""
    t = nao_negativo("intervalo TWT", t_ms)
    v = positivo("v", v_m_s)
    ut = nao_negativo("u_t", u_t_ms)
    uv = nao_negativo("u_v", u_v_m_s)
    valor = t*v/2000
    ctempo, cvel = v*ut/2000, t*uv/2000
    u = combinacao_quadratica(ctempo, cvel)
    return dict(t_ms=t, u_t_ms=ut, valor=valor, unidade="m",
                u_tempo=ctempo, u_velocidade=cvel, u=u,
                u_relativa_pct=100*u/abs(valor) if valor else None)


def volume_convertido(integral_ms_m2: float, area_km2: float, u_t_esp_ms: float,
                      v_m_s: float, u_v_m_s: float) -> dict:
    """Eqs. 27–28. Área fixa; I não é substituída por área × espessura média."""
    integral = nao_negativo("integral temporal", integral_ms_m2)
    area = positivo("área", area_km2)*1e6
    v = positivo("v", v_m_s)
    ui = area*nao_negativo("u_t_esp", u_t_esp_ms)
    uv = nao_negativo("u_v", u_v_m_s)
    valor = integral*v/2e12
    ctempo, cvel = v*ui/2e12, integral*uv/2e12
    u = combinacao_quadratica(ctempo, cvel)
    return dict(integral_ms_m2=integral, area_m2=area, u_I_ms_m2=ui,
                valor=valor, unidade="km3", u_tempo=ctempo, u_velocidade=cvel,
                u=u, u_relativa_pct=100*u/abs(valor) if valor else None)


def desnivel_e_inclinacao(delta_horizonte_ms: float, delta_fundo_ms: float,
                         v_m_s: float, u_v_m_s: float,
                         distancia_m: float | None = None,
                         u_pick_ms: float | None = None,
                         u_fundo_ms: float = RMSE_FUNDO_MS) -> dict:
    """Eqs. 19, 29–30; uma MESMA velocidade sedimentar em ambos os extremos.

    Deltas na ordem final−inicial. O desnível mantém sinal; altura e ângulo
    são apresentados em módulo. L e velocidade da água são fixos.
    """
    if not all(math.isfinite(x) for x in (delta_horizonte_ms, delta_fundo_ms)):
        raise ValueError("Diferenças de TWT não finitas")
    v = positivo("v", v_m_s)
    uv = nao_negativo("u_v", u_v_m_s)
    up = repetibilidade()["u_pick_ms"] if u_pick_ms is None else nao_negativo("u_pick", u_pick_ms)
    uf = nao_negativo("u_fundo", u_fundo_ms)
    uth, utf = math.sqrt(2)*up, math.sqrt(2)*uf
    dh, df = delta_horizonte_ms, delta_fundo_ms
    dz = (v*dh+(VELOCIDADE_AGUA_M_S-v)*df)/2000
    ch = abs(v*uth/2000)
    cf = abs((VELOCIDADE_AGUA_M_S-v)*utf/2000)
    cv = abs((dh-df)*uv/2000)
    uz = combinacao_quadratica(ch, cf, cv)
    resultado = dict(delta_horizonte_ms=dh, delta_fundo_ms=df,
                     u_delta_horizonte_ms=uth, u_delta_fundo_ms=utf,
                     delta_z_assinado_m=dz, altura_m=abs(dz),
                     u_interpretacao_m=ch, u_fundo_m=cf, u_velocidade_m=cv,
                     u_altura_m=uz,
                     u_altura_relativa_pct=100*uz/abs(dz) if dz else None)
    if distancia_m is not None:
        comprimento = positivo("L", distancia_m)
        angulo = math.degrees(math.atan(abs(dz)/comprimento))
        fator = 180/math.pi/(comprimento*(1+(dz/comprimento)**2))
        resultado.update(distancia_m=comprimento, inclinacao_graus=angulo,
                         u_inclinacao_graus=fator*uz,
                         u_inclinacao_relativa_pct=100*fator*uz/angulo if angulo else None)
    return resultado


def resolucao_temporal(frequencia_dominante_hz: float) -> float:
    """Eq. 31, avaliação separada: não é parcela da incerteza propagada."""
    return 500/positivo("f_dom", frequencia_dominante_hz)


def sensibilidade_grade(referencia: float, valores_testados: list[float]) -> dict:
    """Seção 4.3.9.4: comparação separada, sem somar à propagação."""
    if not valores_testados or not all(math.isfinite(x) for x in [referencia, *valores_testados]):
        raise ValueError("Valores de grade ausentes ou não finitos")
    diferencas = [abs(x-referencia) for x in valores_testados]
    maior = max(diferencas)
    return dict(diferencas_absolutas=diferencas, maior_diferenca=maior,
                maior_diferenca_pct=100*maior/abs(referencia) if referencia else None)


# DADOS DO CASO DE ESTUDO
# Ordem temporal de cada lista: mínimo, médio, máximo.
# As velocidades e suas dispersões correspondem aos valores consolidados nos
# Apêndices D e E. A rotina também calcula os valores IDW a partir dos controles
# dos poços, permitindo conferir os pesos e a ponderação pela distância.
# Para MTC 1-B e MTC 2-D, as componentes temporais são informadas diretamente,
# conforme os valores adotados nas respectivas tabelas de resultados.
EVENTOS = [
    dict(evento="MTD 1", tabela=8, v=1561.76, uv=13.57,
         v_pocos=[1567.28,1528.37], d_m=[48808.5,295520.8],
         k=[.72,1.13], area=77.12, integral=4476000000,
         esp=[5.030,57.420,157.165], sot=[17.230,64.332,70.934],
         fonte_tempos="Memória de cálculo, Santos, p.7; Tabela 8",
         fonte_k="Memória de cálculo, Santos, p.7",
         fonte_integral="Memória de cálculo, Santos, p.7"),
    dict(evento="MTD 2", tabela=11, v=1554.90, uv=18.12,
         v_pocos=[1567.28,1528.37], d_m=[32298.8,69222.8],
         k=[.18,.29], area=11.72, integral=571504000,
         esp=[20.490,48.440,89.960], sot=[30.779,56.275,67.967],
         fonte_tempos="Memória de cálculo, Continuação Santos, p.6; Tabela 11",
         fonte_k="Memória de cálculo, Continuação Santos, p.6",
         fonte_integral="Memória de cálculo, Continuação Santos, p.6"),
    dict(evento="MTD 3", tabela=14, v=1715.30, uv=32.90,
         v_pocos=[1646.21,1730.96], d_m=[44079.5,9994.8],
         k=[1.69,2.48], area=115.96, integral=8622130000,
         esp=[17.940,73.950,189.890], sot=[39.759,265.065,326.767],
         fonte_tempos="Memória de cálculo, Recorrência, p.16; Tabela 14",
         fonte_k="Memória de cálculo, Recorrência, p.16",
         fonte_integral="Memória de cálculo, Recorrência, p.16"),
    dict(evento="MTC 1-A", tabela=16, v=1709.91, uv=53.82,
         v_pocos=[1656.99,1764.65], d_m=[38481.359,39807.014],
         k=[21.09,22.97], area=1585.48, integral=153900000000,
         esp=[28.88,97.07,242.66], sot=[31.1,263.533,510.7],
         fonte_tempos="Tabela 16 e Apêndice D",
         fonte_k="Variâncias médias de krigagem: topo 21,09; base 22,97",
         fonte_integral="Tabela 16: 153,900 x 10^9 ms.m²"),
    dict(evento="MTC 1-B", tabela=17, v=1774.62, uv=78.72,
         v_pocos=[1723.92,1896.85], d_m=[20104.965,48471.336],
         k=None, ut_esp=1.2, ut_sot=.85, area=110.20, integral=11400000000,
         esp=[19.0,102.76,154.7],
         sot=[233.1089630126953,322.1647500624132,389.1745910644531],
         fonte_tempos="Tabela 17 e Apêndice D",
         fonte_k="Componentes temporais adotadas na Tabela 17",
         fonte_integral="Tabela 17: 11,40 x 10^9 ms.m²"),
    dict(evento="MTC 1-C", tabela=18, v=1552.59, uv=18.86,
         v_pocos=[1567.28,1528.37], d_m=[31032.2,51170],
         k=[8.21,9.56], area=648.50, integral=65500000000,
         esp=[10.93,100.10,230.61],
         sot=[4.703070640563965,111.69389901398249,203.43434143066406],
         fonte_tempos="Tabela 18 e Apêndice D",
         fonte_k="Variâncias médias de krigagem: topo 8,21; base 9,56",
         fonte_integral="Tabela 18: 65,5 x 10^9 (entrada arredondada)"),
    dict(evento="MTC 2-A", tabela=20, v=1785.72, uv=96.43,
         v_pocos=[1702.04,1896.85], d_m=[21018.562,27909.689],
         k=[.30,.45], area=73.02, integral=4320270000,
         esp=[22.370,58.595,103.340], sot=[104.664,324.596,400.276],
         fonte_tempos="Memória de cálculo, Búzios 3rd, p.10; Tabela 20",
         fonte_k="Memória de cálculo, Búzios 3rd, p.10",
         fonte_integral="Memória de cálculo, Búzios 3rd, p.10"),
    dict(evento="MTC 2-B", tabela=21, v=1574.50, uv=30.09,
         v_pocos=[1600.62,1539.83], d_m=[21018.562,27909.689],
         k=[.46,.79], area=95.64, integral=8400650000,
         esp=[16.570,86.961,144.440], sot=[22.103,214.101,279.694],
         fonte_tempos="Memória de cálculo, Búzios 4th, p.11; Tabela 21",
         fonte_k="Memória de cálculo, Búzios 4th, p.11",
         fonte_integral="Memória de cálculo, Búzios 4th, p.11"),
    dict(evento="MTC 2-C", tabela=22, v=1574.15, uv=17.69,
         v_pocos=[1580.98,1528.37], d_m=[6004.060,40198.169],
         k=[1.53,2.02], area=42.29, integral=3106000000,
         esp=[17.2,72.91,175.9],
         sot=[130.06224060058594,197.2467784127075,308.3714599609375],
         fonte_tempos="Tabela 22 e Apêndice D",
         fonte_k="Variâncias médias de krigagem: topo 1,53; base 2,02",
         fonte_integral="Tabela 22: 3,106 x 10^9 ms.m²"),
    dict(evento="MTC 2-D", tabela=24, v=1561.18, uv=14.15,
         v_pocos=[1567.28,1528.37], d_m=[6704.8,36022.4],
         k=None, ut_esp=2.7, ut_sot=2.2, area=232.73, integral=25150000000,
         esp=[5.78,107.23,290.87], sot=[4.67,119.202,185.52],
         fonte_tempos="Tabela 24 e Apêndice D",
         fonte_k="Componentes temporais adotadas na Tabela 24",
         fonte_integral="Tabela 24: 25,15 x 10^9 (entrada arredondada)"),
]

# Entradas do Apêndice E. Não confundir L do segmento com comprimento do eixo.
SEGMENTOS = [
    ("MTD 1", "Ht", "A-C", 1872.44,1819.73,2190.13,2079.84,None),
    ("MTD 2", "Ht", "A-C", 1979.96,1938.34,2147.95,2059.03,None),
    ("MTD 3", "Ht", "A-C", 2358.80,2139.12,2897.46,2561.82,None),
    ("MTC 2-D", "Ht", "A-C1", 2179.34,2126.42,2926.58,2808.03,None),
    ("MTC 2-D", "Ht", "A-C2", 2179.34,2126.42,2774.16,2556.13,None),
    ("MTD 1", "S", "P1-P2",2116.71,2032.27,2199.62,2083.20,5877),
    ("MTD 2", "S", "P1-P2",2003.88,1910.67,2224.98,2127.43,2581),
    ("MTD 3", "S", "P1-P2",2367.66,2096.71,2738.26,2455.11,13740),
    ("MTC 1-C", "S", "P1-P2",1194.48,1125.68,2269.85,2047.60,33459),
    ("MTD 1", "St", "P3-P4",2183.64,2070.47,2188.42,2078.42,966),
    ("MTD 2", "St", "P3-P4",2140.21,2056.21,2133.18,2059.09,1093),
    ("MTC 1-C", "St", "P3-P4",2236.91,1978.65,2249.61,2029.46,3197),
]


def gravar_csv(caminho: Path, linhas: list[dict]) -> None:
    campos = list(dict.fromkeys(chave for linha in linhas for chave in linha))
    with caminho.open("w", encoding="utf-8-sig", newline="") as arquivo:
        escritor = csv.DictWriter(arquivo, fieldnames=campos, delimiter=";")
        escritor.writeheader()
        escritor.writerows(linhas)


def executar(saida: Path) -> dict:
    saida.mkdir(parents=True, exist_ok=True)
    pick = repetibilidade()
    up, uf = pick["u_pick_ms"], RMSE_FUNDO_MS
    componentes, medidas, segmentos = [], [], []
    avisos = []
    for e in EVENTOS:
        idw = velocidade_idw(e["v_pocos"], e["d_m"])
        k = e["k"]
        if k is not None:
            temporal = componentes_temporais(*k, up, uf)
            modo = "componentes temporais calculadas a partir de K"
        else:
            temporal = dict(u_t_esp_ms=e["ut_esp"], u_t_sot_ms=e["ut_sot"])
            modo = "componentes temporais informadas diretamente"
            avisos.append(f"{e['evento']}: componentes temporais fornecidas diretamente; "
                          "decomposição por superfície não aplicada.")
        componentes.append(dict(evento=e["evento"], modo_temporal=modo,
            k_topo_ms2=k[0] if k else None, k_base_ms2=k[1] if k else None,
            u_pick_ms=up, u_fundo_ms=uf, **temporal,
            v_adotada_m_s=e["v"], u_v_adotada_m_s=e["uv"],
            v_idw_recalculada_m_s=idw["v_idw_m_s"],
            u_v_idw_recalculada_m_s=idw["u_v_idw_m_s"],
            peso_137=idw["pesos"][0], peso_127=idw["pesos"][1],
            delta_v_idw_m_s=idw["v_idw_m_s"]-e["v"],
            delta_uv_idw_m_s=idw["u_v_idw_m_s"]-e["uv"],
            fonte_k=e["fonte_k"]))
        for grupo, nome, ut in [("esp", "espessura", temporal["u_t_esp_ms"]),
                                ("sot", "soterramento", temporal["u_t_sot_ms"])]:
            for extremo, t in zip(["minimo", "medio", "maximo"], e[grupo]):
                r = medida_convertida(t, ut, e["v"], e["uv"])
                r.update(evento=e["evento"], parametro=f"{nome}_{extremo}",
                         tabela=e["tabela"], modo_temporal=modo)
                if k is not None:
                    r.update(u_interpretacao=e["v"]*(math.sqrt(2)*up if grupo=="esp" else up)/2000,
                        u_interpolacao=e["v"]*math.sqrt(sum(k) if grupo=="esp" else k[0])/2000,
                        u_fundo=0.0 if grupo=="esp" else e["v"]*uf/2000)
                else:
                    r.update(u_interpretacao=None, u_interpolacao=None, u_fundo=None)
                medidas.append(r)
        volume = volume_convertido(e["integral"],e["area"],temporal["u_t_esp_ms"],e["v"],e["uv"])
        volume.update(evento=e["evento"], parametro="volume", tabela=e["tabela"], modo_temporal=modo)
        if k is not None:
            fator = e["v"]*e["area"]*1e6/2e12
            volume.update(u_interpretacao=fator*math.sqrt(2)*up,
                          u_interpolacao=fator*math.sqrt(sum(k)), u_fundo=0.0)
        medidas.append(volume)
    por_evento = {e["evento"]:e for e in EVENTOS}
    for evento,tipo,nome,th1,tf1,th2,tf2,l in SEGMENTOS:
        e = por_evento[evento]
        r = desnivel_e_inclinacao(th2-th1,tf2-tf1,e["v"],e["uv"],l,up,uf)
        r.update(evento=evento, medida=tipo, segmento=nome,
                 fonte="Apêndice E, Tabelas E.3–E.6")
        segmentos.append(r)
    gravar_csv(saida/"componentes.csv",componentes)
    gravar_csv(saida/"espessuras_soterramentos_volumes.csv",medidas)
    gravar_csv(saida/"alturas_inclinacoes.csv",segmentos)
    dados = dict(repetibilidade=pick, rmse_fundo_ms=uf,
                 velocidade_agua_m_s=VELOCIDADE_AGUA_M_S, eventos=EVENTOS,
                 segmentos=SEGMENTOS)
    texto = json.dumps(dados,ensure_ascii=False,indent=2,allow_nan=False)
    (saida/"entradas_utilizadas.json").write_text(texto,encoding="utf-8")
    resumo = dict(n_eventos=len(EVENTOS), n_medidas=len(medidas), n_segmentos=len(segmentos),
                  sha256_entradas=hashlib.sha256(texto.encode()).hexdigest(),
                  avisos=avisos,
                  nota="Cálculos realizados com as entradas numéricas incorporadas; "
                       "as tabelas do TCC apresentam valores arredondados.")
    (saida/"execucao.json").write_text(json.dumps(resumo,ensure_ascii=False,indent=2),encoding="utf-8")
    return resumo


def autoteste() -> None:
    """Verificações com resultados analíticos e derivadas numéricas independentes."""
    def perto(a,b,tol=1e-9):
        if not math.isclose(a,b,rel_tol=tol,abs_tol=tol):
            raise AssertionError((a,b))
    perto(repetibilidade()["u_pick_ms"],.38183766184073564)
    perto(velocidade_idw([1500,1700],[10,10])["u_v_idw_m_s"],100)
    perto(velocidade_idw([1500,1700],[10,30])["v_idw_m_s"],1550)
    perto(componentes_temporais(.72,1.13,.54/math.sqrt(2),.09826)["u_t_esp_ms"],
          math.sqrt(2.1416))
    perto(medida_convertida(100,2,1600,20)["u"],math.hypot(1.6,1.0))
    perto(volume_convertido(1e9,10,2,1600,20)["u"],math.hypot(.016,.01))
    # Analítico e derivadas centrais por diferenças finitas para dz e theta.
    dh,df,v=221.10,216.76,1554.9
    x=[dh,df,v]
    us=[.54,math.sqrt(2)*.09826,18.12]
    def dz(a): return (a[2]*a[0]+(1500-a[2])*a[1])/2000
    def ang(a): return math.degrees(math.atan(abs(dz(a))/2581))
    for func,campo in [(dz,"u_altura_m"),(ang,"u_inclinacao_graus")]:
        derivadas=[]
        for i in range(3):
            a,b=x.copy(),x.copy()
            passo=1e-4
            a[i]+=passo; b[i]-=passo
            derivadas.append((func(a)-func(b))/(2*passo))
        r=desnivel_e_inclinacao(dh,df,v,18.12,2581)
        perto(propagacao_primeira_ordem(derivadas,us),r[campo],1e-7)
    # A mesma velocidade e o mesmo fundo em ambos os extremos: invariância
    # à troca dos extremos e cancelamento da parcela de uv quando dh=df.
    r=desnivel_e_inclinacao(20,20,1600,50,1000)
    perto(r["u_velocidade_m"],0)
    r2=desnivel_e_inclinacao(-20,-20,1600,50,1000)
    perto(r["u_altura_m"],r2["u_altura_m"])
    perto(r["altura_m"],15)
    perto(resolucao_temporal(25),20)
    perto(sensibilidade_grade(100,[98,103])["maior_diferenca_pct"],3)
    for func in [lambda:componente_interpolacao(-1),
                 lambda:velocidade_idw([1500],[0]),
                 lambda:desnivel_e_inclinacao(1,1,1500,0,0)]:
        try: func()
        except ValueError: pass
        else: raise AssertionError("Entrada inválida foi aceita")


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saida",type=Path,default=Path(__file__).resolve().parent/"resultados")
    parser.add_argument("--autoteste",action="store_true")
    args=parser.parse_args()
    if args.autoteste:
        autoteste()
        print("Autotestes analíticos, de derivadas e de validação: OK")
    resumo=executar(args.saida)
    print(f"Calculados {resumo['n_medidas']} resultados morfométricos e {resumo['n_segmentos']} segmentos.")
    print(f"Resultados: {args.saida.resolve()}")
    for aviso in resumo["avisos"]:
        print("NOTA:",aviso)
