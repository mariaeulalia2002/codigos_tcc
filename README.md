# Códigos do TCC

Este diretório reúne as rotinas complementares utilizadas na verificação dos
dados sísmicos, na preparação do modelo de velocidades e na identificação de
picos secundários das curvas HVSR.

Os arquivos sísmicos, registros de poços, arquivos DLIS, curvas HVSR e demais
dados de entrada não são distribuídos neste repositório. Os caminhos devem ser
informados no momento da execução de cada programa.

## Processamento principal do HVSR

O cálculo principal das curvas HVSR, a segmentação das séries temporais, a
combinação das componentes horizontais, as suavizações espectrais, a estatística
lognormal e a determinação de `f0` foram realizados com os notebooks do
[hvsrweb](https://github.com/jpvantassel/hvsrweb) e com a biblioteca
[hvsrpy](https://github.com/jpvantassel/hvsrpy), desenvolvidos por Vantassel e
colaboradores. Esses códigos não são redistribuídos aqui.

O arquivo `identificacao_picos_secundarios.py` é uma rotina complementar criada
para este trabalho. Ele recebe uma curva já processada pelo hvsrweb/hvsrpy e o
valor de `f0`, identificando somente possíveis picos secundários `f1` e `f2`.

## Arquivos

### `verificacao_headers_segy.py`

Compara o intervalo amostral do *binary header* com o intervalo registrado nos
*trace headers* de uma quantidade definida de traços do arquivo SEG-Y.

```powershell
python verificacao_headers_segy.py levantamento.sgy --tracos 200
```

### `analise_Espectral.py`

Estima a frequencia dominante do sinal sismico nos pontos de controle usados
no TCC. Para cada ponto, extrai uma vizinhanca de 3 x 3 tracos, remove a
tendencia linear, aplica uma janela de Hann e calcula o espectro de potencia.
O pico e refinado por interpolacao parabolica local em escala logaritmica. O
programa salva os resultados por traco, ponto, setor e evento, bem como os
graficos dos espectros. Os pontos e tempos de referencia permanecem registrados
no codigo, enquanto o caminho do SEG-Y e o diretorio de saida sao fornecidos na
execucao.

```powershell
python analise_Espectral.py levantamento.sgy `
  --saida resultados/frequencias `
  --janela-ms 256
```

### `extracao_dados_vsp_poco_137.py`

Lê o DLIS do poço 1-BRSA-137-RJS, extrai os canais de profundidade e tempo e
calcula as velocidades intervalares entre registros consecutivos. O resultado é
exportado em CSV e constitui a entrada da etapa de suavização.

```powershell
python extracao_dados_vsp_poco_137.py poco_137.dlis `
  --saida resultados/velocidades_originais_poco_137.csv
```

### `suavizacao_poco_137.py`

Calcula a média móvel simples, o filtro Savitzky–Golay e a média móvel ponderada
da curva de velocidades do poço 137. A média móvel ponderada foi adotada na
conversão morfométrica dos MTDs/MTCs, enquanto os valores rasos do filtro
Savitzky–Golay foram usados na análise HVSR.

```powershell
python suavizacao_poco_137.py resultados/velocidades_originais_poco_137.csv `
  --saida resultados/velocidades_suavizadas_poco_137.csv `
  --figura resultados/curvas_velocidade_poco_137.png
```

### `conversao_tempo_profundidade.py`

Reproduz as velocidades equivalentes dos intervalos dos MTDs/MTCs nos poços 137
e 127 e calcula a ponderação pelo inverso da distância. Os controles utilizados
estão documentados nas Tabelas D.1 e D.2 do TCC.

```powershell
python conversao_tempo_profundidade.py --saida resultados
```

As distâncias incorporadas ao código correspondem aos valores apresentados no
Apêndice D. Como esses valores estão arredondados, diferenças residuais de
0,01 m/s ou 0,01 ponto percentual podem ocorrer em relação aos cálculos
originais, realizados antes do arredondamento das distâncias.

### `identificacao_picos_secundarios.py`

Agrupa a curva em intervalos de frequência e utiliza `scipy.signal.find_peaks`
para identificar máximos secundários. Os valores padrão reproduzem os critérios
do TCC: H/V mínimo de 1,5, proeminência mínima de 0,4 e separação relativa de
15%.

```powershell
python identificacao_picos_secundarios.py curva_hvsr.csv --f0 0.857 `
  --saida resultados/picos_secundarios.csv `
  --figura resultados/picos_secundarios.png
```

## Instalação

Recomenda-se Python 3.10 ou posterior.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Observação sobre os dados

Os dados de entrada devem ser obtidos junto às respectivas instituições e
utilizados de acordo com suas condições de acesso. A ausência desses arquivos
no repositório não impede a inspeção dos procedimentos e dos parâmetros
adotados nos códigos.
