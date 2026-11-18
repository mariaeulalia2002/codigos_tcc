# Cálculo das incertezas morfométricas — TCC BS-500

## Executar

Requer Python 3.10 ou posterior. Não requer instalação de pacotes.

Nesta pasta, execute:

```bash
python incertezas_morfometricas.py --autoteste
```

O comando executa os testes internos, calcula as componentes e exporta os resultados. Também é possível abrir o arquivo principal no VS Code ou Spyder e executá-lo.

## O que calcula

- Repetibilidade de uma interpretação: s_d/√2, usando s_d=0,54 ms. Também aceita as diferenças individuais, quando disponíveis.
- Componente de interpolação: raiz da variância média de krigagem por superfície.
- Pesos IDW, velocidade ponderada e dispersão entre poços.
- Componentes temporais de espessura e soterramento e sua conversão para metros.
- Incerteza da integral temporal e do volume, mantendo a área fixa e adotando u_I=A×u_t,esp.
- Desníveis com separação dos percursos na água e nos sedimentos, alturas totais de queda e inclinações, mantendo distância horizontal e velocidade da água fixas.
- Percentuais de incerteza antes do arredondamento.

São 70 resultados de espessura, soterramento e volume para os dez eventos, além de cinco alturas e sete inclinações. As funções `resolucao_temporal` e `sensibilidade_grade` também estão disponíveis para as avaliações mantidas separadas da propagação.

## Referências

Taylor (1997) fundamenta a propagação; Isaaks & Srivastava (1989), a variância de estimação da krigagem; Shepard (1968), o princípio de ponderação por distância; Illowsky & Dean (2023), a variância de uma distribuição discreta; JCGM 100:2008, a expressão de resultados e suas incertezas. As regras operacionais específicas são as descritas no TCC.

## Verificação

O argumento `--autoteste` verifica casos analíticos, unidades, cancelamento do termo comum de velocidade, inversão da ordem dos extremos, validação de entradas e derivadas de desnível e ângulo por diferenças finitas.
