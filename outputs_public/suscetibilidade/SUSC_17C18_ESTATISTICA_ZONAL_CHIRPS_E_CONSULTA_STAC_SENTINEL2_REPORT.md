# SUSC-17C18 - Estatistica zonal CHIRPS e consulta STAC Sentinel-2 por AOI

## Objetivo
O 17C18 executa a primeira tentativa de extracao sensorial leve por AOI candidata para os 5 patches de Recife (evento REC_2022_05_24_30, periodo 2022-05-24/2022-05-30). CHIRPS e consultado como fonte de chuva antecedente e Sentinel-2/CDSE e consultado por AOI e janela temporal para inventario de cenas candidatas. O build publico e offline e deterministico; a consulta real de rede exige `SUSC_17C18_ALLOW_NETWORK=1`.

## Separacao conceitual
CHIRPS e Sentinel-2 sao camadas sensoriais/contextuais. Podem gerar feature de chuva antecedente (se houver dado verificavel), inventario de cenas e metadado de fonte. Nao geram evento observado, Ground Reference Candidate, ground truth, label, validacao 17B nem score v7.

## CHIRPS
- Fonte CHIRPS resolvida para acesso leve por AOI: False (publico disponivel apenas como raster global pesado ou colecao GEE com runtime).
- Artefatos CHIRPS leves obtidos: 0.
- Estatisticas zonais CHIRPS reais calculadas: 0.
- Features reais de chuva antecedente criadas: 0.

## Sentinel-2 / CDSE
A raiz STAC publica do CDSE nao expoe a colecao Sentinel-2; a consulta espaco-temporal de cenas usa o catalogo OData oficial do CDSE filtrando por AOI e janela temporal. As cenas retornadas sao metadado leve, nunca produto baixado.
- Consultas por AOI tentadas: 5.
- Consultas bem sucedidas: 5.
- Cenas Sentinel-2 candidatas registradas: 100.
- Cenas pre-evento: 60.
- Cenas pos-evento: 30.
- Produtos Sentinel-2 baixados: 0.
- Tiles criados: 0.
- Embeddings criados: 0.

## Gates
Feature CHIRPS e metadado de cena Sentinel-2 podem passar gates de fonte/tempo/proveniencia como camada sensorial candidata. Nenhum passa G4 (vinculo espacial de evento) ou G5 (separacao de fenomeno) como evento observado; nenhum vira Ground Reference Candidate.

## Guardrails
Nenhum produto Sentinel-2 foi baixado, nenhum tile foi criado, nenhum embedding foi calculado, nenhuma feature pos-evento virou pre-evento, o score v6 permanece intacto, o score v7 nao foi criado e o 17B continua bloqueado.

## Proximo marco recomendado
SUSC-17C19 Politica de tile leve Sentinel-2 e runtime CHIRPS por AOI para primeira feature sensorial real reproduzivel
