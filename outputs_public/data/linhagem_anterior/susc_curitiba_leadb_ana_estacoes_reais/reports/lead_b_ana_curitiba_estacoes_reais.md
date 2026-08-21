# Lead B (Curitiba) -- Estações Fluviométricas/Cotagrama Reais da ANA

**Status**: concluído (endpoint real consultado, 63 estações reais baixadas e analisadas,
corroboração hidrológica real encontrada para o evento de 2022-01-15/16).

Reproduz para Curitiba o mesmo método real já usado e documentado no Lead B de Recife
(`lead_b_ana_estacoes_reais.md`, SUSC-20A), com rede habilitada nesta sessão (diferente da
tentativa offline-first anterior registrada em
`SUSC_18C_AQUISICAO_GEOMETRIA_OFICIAL_CURITIBA.md`).

## Endpoint e método (idêntico ao Lead B Recife)

`https://www.snirh.gov.br/arcgis/rest/services/Hidroweb_BH/INVENTARIOS_ESTACOES/MapServer/0`
(bbox metropolitana `-49.45,-25.65,-49.10,-25.30`) + endpoint legado
`telemetriaws1.ana.gov.br/ServiceANA.asmx/HidroSerieHistorica` para séries reais.

**Diferença metodológica registrada**: para Recife, cada estação foi checada só num
`tipoDados` (vazão ou cota, conforme já se sabia qual tinha dado real). Para Curitiba,
como não havia conhecimento prévio, os dois tipos (`tipoDados=3` vazão, `tipoDados=1`
cota) foram baixados para cada estação e o melhor (mais dias com dado) foi mantido --
achado real: **cota tem cobertura muito mais rica que vazão** para quase todas as
estações de Curitiba (ex: 65013005 Rio Iguaçu, cota 1984-2012 n=28773 dias vs vazão
que parava em 2010 com muito menos densidade).

## Inventário

104 estações fluviométricas na bbox metropolitana; 63 com `OPERANDO=Sim` e
`POSSUI_DADOS=Sim`. 23 estão no município de Curitiba propriamente dito.

## Cobertura do evento-alvo (2022-01-15/16, S17C_REF_0060, já documentado com fonte
administrativa em `SUSC_18C_AQUISICAO_GEOMETRIA_OFICIAL_CURITIBA.md`)

**9 estações** têm série de cota real cobrindo as datas exatas do evento:

| Código | Rio | Município | p95 série completa | Valor 15/01 | Valor 16/01 | Acima do p95? |
|---|---|---|---|---|---|---|
| 65010000 | Rio Pequeno | São José dos Pinhais | 203 cm | 167 | **205** | **SIM** |
| 65015400 | Rio Miringuava | São José dos Pinhais | 211 cm | 200 | **220** | **SIM** |
| 65019675 | Rio Barigui | **Curitiba** | 140 cm | 28 | 105 | Não |
| 65004995 | Rio Piraquara | Piraquara | 214 cm | 144 | 182 | Não |
| 65021770 | Rio Cachoeirinha | Almirante Tamandaré | 147 cm | 108 | 112 | Não |
| 65019980 | Rio Iguaçu | Araucária | 285 cm | 164 | 250 | Não |
| 65021800 | Rio Passaúna | Campo Largo | 166 cm | 54 | 78 | Não |
| 65020995 | Rio Passaúna | Almirante Tamandaré | 88 cm | 62 | 60 | Não |
| 65019700 | Rio Barigui | Araucária | 80 cm | 32 | 70 | Não |

## Achado honesto

- **Corroboração hidrológica real existe**, mas em 2 estações da região metropolitana
  (São José dos Pinhais, bacia Iguaçu-sul), não dentro do município de Curitiba
  propriamente dito -- mesmo padrão já usado no Lead B de Recife, onde a corroboração
  mais forte também veio de estações fora do município-sede (São Lourenço da Mata).
- A **única estação real dentro do município de Curitiba** que cobre a data exata do
  evento (65019675, Rio Barigui, cota 2003-2024, n=19150 dias) **não** mostrou nível
  acima do seu próprio p95 nos dois dias do evento (28cm e 105cm vs p95=140cm) -- um
  resultado real e não-corroborante, reportado sem omissão. Isso é consistente com o
  evento de jan/2022 ter sido predominantemente um evento de chuva intensa/alagamento
  urbano pontual (drenagem, não necessariamente cheia de rio Barigui nesse trecho
  específico) -- não invalida o evento, só não o corrobora por essa via hidrológica
  específica.
- Diferente de Recife (onde só 2 estações tinham dado moderno de qualquer tipo),
  Curitiba tem **muitas** estações de cota com séries longas e correntes (várias até
  2024/2025) -- a região é hidrologicamente bem mais instrumentada que a RMR para este
  tipo de dado.

## Limitações explícitas

- Isto é corroboração hidrológica (mesmo padrão do Lead B Recife), não geometria de
  ocorrência -- não resolve o bloqueio de `SUSC_18C` (geometria oficial ainda ausente).
- `MediaDiaria=0` em muitas linhas de cota indica leitura instantânea/telemétrica, não
  média diária -- os valores usados são os reais retornados pelo campo `CotaNN` do dia,
  consistente com o que o endpoint disponibiliza publicamente.
- Nenhum label criado; isto é evidência de corroboração, não confirmação de ocorrência
  em si (a confirmação administrativa já existe via fonte oficial, documentada em 18C).

## Arquivos

- `results/ana_curitiba_series_coverage.csv` -- cobertura completa das 63 estações
- `results/ana_curitiba_full_inventory.json` -- inventário bruto das 104 estações da bbox
- `scripts/fetch_ana_curitiba.py` -- script reexecutável (rede real, sem credenciais)
