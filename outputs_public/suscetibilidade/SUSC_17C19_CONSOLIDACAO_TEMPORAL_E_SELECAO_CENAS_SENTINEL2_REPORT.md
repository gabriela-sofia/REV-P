# SUSC-17C19 - Consolidacao temporal sensorial e selecao canonica de cenas Sentinel-2

## O que o 17C18 demonstrou
O 17C18 provou que a consulta Sentinel-2 por AOI funciona: retornou 100 cenas reais para os 5 patches candidatos de Recife. Mas revelou uma lacuna de linhagem: a data/periodo do evento REC_2022_05_24_30 so estava disponivel em `susc_17c4_extracted_reference_candidates.csv`, nao nos artefatos 17C6-17C17.

## Por que a ausencia de periodo era uma lacuna de linhagem
Sem o periodo propagado, qualquer marco sensorial downstream teria que reconstruir a janela temporal a partir do `event_id`, o que e fragil e proibido. A auditoria de lacuna mostra que 17c6 (grid e links) e 17c7 carregam o `event_id` mas nao o periodo, e 17c9/17c17 nem o `event_id`. Apenas o 17c18 finalmente propagou `event_id` + `event_date_or_period`.

## Como o 17C19 consolida o periodo sem parsear o ID
O periodo e lido do campo temporal oficial da fonte upstream `outputs_public/suscetibilidade/susc_17c4_extracted_reference_candidates.csv` (referencia S17C4_REF_REC_2022_05_24_30), com `parse_from_event_id=false` sempre. Periodo consolidado: 2022-05-24 a 2022-05-30 (precisao date_range_period).

## Janelas pre/durante/pos-evento
Para cada patch foi criado um binding temporal explicito: janela pre-evento de 30 dias antes do inicio ate 1 dia antes do inicio; janela durante-evento do inicio ao fim; janela pos-evento de 1 dia apos o fim ate 30 dias apos o fim.

## Selecao de cenas Sentinel-2
Entre as cenas pre-evento (is_pre_event=true, is_post_event=false, com datetime, scene_id e interseccao de AOI), a canonica e escolhida por menor cobertura de nuvem, desempatando por proximidade ao inicio do evento e preferencia por produto L2A.
- Cenas de entrada: 100.
- Cenas canonicas pre-evento: 5 (5 de 5 patches com cena canonica).
- Cenas rejeitadas/bloqueadas: 95.

## Por que cenas durante/pos-evento foram bloqueadas
Cenas durante o evento sao marcadas `during_event_blocked` e cenas apos o evento `post_event_blocked`; nenhuma pode virar feature pre-evento, para evitar vazamento temporal. Elas permanecem registradas apenas como contexto observacional futuro.

## Por que tile ainda nao foi criado
Foi manifestado apenas o pedido futuro de tile leve por cena canonica (`can_execute_now=false`), sujeito a politica de download e storage. Nenhum produto Sentinel-2 foi baixado e nenhum tile foi gerado.

## Por que CHIRPS ainda nao foi calculado
O plano de runtime CHIRPS reafirma que a fonte publica so oferece raster global pesado ou colecao GEE com runtime; sem endpoint leve por AOI, nenhuma estatistica zonal foi calculada.

## Por que nada vira Ground Reference
Linhagem temporal e cena Sentinel-2 sao camadas sensoriais/contextuais. Nenhuma passa G4 (vinculo espacial de evento) ou G5 (separacao de fenomeno); nenhuma vira Ground Reference Candidate, ground truth ou label.

## Score v6, score v7 e 17B
O score v6 nao mudou (nenhum dataset oficial tocado). O score v7 continua inexistente e o 17B permanece bloqueado ate existir artefato de evento com geometria e fenomeno, tile/feature sensorial real e politica de patch candidato.

## Proximo marco recomendado
SUSC-17C20 Politica de execucao de tile leve Sentinel-2 sobre cena canonica e runtime CHIRPS por AOI com storage aprovado
