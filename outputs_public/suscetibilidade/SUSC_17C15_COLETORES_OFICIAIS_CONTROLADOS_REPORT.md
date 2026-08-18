# SUSC-17C15 - Coletores oficiais controlados e drivers de canal

## Objetivo
O 17C15 passa do dry-run puro para drivers de canal com capacidade de coleta publica controlada. O build continua offline e reprodutivel; rede exige `SUSC_17C15_ALLOW_NETWORK=1`.

## Drivers implementados
OfficialStaticDownloadDriver, OfficialPageProbeDriver, OfficialPdfTextProbeDriver, OfficialDataPortalDriver, OfficialSensorCatalogProbeDriver, BlockedAuthenticatedPortalDriver

## Capacidade de canais
- Canais avaliados: 9.
- Canais coletaveis publicamente com opt-in de rede: 2.
- Canais bloqueados por autenticacao/protocolo/verificacao humana: 5.

## Resultado offline
- Tentativas de coleta registradas: 9.
- Artefatos coletados: 0.
- PDFs triados: 0.
- Catalogos sensores sondados com metadados: 0.
- Ground Reference Candidates aceitos: 0.
- Rejeitados ou pendentes: 9.

## Gates
Sem artefato real no build offline, G1, G3, G4, G5 e G6 permanecem bloqueados. G2 so passa para canais oficiais confirmados; G7 permanece preservado porque nenhum dado pos-evento virou feature pre-evento.

## Guardrails
Nenhuma submissao externa, protocolo, resposta inventada, ground truth, label, score v7, patch oficial ou patch-link oficial foi criado. Bruto pesado nao foi salvo em `outputs_public`.

## Proximo marco recomendado
SUSC-17C16 Coleta publica com opt-in de rede para catalogos CHIRPS/CDSE ou intake de respostas oficiais reais
