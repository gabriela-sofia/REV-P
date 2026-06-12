# v2bj Recife Candidate Gate Reconciliation

Consolidacao dos resultados reais do intake v2bi (Charter 758, APAC, ANA HidroWeb) e da
auditoria de disponibilidade INMET, com reconciliacao dos gates C0-C7 de Recife. Nada e
promovido: o status de referencia do candidato `REC_2022_05_24_30` e `CANDIDATE_REFERENCE_PENDING_HUMAN_REVIEW`.

Charter Recife: mapa raster em resolucao plena presente para revisao, sem vetor e sem CRS
legivel -> C4 permanece pendente de vetor/CRS. Feicao = landslide scars (nao flood extent).
Temporal: ANA Capibaribe (Sao Lourenco da Mata, RMR) fornece cota datada na janela como
contexto hidrologico; APAC mensal como contexto; serie local de chuva de Recife continua
pendente (A301 com precipitacao vazia, Cemaden por download manual).

C3 permanece PASS. C7 permanece BLOCKED. Ground truth final, labels, negativos e treino = 0.
River level nao e precipitacao; precipitacao nao e flood extent; proxy regional nao e a
estacao local de Recife.
