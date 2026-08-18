# SUSC-17C30 - Gate Promotion Engine, Source Fingerprint e Replication Pipeline

## Objetivo
Macro-marco que une (1) Source Quality Fingerprint - aprender por que Agencia Brasil/EBC funcionou e classificar as fontes ja adquiridas; (2) Gate Promotion Engine - reavaliar 17C28/17C29 abrindo gates documentais G1/G2/G3/G6/G7 e decompondo G4/G5 em subgates; e (3) Replication Pipeline - gerar fila de aquisicao futura orientada pelos subgates faltantes e decidir follow-up multimodal (CHIRPS/Sentinel-2/Sentinel-1 SAR/embedding/QA).

## Fingerprint
- Fontes classificadas: 12 (12 artefatos oficiais/institucionais 17C28 + 17C29).
- Perfil de referencia Agencia Brasil/EBC: criado (tipo=official_institutional_public_agency, tier=institutional_public).
- Scoring policy: criada (10 sinais, 6 classes A-F).

## Gate Promotion
- Gates documentais abertos: 60
- G1=12 G2=12 G3=12 G6=12 G7=12
- G4a=11 G4b=9 G4c=0 G4d=0 G4_full=0
- G5a=3 G5b=8 G5c=3 G5d=0 G5_full=0
- Gate transition ledger: 94 transicoes registradas.
- Subgate matrix: 12 linhas.

## Replication
- Fila de aquisicao: 36 candidatos.
- Mappings fonte->evento: 12.
- Documentary evidences: 12. Event record candidates: 11.
- Decisoes multimodais: 12.

## Guardrails
- Nenhuma fonte virou ground truth, patch positivo, training label, score v7 ou desbloqueou 17B.
- G4_full/G5_full permanecem false: sem geometria patch-level e fenomeno misto.
- Bairro/cidade nao virou G4_full sem incerteza; fenomeno misto nao virou G5_full.
- CHIRPS/Sentinel/SAR sao follow-up tecnico e nao evento observado. Embedding nao recebe noticia como input.
- Score v6 intacto. Score v7 inexistente.

## minimum_success_achieved: True

## Proximo marco recomendado
SUSC-17C31 Aquisicao dirigida das fontes prioritarias da fila (Defesa Civil PE, APAC, CEMADEN, SGB/CPRM, Copernicus EMS, S2iD) para atacar G4c/G4d/G5d com geometria patch-level e fenomeno separado, mantendo score v6 intacto e 17B fail-closed.
