# SUSC-17C27 - Aquisicao dirigida de artefatos de evento observado para G4/G5

## Objetivo
Executar aquisicao real de artefatos publicos/semipublicos do evento observado (enchentes e deslizamentos em Pernambuco/Recife, maio 2022) usando a fila do 17C26, com hash, parsing, extracao de candidatos e avaliacao G4/G5.

## Aquisicao
- Query packages consumidos: 15.
- Tentativas de aquisicao: 18.
- Artefatos reais adquiridos: 4 (manifestos: 4).
- Artefatos parseados: 4.

## Candidatos de evento observado e G4/G5
- Observed event candidates: 1.
- Location resolutions: 1; phenomenon classifications: 1.
- Avaliacoes G4/G5: 2.
- Ground Reference Candidates avaliados: 1; aceitos: 0.
- G4_true_count=0, G5_true_count=0.

## Resultado cientifico (honesto)
- O evento de maio 2022 foi misto (inundacoes + deslizamentos): G5 (separacao de fenomeno) nao e satisfeito por fonte de referencia.
- A localizacao disponivel e cidade/regiao metropolitana, sem geometria patch-level: G4 (vinculo espacial de evento) nao e satisfeito. Nenhuma coordenada foi inventada.
- Fonte de referencia (Wikipedia) serve como observed_event_candidate/triagem, nunca como Ground Reference aceito sozinha.

## Guardrails
- Sensor e CHIRPS nao viraram evento observado; nenhum ground truth, label, treino, score v7 ou patch oficial foi criado; score v6 intacto.
- eligible_for_17b_now=False (permanece bloqueado sem Ground Reference Candidate aceito por G1-G7).

## minimum_success_achieved: True

## Proximo marco recomendado
SUSC-17C28 Consulta dirigida a boletins/relatorios oficiais especificos (APAC/CEMADEN/CPRM) com geometria e classificacao de fenomeno para tentar satisfazer G4/G5 patch-level
