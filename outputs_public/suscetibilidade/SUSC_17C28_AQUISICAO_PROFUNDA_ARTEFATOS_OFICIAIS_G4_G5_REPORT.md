# SUSC-17C28 - Aquisicao profunda de artefatos oficiais especificos para G4/G5

## Objetivo
Aprofundar a aquisicao em fontes oficiais/institucionais especificas do evento (enchentes e deslizamentos em Pernambuco/Recife, maio 2022) via snapshots arquivados (Wayback) e links seguidos, com hash, parse e avaliacao G4/G5.

## Aquisicao profunda
- Plano expandido: 30 linhas (event/location/phenomenon/geometry).
- Tentativas de busca profunda: 56.
- Links seguidos: 2.
- Artefatos oficiais especificos adquiridos: 4; parseados: 4.

## Candidatos e G4/G5
- Observed event candidates especificos: 4.
- Candidatos com local patch-level/bairro: 2; com fenomeno especifico: 4.
- Avaliacoes G4/G5: 8; Ground Reference Candidates avaliados: 4; aceitos: 0.
- G4_true_count=0, G5_true_count=0.

## Resultado cientifico (honesto)
- Fontes oficiais institucionais (Agencia Brasil/EBC) confirmam o evento com data, Recife/Jaboatao/Olinda e fenomeno, mas o fenomeno e misto (inundacao + deslizamento): G5 nao e satisfeito.
- A localizacao disponivel e municipal/bairro, sem geometria ou coordenada patch-level: G4 nao e satisfeito. Nenhuma coordenada foi inventada.
- Nenhum Ground Reference Candidate foi aceito; 17B permanece bloqueado.

## Guardrails
- Sensor/CHIRPS nao viraram evento observado; noticia nao virou Ground Reference sozinha; nenhum ground truth, label, treino, score v7 ou patch oficial; score v6 intacto.

## minimum_success_achieved: True

## Proximo marco recomendado
SUSC-17C29 Aquisicao de geometria oficial de evento (mancha/poligono/coordenada) e classificacao de fenomeno por local para tentar G4/G5 patch-level
