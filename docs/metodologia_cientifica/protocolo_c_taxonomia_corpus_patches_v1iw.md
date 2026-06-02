# Protocolo C v1iw - Taxonomia dos corpus de patches

## Objetivo

Esta etapa formaliza uma distincao que precisa ficar explicita no TCC: o REV-P usa mais de um conjunto contado em "patches" ou "assets", mas esses conjuntos nao tem a mesma unidade metodologica.

Nao ha contradicao entre 59 e 128. Eles respondem a perguntas diferentes.

## Camadas

| Camada | Total | Recife | Petropolis | Curitiba | O que representa |
|---|---:|---:|---:|---:|---|
| Corpus territorial consolidado | 59 | 18 | 27 | 14 | Unidades territoriais/contextuais do estudo |
| Manifesto Sentinel candidato | 128 | 37 | 48 | 43 | Assets Sentinel elegiveis para pipeline e revisao |
| Subset DINO operacional | derivado em execucao | derivado | derivado | derivado | Patches com embedding local executado |
| Unidades oficiais documentadas | derivado da v1ir | derivado | derivado | derivado | Eventos/localidades documentados por fonte oficial |
| Anchors espaciais oficiais | derivado da v1is | derivado | derivado | derivado | Coordenadas oficiais que podem orientar aquisicao futura |

## Como interpretar 59

O numero 59 e o universo territorial consolidado: 18 Recife, 27 Petropolis e 14 Curitiba. Ele deve ser usado quando o texto falar do corpus territorial do estudo, das regioes analisadas e da base contextual sobre a qual a metodologia foi organizada.

Esses 59 nao sao rotulos. Eles nao provam ocorrencia observada de evento em cada patch. Eles tambem nao autorizam treinamento supervisionado. Sao unidades territoriais para auditoria, revisao estrutural e organizacao de evidencias.

## Como interpretar 128

O numero 128 vem do manifesto Sentinel candidato. Ele conta referencias de assets Sentinel preparados para o pipeline Sentinel-first. Como uma unidade territorial pode ter mais de uma referencia Sentinel, e como o manifesto organiza assets e candidatos de entrada, esse numero nao precisa coincidir com 59.

Os 128 devem ser usados quando o texto falar de disponibilidade Sentinel, planejamento de embeddings, preflight de assets e cobertura operacional do pipeline. Eles nao sao ground truth, nao sao rotulos e nao substituem revisao humana.

## Qual numero usar no TCC

Na Metodologia, use 59 para descrever o corpus territorial consolidado e use 128 para explicar a camada de assets Sentinel candidatos. A frase correta e: "O corpus territorial consolidado tem 59 patches; o manifesto Sentinel-first possui 128 assets candidatos associados a esses contextos territoriais."

Nos Resultados, use 59 quando discutir o alcance territorial do estudo. Use 128 quando discutir a preparacao Sentinel e a auditabilidade do pipeline de entrada. Use o numero do subset DINO apenas quando falar de embeddings efetivamente executados.

Ao falar de ground truth, nenhum desses numeros deve ser apresentado como evidencia operacional. Ground truth operacional continua bloqueado ate existir evidencia oficial suficiente, patch Sentinel centrado no anchor, bandas, QA e manifest estavel.

## Regra de escrita

Patches nao sao labels. Candidatos Sentinel nao sao ground truth. Embeddings DINO sao representacoes estruturais de revisao, nao categorias supervisionadas. Unidades oficiais e anchors do Protocolo C organizam a busca de evidencia, mas nao criam target de treinamento.
