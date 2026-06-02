# Protocolo C v1jd - Recuperacao de coordenadas em PDFs CPRM

A v1jd aprofunda a auditoria documental dos anexos CPRM/DIGEAP locais para tentar ampliar o numero de anchors oficiais. A motivacao vem da v1jc: existe apenas um anchor confirmado, o que permite revisao multimodal, mas nao sustenta treino.

Esta etapa nao baixa novos dados, nao geocodifica bairros, nao usa centroide e nao cria ponto aproximado.

## Metodo

Foram auditados 10 PDFs locais dos anexos CPRM em `local_runs`. Os caminhos privados completos permanecem fora dos registries publicos.

O script tentou:

- extracao nativa com PyMuPDF, quando disponivel;
- extracao de texto/tabelas com pdfplumber, quando disponivel;
- varredura textual crua limitada em PDFs locais;
- OCR leve apenas se dependencias locais estivessem instaladas.

No ambiente desta execucao, PyMuPDF, pdfplumber, pytesseract e pdf2image nao estavam disponiveis. O status foi registrado como `OCR_NOT_AVAILABLE` e a etapa seguiu sem falhar.

## Padroes buscados

A busca considerou coordenadas em:

- graus decimais, como latitude/longitude;
- UTM, com validacao aproximada para Petropolis/RJ;
- graus, minutos e segundos;
- expressoes textuais como coordenadas GPS, Latitude, Longitude, Ponto, Localizacao, UTM, SIRGAS e WGS84.

Coordenadas inferidas por bairro, rua, municipio, centroide ou geocodificacao externa foram bloqueadas.

## Resultado

Resultado da auditoria:

- PDFs/anexos auditados: 10;
- coordenadas candidatas encontradas: 1;
- coordenadas validas: 1;
- novos anchors oficiais adicionais: 0;
- eventos que seguem sem coordenada explicita: 9.

A coordenada valida foi a ja consolidada para o ANEXO-II, Moinho Preto:

- latitude: -22.484251;
- longitude: -43.211257;
- status: `EXPLICIT_COORDINATE_HIGH`;
- anchor: `ANCHOR_PET2022_CPRM_ANEXOII_19022022`.

Nenhuma coordenada adicional foi recuperada dos demais anexos com os extratores disponiveis.

## Status metodologico

A v1jd confirma que o gargalo permanece documental e operacional: ha evidencias oficiais de vistoria, mas os anexos auditados nao forneceram novas coordenadas extraiveis no ambiente atual.

O status de treino permanece:

- `TRAINING_BLOCKED_INSUFFICIENT_LABELS`;
- `can_create_training_label=false`;
- `can_train_model=false`;
- `can_unfreeze_dino_for_scientific_claim=false`;
- `can_reopen_protocol_b=false`.

Se novos anchors forem recuperados em uma etapa posterior, o proximo passo tecnico sera gerar patches Sentinel em lote para esses anchors e repetir QA, selecao de par e embedding frozen.
