# Relatorio v1ji - Batch multimodal dos anchors oficiais

## Escopo

A v1ji executa a primeira tentativa concreta de lote multimodal para os anchors oficiais com coordenada explicita. A etapa usa GEE autenticado quando disponivel, baixa somente patches pequenos para armazenamento local e publica apenas metadados sanitizados.

## Anchors oficiais

As coordenadas recuperadas dos PDFs CPRM sao deduplicadas por unidade documental. Coordenadas repetidas ou multiplas dentro da mesma unidade documental sao fundidas, sem perder os identificadores originais de recuperacao.

O resultado e um registro de anchors oficiais unicos, cada um com data, fenomeno, latitude, longitude, confianca da coordenada e status de deduplicacao.

## Sentinel-2, Sentinel-1 e DEM

Para cada anchor, a etapa busca janelas pre e pos em relacao a data documental:

- Sentinel-2 usa bandas opticas e mascara local quando disponivel;
- Sentinel-1 usa polarizacoes disponiveis para revisao SAR;
- DEM inclui terreno quando o produto GEE permite.

Cada patch recebe QA local. Ausencia de S1, DEM ou modelo nao quebra a etapa: o bloqueio fica registrado por candidato.

## DINO frozen

Quando o par Sentinel-2 pre e pos passa em QA, a v1ji prepara uma composicao visual B04/B03/B02 e extrai embedding DINOv2 frozen. O vetor bruto nao e versionado. O registro publico guarda apenas dimensao, similaridade e distancia.

O embedding serve para revisao estrutural. Ele nao cria label, nao define classe e nao libera treino.

## Status de treino

A matriz v1ji atualiza a prontidao multimodal, mas mantem o gate supervisionado bloqueado:

- positivos oficiais sao candidatos de referencia;
- negativos formais continuam zerados;
- controles candidatos nao viram negativos;
- DINO nao vira label;
- split e protocolo de vazamento continuam obrigatorios antes de qualquer treino.

O resultado defensavel da etapa e um lote review-only multimodal, quando os patches passam em QA.
