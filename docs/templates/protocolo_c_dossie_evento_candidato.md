# Dossiê de Evento Candidato — Protocolo C

> **ATENÇÃO**: Este dossiê não é ground truth operacional.
> Ele documenta o estado atual de busca de evidências para um evento candidato.
> Ground truth operacional requer satisfação de todos os gates G1–G9 e revisão supervisora formal.

---

## Identificação

| Campo | Valor |
|-------|-------|
| **Dossier ID** | [DOSSIER_ID] |
| **Screening ID** | [SCREENING_ID] |
| **Região REV-P** | [REGIAO] |
| **Município** | [MUNICIPIO] |
| **Evento candidato** | [EVENTO_CANDIDATO] |
| **Período candidato** | [PERIODO_CANDIDATO] |
| **Status do evento** | [STATUS_DO_EVENTO] |
| **Status do dossiê** | [STATUS_DO_DOSSIE] |

---

## Fontes-alvo identificadas

[FONTES_ALVO]

*Liste as instituições, portais ou produtos identificados no backlog de busca (event_source_search_backlog.csv).*

---

## Evidência temporal esperada

[EVIDENCIA_TEMPORAL_ESPERADA]

*Descreva a data ou janela temporal do evento que precisa ser confirmada por fonte rastreável. Não inferir de contexto regional sem fonte.*

---

## Evidência espacial esperada

[EVIDENCIA_ESPACIAL_ESPERADA]

*Descreva o tipo de cobertura espacial necessária: mapa de área afetada, polígono de ocorrência, produto de detecção. Deve ter cobertura sobre os patches do corpus.*

---

## Produto ou sensor esperado

[PRODUTO_OU_SENSOR_ESPERADO]

*Ex: Sentinel-2 L2A, SAR (Sentinel-1), GFM Copernicus, laudo CPRM, shapefile Defesa Civil.*

---

## Patches potencialmente relacionados

[PATCHES_POTENCIALMENTE_RELACIONADOS]

*Liste os patch_ids do corpus DINO que estão no perímetro de busca deste evento. Ver event_patch_screening_scope.csv.*

*O DINOv2 não fecha gate de evento, temporalidade, espacialidade ou ground truth.*

---

## Gates potencialmente endereçáveis

[GATES_POTENCIALMENTE_FECHADOS]

*Liste os gates G1–G9 que este evento pode futuramente ajudar a fechar se as evidências forem reunidas.*

---

## Lacunas atuais

[LACUNAS_ATUAIS]

*Liste os requisitos mínimos ainda não satisfeitos. Ex: ausência de fonte confirmada, licença desconhecida, sem cobertura espacial dos patches.*

---

## Bloqueadores ativos

[BLOQUEIOS]

*Liste qualquer bloqueador que impede avanço: acesso restrito, licença desconhecida, conflito entre fontes, etc. Se não houver bloqueador ativo, escrever "Nenhum bloqueador ativo no estado atual".*

---

## Próxima ação

[PROXIMA_ACAO]

*Ex: iniciar busca no portal CPRM; solicitar formalmente à Defesa Civil; verificar disponibilidade de Sentinel-2 no período; clarificar licença de uso acadêmico.*

---

## Claims

| Tipo | Descrição |
|------|-----------|
| **Claim permitido** | [CLAIM_PERMITIDO] |
| **Claim proibido** | [CLAIM_PROIBIDO] |

*Claim permitido: contexto territorial, referência metodológica, identificação de lacuna.*
*Claim proibido: ground truth operacional, flood label, training label, flood prediction, flood detection solo.*

---

*Dossiê gerado pelo Protocolo C — etapa v1ho. Não constitui declaração de ground truth.*
