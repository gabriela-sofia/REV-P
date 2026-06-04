# Protocolo C — Relatório v1ud Real Source Acquisition

**Gerado em:** 2026-06-03T08:56:33.945335  
**Versão:** v1ud  

## Resumo de Aquisição

| Métrica | Valor |
|---------|-------|
| Total de entradas | 20 |
| Downloads completados | 7 |
| Skipped (não permitido) | 13 |
| Dry-run | 0 |
| Erros de rede | 0 |
| Dependência ausente | 0 |

## Gate Delta (v1uc → v1ud)

| Ganho | Contagem |
|-------|----------|
| Hash SHA256 | 7 |
| Texto de PDF | 0 |
| Geometria | 0 |
| Links HTML | 3 |
| Ainda bloqueados | 20 |

## Classificação de Fontes

| Tipo | Descrição |
|------|-----------|
| Fonte genérica (portal homepage) | Página inicial de portal — precisa de URL específica |
| Fonte específica de evento | URL que referencia evento ou período específico |
| Potencial de referência | Fonte oficial com dados observacionais e geometria |
| Apenas contexto | Inventário macro, quickview, suscetibilidade |

## Próximas Ações

| Ação | Contagem |
|------|----------|
| EVENT_SPECIFIC_SOURCE_NEEDED | 2 |
| FORMAL_REQUEST_REQUIRED | 3 |
| KEEP_CONTEXT_ONLY | 6 |
| LICENSE_REVIEW_REQUIRED | 2 |
| MANUAL_REVIEW | 7 |

## Prioridades de Solicitação Formal

1. **SGB_CPRM_CARTOGRAFIA** / PET_2022_02_15: FORMAL_REQUEST_REQUIRED for SGB_CPRM_CARTOGRAFIA/PET_2022_02_15 (portal generic — may need event-specific URL)
1. **SGB_CPRM_CARTOGRAFIA** / PET_2022_02_15: FORMAL_REQUEST_REQUIRED for SGB_CPRM_CARTOGRAFIA/PET_2022_02_15 (portal generic — may need event-specific URL)
1. **SGB_CPRM_CARTOGRAFIA** / PET_2022_02_15: FORMAL_REQUEST_REQUIRED for SGB_CPRM_CARTOGRAFIA/PET_2022_02_15

## Por Que Ground Truth Continua Bloqueado

1. **G10 (patch_overlay_possible):** Overlay não implementado em v1ud
2. **G11 (supervisor_review_completed):** Revisão humana não executada
3. **Fenômeno misto:** Eventos PET ainda com separação pendente
4. **Geometria ausente:** Maioria das fontes não fornece geometria diretamente
5. **Licenças pendentes:** Algumas fontes requerem revisão de termos

## Invariantes — Confirmação Explícita

- **ground_truth_operational** = `False`
- **can_create_training_label** = `False`
- **can_reopen_protocol_b** = `False`
- **dino_usage** = `SUPPORT_ONLY`
- **no_overlay_executed** = `True`
- **no_coordinates_invented** = `True`

---
*Relatório gerado por Protocol C v1ud.*
