# Protocolo C — v1ud Real Source Acquisition and Resolver Hardening

## Objetivo

Transformar a infraestrutura v1uc (dry-run) em uma primeira aquisição real controlada,
segura e auditável de fontes externas oficiais, mantendo todos os guardrails.

## Pipeline

```
v1uc Configs + Events
        ↓
v1ud Targets YAML → URL Resolver → Resolution Registry
                                        ↓
                            Manifest Builder → Download Manifest
                                        ↓
                            Real Acquisition → Extraction Registry
                                        ↓
                            Integrity Audit → Integrity + Gate Delta
                                        ↓
                            Source Parsers → Next Actions + Report
```

## Fontes e Prioridades

### Prioridade 1 — Aquisição Real
| Fonte | Papel | Download? |
|-------|-------|-----------|
| INMET/BDMEP | Precipitação observada | Sim (dados históricos públicos) |
| ANA/HidroWeb | Nível d'água, vazão | Sim (se URL direta) |
| Cemaden | Pluviômetros, alertas | Metadata only (HTML discovery) |
| SGB/CPRM | Cartografia de risco | Metadata only (requer solicitação) |

### Prioridade 2 — Metadata Discovery
| Fonte | Papel | Download? |
|-------|-------|-----------|
| Copernicus EMS | Produtos operacionais | Metadata (buscar ativação) |
| International Charter | Quickviews | Metadata (contextual only) |

### Prioridade 3 — Context Only
| Fonte | Papel | Download? |
|-------|-------|-----------|
| Maxar/Vantor | VHR review | Metadata only |
| Planet | VHR review | Metadata only |
| EM-DAT | Inventário macro | Context only |

## Controles de Segurança

1. **Domain allowlist:** Apenas domínios listados em `v1ud_allowed_domains.yaml`
2. **Download manifest:** Nada baixa sem estar no manifesto
3. **Size limits:** max_download_mb configurável por domínio e globalmente
4. **Rate limiting:** Mínimo 1.5s entre requests
5. **User-Agent:** Identificação acadêmica educada
6. **Hash verification:** SHA256 para todo download
7. **License check:** Downloads com licença desconhecida bloqueados
8. **No scraping:** Sem browser automation, sem bypass de bloqueios

## Tipos de Ação Resultante

| Ação | Significado |
|------|-----------|
| DOWNLOAD_RETRY | Retry de download que falhou por rede |
| MANUAL_REVIEW | Requer revisão humana do conteúdo |
| FORMAL_REQUEST_REQUIRED | Requer solicitação formal à instituição |
| LICENSE_REVIEW_REQUIRED | Licença precisa ser revisada |
| PDF_TEXT_REVIEW | PDF baixado precisa de revisão de conteúdo |
| GEOMETRY_AUDIT_REQUIRED | Geometria detectada precisa de auditoria |
| PHENOMENON_SEPARATION_REQUIRED | Evento misto precisa de separação |
| EVENT_SPECIFIC_SOURCE_NEEDED | URL genérica precisa de versão específica |
| KEEP_CONTEXT_ONLY | Fonte é apenas contexto, nunca ground reference |
| REJECT_AS_GROUND_REFERENCE | Fonte rejeitada para ground reference |

## Guardrails Permanentes

- `ground_truth_operational = false`
- `can_create_training_label = false`
- `can_reopen_protocol_b = false`
- `dino_usage = SUPPORT_ONLY`
- `no_overlay_executed = true`
- `no_coordinates_invented = true`
