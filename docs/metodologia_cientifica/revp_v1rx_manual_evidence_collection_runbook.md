# v1rx — Manual Evidence Collection Runbook

## Objetivo

Guia passo a passo para coletar documentos externos (P1) e preencher respostas de revisão A/B + decisão supervisora (P2).

## Fontes prioritárias

| ID | Fonte | Família | Regiões | Prioridade |

| -- | ----- | ------- | ------- | ---------- |

| SRC01 | CEMADEN | OFFICIAL_HYDROMETEOROLOGICAL | ALL | P0 |

| SRC02 | ANA / HidroWeb | OFFICIAL_HYDROMETEOROLOGICAL | RECIFE;CURITIBA | P0 |

| SRC03 | INMET / BDMEP | OFFICIAL_HYDROMETEOROLOGICAL | ALL | P0 |

| SRC04 | SGB / CPRM | OFFICIAL_GEOLOGICAL | PET | P0 |

| SRC05 | Defesa Civil municipal/estadual | OFFICIAL_CIVIL_DEFENSE | ALL | P0 |

| SRC06 | Diário Oficial | OFFICIAL_GOVERNMENT_PUBLICATION | ALL | P1 |

| SRC07 | Relatórios técnicos/artigos | TECHNICAL_REPORT | ALL | P1 |

| SRC08 | Mídia jornalística | NEWS_MEDIA_SECONDARY | ALL | P2 |

| SRC09 | Redes sociais | SOCIAL_MEDIA_SECONDARY | ALL | P3 |

## Fluxo completo

**RS01** [COLLECT] Collect official document for event/date/location
- Script: `manual web search`
- Template: v1rb intake template
- Esperado: Document reference saved

**RS02** [REGISTER] Record source metadata in intake template (v1rb)
- Script: `manual edit`
- Template: datasets/protocol_c_external_document_intake_template_v1rb.csv
- Esperado: Row added with all required fields

**RS03** [VALIDATE] Validate intake with v1rc
- Script: `python scripts/protocolo_c/revp_v1rc_external_document_intake_validator.py`
- Template: —
- Esperado: EXTERNAL_INTAKE_VALIDATION_PASS_REVIEW_ONLY

**RS04** [CANDIDATES] Generate event candidates from intake (v1rd)
- Script: `python scripts/protocolo_c/revp_v1rd_event_candidate_builder_from_external_intake.py`
- Template: —
- Esperado: Event candidates review-only created

**RS05** [LINK] Link event candidates to patches (v1re)
- Script: `python scripts/protocolo_c/revp_v1re_external_event_patch_candidate_linker.py`
- Template: —
- Esperado: Link candidates review-only

**RS06** [FILL_AB] Fill A/B review responses using v1rg template
- Script: `manual edit`
- Template: datasets/protocol_c_review_response_intake_template_v1rg.csv
- Esperado: Responses filled for all samples

**RS07** [VALIDATE_RESP] Validate A/B responses with v1rh
- Script: `python scripts/protocolo_c/revp_v1rh_review_response_validator.py`
- Template: —
- Esperado: REVIEW_RESPONSES_VALIDATION_PASS_REVIEW_ONLY

**RS08** [SCORE] Score completed double-review with v1ri
- Script: `python scripts/protocolo_c/revp_v1ri_completed_review_scoring_replay.py`
- Template: —
- Esperado: Review scores computed

**RS09** [SUPERVISOR] Generate supervisor packet with v1rj
- Script: `python scripts/protocolo_c/revp_v1rj_supervisor_review_packet_generator.py`
- Template: —
- Esperado: Supervisor packets ready if C3 candidate

**RS10** [SUP_FILL] Fill supervisor decision using v1rk template
- Script: `manual edit`
- Template: datasets/protocol_c_supervisor_decision_intake_template_v1rk.csv
- Esperado: Decision filled

**RS11** [SUP_VALIDATE] Validate supervisor decision with v1rl
- Script: `python scripts/protocolo_c/revp_v1rl_supervisor_decision_validator.py`
- Template: —
- Esperado: SUPERVISOR_DECISIONS_VALIDATION_PASS_REVIEW_ONLY

**RS12** [BUNDLE] Consolidate with v1rm and v1rr
- Script: `python scripts/protocolo_c/revp_v1rm_review_supervisor_gate_bundle.py && python scripts/protocolo_c/revp_v1rr_scientific_roadmap_bundle.py`
- Template: —
- Esperado: Final status updated

## Regras

1. Mídia jornalística é secundária: nunca suficiente sozinha para C3.

2. Redes sociais apenas para triagem: evidência fraca.

3. Nunca usar ausência de evidência como negativo formal.

4. DINO pode priorizar revisão, mas nunca valida evento.

5. Toda decisão C3 exige supervisor humano.
