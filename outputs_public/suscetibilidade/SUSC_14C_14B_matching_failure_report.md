# SUSC-14C - auditoria dos gargalos herdados do SUSC-14B

Status: review-only. Sem ground truth, sem treino supervisionado e sem score v7.

- Ocorrencias auditadas: **4412**
- Features oficiais 14B disponiveis: **100000**
- Falhas por classe: **{'blocked_no_official_reference': 146, 'reference_not_processed_due_batch_cap': 146, 'street_type_mismatch': 250, 'weak_neighborhood_only': 4194, 'number/complement_noise': 37, 'blocked_ambiguous_address': 67, 'low_similarity': 54, 'other_review_required': 4, 'candidate_tie': 13, 'abbreviation_mismatch': 15, 'missing_street_name': 1598}**
- Ruas sem referencia ranqueadas: **36**
- Ruas ambiguas ranqueadas: **16**
- Bairros com falhas ranqueados: **79**
- Fontes nao tentadas por teto de lote: **392**

A auditoria direciona o 14C para recursos oficiais ainda nao tentados e para
casos onde a normalizacao/fuzzy matching pode reduzir ambiguidade sem inventar
coordenadas ou promover bairro para patch-level.
