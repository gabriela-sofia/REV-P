# v1rc — External Document Intake Validator

## Objetivo

Validar um CSV de intake preenchido manualmente (REVP_PROTOCOL_C_EXTERNAL_INTAKE_PATH). Sem o arquivo, fail-closed (EXTERNAL_INTAKE_WAITING_MANUAL_DOCUMENTS). Uma linha por checagem por documento.

## Checagens

required_field_*, weak_source_family, missing_temporal_precision, missing_spatial_precision, missing_provenance, license_access_unknown, path_url_unsafe. URLs nunca sao acessadas.

## Resultado

Status: EXTERNAL_INTAKE_WAITING_MANUAL_DOCUMENTS. Documentos: 0. Checagens: 0 (passou 0, falhou 0).

## Guardrails

Nenhuma URL e baixada. Nenhum evento confirmado. review_only=true. formal_negative=false. Nenhum label/target/ground truth operacional.
