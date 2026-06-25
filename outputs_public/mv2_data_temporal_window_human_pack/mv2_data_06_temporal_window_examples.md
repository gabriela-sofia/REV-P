# DATA-06 - Exemplos (ilustrativos, nao preencher o template com estes valores)

Os exemplos abaixo mostram o formato esperado. Eles NAO devem ser copiados para
o template real: cada linha precisa da sua propria fonte verificada.

## Exemplo valido (promove)
- temporal_window_start: 2022-05-24
- temporal_window_end: 2022-05-31
- temporal_window_source: Defesa Civil PE - boletim de ocorrencia
- source_ref: protocolo-interno-REC-2022-05-24 (registro auditavel)
- source_type: DEFESA_CIVIL
- evidence_strength: STRONG
- review_status: APPROVED

## Exemplo bloqueado (sem fonte)
- temporal_window_start: 2022-05-24
- temporal_window_end: 2022-05-31
- temporal_window_source: (vazio)
- source_ref: (vazio)
- review_status: PENDING_HUMAN_FILL
- Resultado: BLOCKED_NO_SOURCE.

## Exemplo recusado (inferencia proibida)
- Preencher data so porque "choveu muito naquele mes" sem boletim: proibido.
- Inferir janela a partir do bbox: proibido.
