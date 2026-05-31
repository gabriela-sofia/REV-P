# v1ou — External Evidence Source Inventory

## Objetivo

Escanear arquivos existentes do repositório para identificar candidatos a fontes e evidências externas de eventos observados. Não usa internet, não baixa nada, não executa OCR. Lê apenas headers + primeiras linhas (metadata-only).

## Resultado

Total de candidatos encontrados: 330. Permitidos para registro de eventos: 22. Bloqueados: 308. Fixture/sintético excluídos: 0.

## Guardrails

Nenhum candidato é promovido a ground truth operacional. allowed_for_event_registry=true significa apenas que o arquivo contém termos relevantes e pode ser inspecionado para construir o registro de eventos. Não implica confirmação de evento, label ou target.

## Relação com v1og-v1ot

v1og-v1ot confirmou TEMPORAL_RECOVERY_FAIL_CLOSED para Recife: 0 product_dates confirmadas, 0 C3+ candidates, 0 formal negatives. v1ou não tenta destravar temporal artificialmente.
