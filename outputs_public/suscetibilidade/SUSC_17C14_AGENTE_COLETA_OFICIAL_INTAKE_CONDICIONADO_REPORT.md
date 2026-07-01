# SUSC-17C14 - Agente de coleta oficial e intake condicionado

## Objetivo
Este marco troca a avaliacao manual subjetiva por gates objetivos G1-G7 aplicados pelo agente. O agente prepara fila, registra tentativas dry-run, recebe arquivos locais quando houver intake real e classifica automaticamente o resultado.

## Modos do agente
`queue`, `collect`, `submit`, `intake`, `validate-artifact`, `status` e `audit` foram implementados. `collect` e `submit` sao bloqueados por padrao para acao externa; `intake` exige arquivo local real.

## Gates implementados
G1 existencia documental; G2 confiabilidade da fonte; G3 precisao temporal; G4 vinculo espacial; G5 separacao de fenomeno; G6 proveniencia e integridade; G7 politica anti-leakage.

## Ground Reference Candidate
`ground_reference_candidate` e uma referencia candidata para revisao, nao ground truth. Ela exige todos os gates aprovados, permanece `review_only=true`, `trainable=false`, `ground_truth=false` e `eligible_for_17b_now=false`.

## PDF, geometria e fenomeno
PDF sem coordenada, vetor, endereco preciso ou mapa georreferenciavel nao passa G4. A classificacao de fenomeno separa evidencia hidrologica de movimento de massa; `MIXED_CONFIRMED`, `MASS_MOVEMENT` e `UNKNOWN` nao validam inundacao.

## Resultado do build inicial
- Pedidos na fila: 9.
- Tentativas registradas: 9.
- Artefatos ingeridos: 0.
- Artefatos com todos os gates aprovados: 0.
- Ground Reference Candidates aceitos: 0.
- Artefatos rejeitados ou pendentes: 9.

## Guardrails
Nenhuma resposta foi inventada, nenhum contato ou protocolo foi inventado, nenhum artefato sem hash foi aceito, nenhuma fonte nao oficial foi aceita e nenhum dado pos-evento virou feature pre-evento.

## Score e 17B
Score v6 nao mudou, score v7 continua inexistente e 17B permanece bloqueado porque nao ha resposta oficial real com hash, manifesto, geometria, temporalidade, fenomeno e QA aceitos.

## Proximo marco recomendado
SUSC-17C15 Registro de Submissoes Manuais Reais e Intake de Respostas Oficiais quando houver arquivo/resposta real
