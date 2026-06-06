# Protocolo C v2am - claims e guardrails (apendice)

Consolidacao de claims permitidos/proibidos e guardrails para a banca.
Termos proibidos aparecem apenas como exemplos negativos / unsafe wording,
nunca como afirmacao positiva do que o projeto faz.

## Linguagem segura para banca
- Falar em evidencia contextual, candidatos revisaveis e estado review-only.
- Declarar explicitamente a ausencia de ground truth operacional patch-level.
- Tratar revisao e adjudicacao como pendentes.

## Registro
| registry_id | tipo | status | linguagem_segura | exemplo_proibido_unsafe | risco_banca |
| --- | --- | --- | --- | --- | --- |
| CG_v2am_0000 | claim | allowed | camada revisavel de candidatos | evitar (unsafe): ground truth validado | Subdeclarar evidencia contextual valida. |
| CG_v2am_0001 | claim | allowed | evidencia contextual | evitar (unsafe): deteccao de enchente | Subdeclarar evidencia contextual valida. |
| CG_v2am_0002 | claim | allowed | suporte territorial externo | evitar (unsafe): classe positiva | Subdeclarar evidencia contextual valida. |
| CG_v2am_0003 | claim | allowed | uso review-only | evitar (unsafe): label operacional | Subdeclarar evidencia contextual valida. |
| CG_v2am_0004 | claim | allowed | sem ground truth operacional patch-level | evitar (unsafe): ground truth validado | Subdeclarar evidencia contextual valida. |
| CG_v2am_0005 | claim | allowed | sem label binario | evitar (unsafe): classe positiva | Subdeclarar evidencia contextual valida. |
| CG_v2am_0006 | claim | allowed | sem validacao para uso operacional | evitar (unsafe): validacao de inundacao observada | Subdeclarar evidencia contextual valida. |
| CG_v2am_0007 | claim | allowed | sem predicao | evitar (unsafe): modelo preditivo | Subdeclarar evidencia contextual valida. |
| CG_v2am_0008 | claim | allowed | pacotes aguardando revisao humana | evitar (unsafe): treinamento supervisionado pronto | Subdeclarar evidencia contextual valida. |
| CG_v2am_0009 | claim | prohibited | nao afirmar resultado operacional | evitar (unsafe): Protocolo B aberto | Overclaim ou promocao indevida de candidato. |
| CG_v2am_0010 | guardrail: sem ground truth operacional patch-level | guardrail | Declarar ausencia de referencia operacional patch-level. | evitar (unsafe): ground truth validado | Banca pode acusar overclaim se ground truth for afirmado. |
| CG_v2am_0011 | guardrail: promotion_allowed=false | guardrail | Manter candidatos sem promocao operacional. | evitar (unsafe): classe positiva | Promover candidato sem evidencia seria insustentavel. |
| CG_v2am_0012 | guardrail: revisao humana pendente | guardrail | Tratar revisao como trabalho futuro, nao concluido. | evitar (unsafe): deteccao de enchente | Afirmar revisao concluida seria falso. |
| CG_v2am_0013 | guardrail: sem treino/overlay/predicao | guardrail | Manter pipeline read-only e estrutural. | evitar (unsafe): treinamento supervisionado pronto | Afirmar modelo preditivo seria overclaim. |
