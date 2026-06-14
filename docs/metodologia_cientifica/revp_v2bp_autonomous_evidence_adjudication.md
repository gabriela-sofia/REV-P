# v2bp — Adjudicação autônoma de evidência e auditoria de consistência evento-patch

Versão: `v2bp`
Modo: auditoria metodológica autônoma. Não cria label, não cria negativo, não libera treino.

## Reinterpretação de "human review"

A partir desta etapa, "human review", "manual review", "review required" e
termos equivalentes no pipeline significam **auditoria autônoma estruturada**,
não "parar e pedir para a usuária revisar manualmente".

Quando os artefatos do repositório permitem decidir um caso de forma objetiva e
rastreável, o pipeline decide automaticamente: aceita, rejeita ou bloqueia com
base em consistência de região, patch, evento, janela temporal, suporte
espacial, geometria/overlay, independência de fonte e contradição. A
intervenção da usuária só é necessária quando há **ambiguidade metodológica
real** que os dados existentes não resolvem.

Isso reduz o gargalo manual sem afrouxar guardrail: a auditoria autônoma
classifica a qualidade e a consistência da evidência, mas **não** promove
evidência a label operacional.

## O que a etapa faz

Adjudica o registry de pacotes evento-patch do v2at
(`datasets/v2at_event_patch_package_registry.csv`, 172 pacotes) cruzando com o
catálogo de fontes, overlays, a feature table `v2bn` e o scaffold `v2bo`
(quando presentes; descoberta fail-closed por `rglob`). Para cada pacote produz
uma decisão auditável com regras explícitas e ordem determinística:

1. evento/patch UNKNOWN → `AUTO_REJECT_EVIDENCE_CONTRADICTORY`;
2. região do patch ≠ região declarada → `AUTO_REJECT_REGION_MISMATCH`;
3. tokens de região patch/evento divergentes → `AUTO_REJECT_PATCH_ID_MISMATCH`;
4. `conflict_count > 0` → `AUTO_REJECT_EVIDENCE_CONTRADICTORY`;
5. fonte de feature reusada como label → `AUTO_REJECT_SOURCE_CIRCULARITY`;
6. atribuição de evento ambígua (marcador explícito) → `NEEDS_USER_DECISION`;
7. só contexto / sem fonte oficial / sem âncora temporal → `AUTO_REVIEW_INSUFFICIENT_EVIDENCE`;
8. overlay patch-evento presente e tudo consistente → `AUTO_ACCEPT_EVIDENCE_CONSISTENT`;
9. evidência forte/independente/temporal consistente, sem overlay → `READY_FOR_GT_PROTOCOL_REVIEW` + `AUTO_VALIDATED_CANDIDATE_POSITIVE`;
10. evidência oficial mas secundária, sem overlay → `BLOCKED_NO_EVENT_BINDING`.

A região é comparada com **acento normalizado**, então `Petropolis` e
`Petrópolis` são a mesma região (evita falso mismatch).

## Resultado atual (dados reais)

- 1 auto-rejeitado (evento/patch UNKNOWN);
- 55 candidate-positive auto-validados (Recife, evento `REC_2022_05_24_30`),
  held for overlay;
- 116 blocked (secondary, falta geometria de overlay patch-evento);
- 0 `NEEDS_USER_DECISION`.

A fila "para revisão humana" de 172 itens do v2at foi resolvida autonomamente
em 0 decisões pendentes da usuária.

## Candidate-positive não é label

`AUTO_VALIDATED_CANDIDATE_POSITIVE` significa que a evidência verificável
concorda — **não** que existe label operacional de inundação. Por isso:

- `gt_patch_flood_observed = NA`;
- `allowed_for_training = False`;
- `promotion_to_operational_gt = false`;
- `formal_negatives_created = false`.

O label operacional só pode ser criado por um protocolo formal de
positivo/negativo, com geometria de overlay e negativos comparáveis. Ausência
de evidência nunca virou negativo.

## O que falta para ground truth formal

- Geometria de overlay patch-evento para os candidate-positives (digitalização).
- Protocolo formal de positivo/negativo com negativos comparáveis.
- Adjudicação registrada dos candidate-positives antes de qualquer label.

## Outputs

`local_runs/ground_truth/v2bp/` (11 arquivos `.csv`/`.json`/`.md`, leves).
Nenhum dado bruto, vetor denso ou checkpoint é versionado.
