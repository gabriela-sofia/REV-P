# v2bu — Auditoria de sensibilidade de overlay com geometrias alternativas

Versão: `v2bu`
Modo: auditoria geométrica autônoma. Não cria label, não cria negativo, não libera treino.

## 1. Por que o v2bu existe

O v2bt produziu 5 geometrias alternativas QA-only do evento `REC_2022_05_24_30`
a partir dos pontos Defesa Civil. O v2bu **não escolhe uma como verdade**. Ele
reexecuta o overlay de cada patch boundary recuperada contra cada alternativa e
mede quão estável é a compatibilidade de cada patch entre métodos de
reconstrução.

A pergunta técnica não é "quais patches são positivos", e sim "quais patches
mostram compatibilidade geométrica robusta com a geometria QA-only do evento sob
diferentes reconstruções dos pontos Defesa Civil".

## 2. Como usa as geometrias alternativas QA-only do v2bt

Escopo `37_RETRIED_PATCHES` (36 boundaries recuperadas + REC_00019) × 5
alternativas (convex hull, buffered union 250/500, cluster envelopes c0/c1) =
185 overlays pairwise. Cada alternativa preserva `can_use_for_formal_gt=false` e
`can_create_label=false`.

## 3. Por que múltiplas geometrias = análise de sensibilidade

Um único overlay contra uma geometria frágil é instável. Usar várias
reconstruções transforma isso em um teste de sensibilidade: um patch que só
intersecta a geometria mais permissiva (convex hull / buffer maior) é evidência
mais fraca do que um patch que intersecta reconstruções apertadas e largas ao
mesmo tempo.

## 4. O que significam robusto, method-dependent, buffer-only e noncompatible

- **`QA_COMPATIBLE_ROBUST`**: intersecta ≥3 alternativas, ≥2 famílias de método
  (hull/buffer/cluster), incluindo uma geometria "tight" (buffer≤250 ou cluster).
- **`QA_COMPATIBLE_METHOD_DEPENDENT`**: intersecta algumas alternativas sem
  consenso robusto/tight (ex.: só hull + buffer maior).
- **`QA_COMPATIBLE_BUFFER_ONLY`**: intersecta apenas buffered unions.
- **`QA_NOT_COMPATIBLE_ACROSS_ALTERNATIVES`**: não intersecta nenhuma alternativa
  válida.

Resultado atual: 1 robusto (REC_00276), 1 method-dependent (REC_00299), 0
buffer-only, 35 não-compatíveis.

## 5. Por que QA-compatible ainda não é label

Uma interseção QA apenas diz que uma patch boundary se sobrepõe a uma geometria
de evento QA-only derivada de pontos. Não é label de inundação.
`gt_patch_flood_observed=NA`, `allowed_for_training=False`. Mesmo
`ready_for_formal_gt_review=True` é só uma fila de revisão futura, não um label.

## 6. Por que no-intersection ainda não é negativo formal

A geometria alternativa é QA-only e derivada de pontos. Uma não-interseção não
pode virar `gt_patch_flood_observed=0`. Ausência nunca vira negativo.

## 7. O que falta para abrir protocolo formal de GT

- Validação formal do footprint do evento (geometria oficial revisada).
- Protocolo formal de positivo.
- Negativos formais comparáveis.

## 8. Por que treino segue bloqueado

`labels_created=false`, `allowed_for_training_count=0`,
`promotion_to_operational_gt=false`. Uma sonda de sensibilidade geométrica sobre
geometrias QA-only não cria label nem desbloqueia treino.

## Outputs

`local_runs/ground_truth/v2bu/` (11 arquivos `.csv`/`.json`/`.md`, leves).
Nenhum raster, shapefile bruto ou arquivo pesado é gravado ou versionado.
