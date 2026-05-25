# Protocolo C v1jk - Sandbox review-only

A v1jk e uma etapa local de engenharia exploratoria. Ela usa os 9 anchors oficiais com Sentinel-2, DEM e DINO frozen para entender o comportamento estrutural do lote no espaco de features.

## Escopo

A etapa cria uma tabela local com:

- diagnosticos DINO pre/pos;
- deltas espectrais quando os rasters locais estao disponiveis;
- DEM, slope e aspect;
- disponibilidade S1;
- QA Sentinel-2.

Essas features ajudam a revisar separabilidade estrutural e consistencia entre anchors. Elas nao criam classe, label ou negativo.

## Prototipos

A v1jk calcula ranking de mudanca estrutural, distancias entre diagnosticos pareados e uma projecao PCA. Grupos exploratorios, quando gerados, sao apenas agrupamentos de revisao. Eles nao sao classes.

## One-class sandbox

Se `sklearn` estiver disponivel, a etapa roda um sandbox one-class leve apenas para testar plumbing. O resultado tem status `INVALID_FOR_SCIENTIFIC_CLAIM`.

O sandbox:

- nao salva pesos;
- nao treina modelo cientifico;
- nao cria label operacional;
- nao cria negativo formal;
- nao descongela DINO.

## Boundary

O resultado cientificamente valido continua sendo o material review-only: anchors oficiais, QA multimodal e DINO frozen. O sandbox serve apenas para engenharia.

Treino real continua bloqueado ate existir protocolo formal de labels, negativos ou controles com evidencia de ausencia, split auditavel, controle de vazamento e validacao independente.
