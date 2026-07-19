# MV2-12 — Sumário Executivo

**Data:** 2026-06-23 | **Branch executada:** `marco/validacao-label-free-evidencia-estrutural-mv1`

O MV2-12 transforma "faltam dados" numa matriz objetiva: 19 itens faltantes
(6 P0, 10 P1, 3 P2), 15 fontes de download mapeadas, 9 eventos sem geometria,
3 regiões sem raster espectral nativo e 464 candidatos locais indexados.

---

## O que o MV2-11 resolveu
- Provou empiricamente que o gargalo é **dado/vínculo**, não pipeline.
- Produziu subset balanceado mínimo (n=6) para auditoria label-free **limitada**.
- Mapeou 131 candidatos regionais por varredura (102 Recife + 29 Petrópolis).

## O que o MV2-11 **não** resolveu
- 0 candidatos fortes com vínculo `patch_id/asset_id`; 0 novos canônicos; 0 novos DINOv2.
- DINOv2 total seguiu 60; canônicos visuais seguiram 48; corpus completo segue enviesado (ALTO).
- Nenhum dado espectral, temporal, de geometria ou observacional foi adquirido.

## Quais dados faltam primeiro (P0)
1. **scene_id** dos 128 exports GEE → destrava datetime, cloud_cover, lineage, Trilha A.
2. **acquisition_datetime** + **cloud_cover** por asset (derivam do scene_id).
3. **PNG canônicos REC (~35) e PET (~45)** com vínculo patch_id → corpus balanceado (Dia 8).
4. **GeoTIFFs Sentinel espectrais** (bandas + SCL) → baseline espectral (Dia 10).

## Quais downloads são realmente prioritários
- Apenas **1** fonte é "baixar agora" e é leve/pública: **ANA HidroWeb** (retry estação 39187800).
- Todo o resto exige **scene_id** (Sentinel-2/S1), **LAI** (CEMADEN), **licença a reverificar** (IBGE)
  ou **acesso pago** (vetor EMS). Nenhum bruto pesado deve ir para `outputs_public`.

## O que recuperar localmente **antes** de baixar qualquer coisa
- **scene_id** do histórico de tasks GEE (sem isso, downloads Sentinel não são auditáveis por patch).
- **PNG canônicos REC/PET** do workspace PROJETO (resolve o viés sem download externo).
- **Boundaries vetoriais** dos 36 patches Recife restantes (só REC_00019 está fechado).

## Quais gates continuam bloqueados
- **G1** (lineage/espectral), **G2/G3** (geometria/overlay), **G4** (evento), **G5/G6** (silver/negativos),
  **G7** (split anti-leakage). `can_train=false`, `sandbox=bloqueado`, `ground_truth_operacional=ausente`,
  `labels_created=0` — todos preservados.

## Próxima ação recomendada
**Recuperar o scene_id (GEE) e os PNG canônicos REC/PET localmente, antes de qualquer download externo.**
Essas duas ações locais destravam, em conjunto, o Dia 8 (corpus balanceado) e a Trilha A
(datetime+cloud_cover+lineage) — a maior alavanca por menor custo de aquisição.
