# MV2-12 — Data Intake, Evidence Backlog & Download Readiness

**Data:** 2026-06-23
**Branch executada:** `marco/validacao-label-free-evidencia-estrutural-mv1`
**Branch citada na tarefa:** `marco/mv2-11-rebalanceamento-representacional-regional`
(a working tree atual está na branch de validação MV1; nenhuma troca de branch foi feita — ver nota de divergência ao final)

Este marco **não** treina modelo, **não** cria label, **não** cria silver e **não** inicia sandbox.
Ele converte o bloqueio genérico "faltam dados" numa **frente objetiva de dados**: o que existe,
o que falta, o que baixar, o que recuperar localmente, o que digitalizar e o que cada dado destrava.

---

## 1. Alinhamento MV2-10 (auditoria) × MV2-11 (programação)

| Eixo | MV2-10 diagnosticou | MV2-11 confirmou programaticamente | Convergência | Continua bloqueado |
|------|---------------------|-------------------------------------|--------------|--------------------|
| Viés regional | CUR=100% / REC=5.4% / PET=6.25% | 131 candidatos regionais, **0 fortes**, 0 novos canônicos, 0 novos DINOv2 | O gargalo é **dado/vínculo**, não pipeline | Corpus completo segue enviesado (Dia 8) |
| Baseline espectral | 0 GeoTIFF espectral legível | `regional_visual_features` é visual, não espectral | PNG/NPZ/DINOv2 visual ≠ raster Sentinel | Dia 10 |
| Temporalidade/lineage | 0/128 com datetime e scene_id | manifesto temporal segue vazio | Falta scene_id de origem | Dias 2/3, Trilha A |
| Evidência observacional | 9 eventos `geometry=null` | rebalanceamento não cria evidência externa | Geometria de evento ausente | Dia 18, G3/G4 |
| Silver/negativos/splits | formal_silver=0, negativos=0 | `LABEL_FREE_READY_LIMITED` (subset n=6) não libera treino | Sem GT operacional | Dias 19/21/22 |

**Síntese:** a auditoria disse "o bloqueio é dado, não código"; a programação MV2-11 provou isso ao
encontrar 131 candidatos por varredura mas **nenhum** com vínculo `patch_id/asset_id` forte —
DINOv2 total permaneceu 60 e canônicos visuais permaneceram 48.

---

## 2. O que a auditoria diagnosticou × o que a programação confirmou

- **Diagnóstico (MV2-10):** infraestrutura sólida, guardrails corretos; faltam scene_id, PNG canônicos
  REC/PET, footprint vetorial Recife, GeoTIFFs espectrais, geometrias de evento, fontes formais.
- **Confirmação (MV2-11):** varredura regional rendeu 131 candidatos LOW (102 Recife + 29 Petrópolis),
  `can_canonicalize=false` em todos, bloqueador `no_patch_or_asset_contract_link`. O subset balanceado
  mínimo (n=6) é apenas auditoria label-free limitada; o full segue com risco de confounder **ALTO**.

---

## 3. Bloqueios por dia do cronograma (pós-MV2-11)

| Dia | Fase | Status | Dado que destrava | Tipo de aquisição |
|-----|------|--------|-------------------|-------------------|
| 8 | Corpus DINOv2 balanceado | PARCIAL (subset n=6 ok; full enviesado) | PNG canônicos REC (~35) + PET (~45) com vínculo patch_id | Recuperação local (PROJETO) |
| 10 | Baseline espectral | BLOQUEADO | GeoTIFF/JP2/SAFE multibanda + scene_id | Download externo (Copernicus/GEE) |
| 18 | Evidência observacional | BLOQUEADO | Footprint vetorial Recife 2022 + geometrias dos 9 eventos | Digitalização + portal/manual |
| 19 | Silver set | BLOQUEADO | Cascata: positivos formais (depende Dia 18) | — |
| 21 | Splits / negativos formais | BLOQUEADO | Não-ocorrência documentada + campos anti-leakage | Portal/manual + metadados |
| 22 | Sandbox supervisionado | BLOQUEADO | Cascata de todos os anteriores | — |

---

## 4. Dados faltantes (resumo — detalhe em `mv2_12_missing_data_matrix.csv`)

19 itens: **6 P0**, **10 P1**, **3 P2**. Por tipo de aquisição:

### Recuperável localmente (não exige download externo)
- **scene_id** dos 128 exports (histórico de tasks GEE) — destrava datetime, cloud_cover, lineage, Trilha A.
- **acquisition_datetime** e **cloud_cover** por asset (derivam do scene_id).
- **PNG canônicos REC/PET** com vínculo patch_id (workspace PROJETO read-only).
- **Boundaries vetoriais** dos 36 patches Recife restantes (só REC_00019 tem geometria real).

### Exige download externo
- **GeoTIFFs Sentinel-2 L2A** espectrais (bandas B02/B03/B04/B08/B11/B12 + SCL) — Copernicus/GEE.
- **Sentinel-1 SAR** (água sob nuvem).
- **Vetor Charter758/EMSR601** (acesso pago/institucional).
- **MapBiomas raster** por patch (via GEE), **IBGE grade 1km** (licença a reverificar).

### Exige digitalização
- **Footprint de inundação Recife mai/2022** a partir do PNG Charter758 (CRS + incerteza).
- **9 geometrias de evento** do Protocolo C (`geometry=null`; 4 textual, 4 mapa oficial, 1 insuficiente).

### Exige portal/manual (solicitação formal)
- **CEMADEN** (LAI 12.527/2011), **APAC** série por estação, **DRM-RJ** cicatrizes Petrópolis,
  **Defesa Civil Curitiba** (dossiê v2ce pronto), **ANA HidroWeb** (retry estação 39187800).

---

## 5. Diagnóstico do raster espectral nativo (Dia 10)

Varredura MV2-12 real da working tree: **0 GeoTIFF/JP2/SAFE espectral legível**.
A tabela `mv2_12_sentinel_native_raster_backlog.csv` registra, por região, que existe render visual
(PNG), `.npz` visual e embedding DINOv2 visual — mas `has_native_raster=false`,
`available_bands=NONE`, `can_support_spectral_baseline=false` em todas. **Render visual não é
inferido como espectral em nenhum ponto** (trava de teste cobre isso).

---

## 6. Geometria de evento (G4)

`mv2_12_event_geometry_backlog.csv`: 9 eventos, `has_observed_footprint=false`,
`can_support_g4=false`, `can_support_silver=false` em todos. Petrópolis mantida como
`mass_movement` (cohort separado de flood). Município inteiro **não** é geometria de evento.
O caminho imediato é digitalizar o footprint Recife mai/2022 do PNG Charter758 já baixado.

---

## 7. Fontes institucionais/documentais (papel de cada uma)

| Fonte | Serve para |
|-------|-----------|
| Copernicus (S2/S1) | RASTER_SENTINEL_ESPECTRAL, EVIDENCIA_OBSERVACIONAL |
| Copernicus EMS / Charter758 | GEOMETRIA_EVENTO, EVIDENCIA_OBSERVACIONAL (Recife) |
| CEMADEN | EVIDENCIA_OBSERVACIONAL, TEMPORALIDADE (independente) |
| APAC | EVIDENCIA_OBSERVACIONAL, TEMPORALIDADE (Recife) |
| ANA HidroWeb | EVIDENCIA_OBSERVACIONAL hidrométrica (Capibaribe) |
| DRM-RJ | GEOMETRIA_EVENTO (Petrópolis, mass_movement) |
| Defesa Civil REC/CTB | GEOMETRIA_EVENTO, EVIDENCIA_OBSERVACIONAL |
| IBGE | CONTEXTO_TERRITORIAL (confounder demográfico) |
| MapBiomas | CONTEXTO_TERRITORIAL (cobertura do solo) |
| GeoCuritiba/IPPUC | GEOMETRIA_EVENTO (Curitiba) |
| SGB/CPRM, PE3D/MDE | CONTEXTO_TERRITORIAL, REPRODUTIBILIDADE |

---

## 8. Riscos científicos se avançar sem esses dados

- Tratar **PNG/NPZ/DINOv2 visual como espectral** → claim de baseline espectral inválido (Dia 10).
- Usar **cidade como label** ou o subset enviesado como corpus → confounder ALTO mascara qualquer sinal.
- Tratar **ausência de evidência como negativo** → negativos formais espúrios (Dia 21).
- Promover **candidato regional/IA a canônico/silver** sem vínculo `patch_id/asset_id` forte.
- Digitalizar **município inteiro como footprint** → falso positivo geométrico no overlay (G3/G4).

---

## 9. Nota de divergência de branch

A tarefa cita a branch `marco/mv2-11-rebalanceamento-representacional-regional`, mas a working tree
está em `marco/validacao-label-free-evidencia-estrutural-mv1`. Como as regras proíbem troca de branch,
o MV2-12 foi produzido na branch atual. Os outputs do MV2-11 foram lidos diretamente da sua branch
(via `git show`, sem checkout); a auditoria MV2-10 está presente localmente em
`outputs_public/audits/mv2_10_gap_audit/`. Há ainda uma branch homônima de escopo distinto
(`marco/mv2-12-reconstrucao-espectral-sentinel-baseline`); por isso este marco usa o diretório
próprio `outputs_public/mv2_data_readiness/` para não colidir.

---

## Artefatos deste marco

- `MV2_12_DATA_READINESS_REPORT.md` (este)
- `MV2_12_EXECUTIVE_SUMMARY.md`
- `mv2_12_missing_data_matrix.csv`
- `mv2_12_download_readiness.csv`
- `mv2_12_local_recovery_candidates.csv`
- `mv2_12_event_geometry_backlog.csv`
- `mv2_12_sentinel_native_raster_backlog.csv`
- `mv2_12_data_readiness_summary.json`
- `commands.txt` (reprodução)
