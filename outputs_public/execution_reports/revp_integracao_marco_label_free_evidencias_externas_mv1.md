# Integracao: marco label-free MV1 + evidencias externas revisadas

> Passada review-only. `pode_virar_label_agora` e sempre `false`. Nenhum evento candidato e promovido automaticamente a label. Evidencia externa permanece como suporte observacional/contextual.

## 1. Artefatos internos do marco label-free

- `revp_fechamento_marco_validacao_label_free_evidencia_estrutural_mv1.md` — encontrado
- `revp_manifesto_marco_validacao_label_free_evidencia_estrutural_mv1.csv` — encontrado
- `revp_fechamento_marco_validacao_label_free_evidencia_estrutural_mv1.json` — encontrado
- `revp_proximos_passos_pos_marco_label_free_mv1.csv` — encontrado

## 2. Artefatos externos originais

- `revp_manifesto_evidencias_externas_mv1.csv` — encontrado
- `revp_auditoria_fontes_externas_mv1.csv` — encontrado
- `revp_indice_eventos_externos_candidatos_mv1.csv` — encontrado
- `revp_indice_geometrias_externas_candidatas_mv1.csv` — encontrado
- `revp_curadoria_evidencias_externas_mv1.md` — encontrado
- `revp_curadoria_evidencias_externas_mv1.json` — encontrado

## 3. Artefatos externos revisados

- `revp_manifesto_evidencias_externas_downloads_mv1.csv`
- `revp_auditoria_fontes_externas_downloads_mv1.csv`
- `revp_indice_eventos_externos_candidatos_downloads_mv1.csv`
- `revp_indice_geometrias_externas_candidatas_downloads_mv1.csv`
- `revp_fechamento_downloads_evidencias_externas_mv1.md`
- `revp_fechamento_downloads_evidencias_externas_mv1.json`
- `revp_log_downloads_evidencias_externas_mv1.csv`

## 4-6. Contagem de downloads

- Baixadas de fato nesta rodada: 1
- Ja existiam localmente: 10
- Continuam sem download (manual/falha/bloqueio): 8

## 7. Por que cada fonte sem download nao foi baixada

- `FONTE_CUR_003` (requer_download_manual): Sem URL direta no log v1if (OBS_CUR_001). Curitiba nunca vira negativo formal por ausencia de evidencia.
- `FONTE_NAC_001` (requer_download_manual): Tambem: GeoRisk (georisk.cemaden.gov.br) e gov.br/cemaden. Suscetibilidade nao e geometria de evento observado.
- `FONTE_NAC_002` (requer_download_manual): Tambem dadosabertos.ana.gov.br. Serie pontual/estacao; nao e geometria de extensao de evento.
- `FONTE_NAC_003` (requer_download_manual): Contexto de cobertura, nao geometria de evento. Verificar versao/licenca antes de uso publico.
- `FONTE_NAC_004` (requer_download_manual): Geometria administrativa/censitaria (contexto), nao geometria de evento observado.
- `FONTE_NAC_005` (requer_download_manual): Boletim/serie regional; suporte contextual de chuva, nao geometria de evento.
- `FONTE_NAC_006` (falha_download): Log v1if (OBS_PET_003): NO_URL, requer solicitacao formal DRM-RJ/NADE.
- `FONTE_INT_001` (requer_download_manual): ID exato da ativacao (EMSR) para Petropolis nao foi confirmado na busca; nao inventar. Produto pode delimitar deslizamento, nao flood.

## 8. Fontes com SHA256 real

- `FONTE_REC_001` — `0f8036b19ec60481...`
- `FONTE_REC_002` — `8e833d449effb837...`
- `FONTE_REC_003` — `e26f80ddada0a6c7...`
- `FONTE_REC_004` — `dd2f31e3f53c3aa9...`
- `FONTE_PET_001` — `45d1a9802c648570...`
- `FONTE_PET_002` — `78a185ba06b6f5d4...`
- `FONTE_PET_003` — `053fa4cbccac09a6...`
- `FONTE_PET_004` — `3399af178d756a19...`
- `FONTE_CUR_001` — `8997b1edd6025e46...`
- `FONTE_CUR_002` — `4db40582b10238be...`
- `FONTE_INT_002` — `a8f1468feb77c412...`

## 9. Fontes com geometria candidata

- `FONTE_REC_001` (Point, EPSG:4326) — overlay bloqueado: geometria_de_suscetibilidade_nao_e_evento_observado
- `FONTE_REC_002` (MultiPolygon, EPSG:4326 (origem EPSG:32725)) — overlay bloqueado: produto_pode_misturar_deslizamento_e_inundacao_revisao_obrigatoria

## 10. Fontes apenas como contexto documental

- `FONTE_REC_003` — registro_de_ocorrencia
- `FONTE_REC_004` — registro_de_solicitacao_urbana
- `FONTE_PET_001` — relatorio_tecnico_pos_desastre
- `FONTE_PET_002` — anexos_avaliacao_pos_desastre
- `FONTE_PET_003` — relatorio_tecnico_bairro
- `FONTE_PET_004` — diario_oficial_decreto
- `FONTE_CUR_001` — noticia_institucional
- `FONTE_CUR_002` — documento_pdf_contextual
- `FONTE_CUR_003` — mapeamento_alagamento
- `FONTE_NAC_001` — mapa_interativo_alertas_suscetibilidade
- `FONTE_NAC_002` — serie_hidrologica
- `FONTE_NAC_003` — uso_cobertura_solo
- `FONTE_NAC_004` — malha_setor_censitario
- `FONTE_NAC_005` — boletim_pluviometrico_hidrologico
- `FONTE_NAC_006` — relatorio_pos_desastre
- `FONTE_INT_001` — mapa_resposta_emergencial
- `FONTE_INT_002` — artigo_revisado_por_pares

## 11. Fontes que podem apoiar revisao humana futura

- `FONTE_REC_001` — geometria candidata para revisao humana
- `FONTE_REC_002` — geometria candidata para revisao humana

## 12. Fontes bloqueadas para label patch-level

- Todas as fontes: nenhuma sustenta label patch-level nesta passada (pode_virar_label_agora=false para todos os itens da tabela integrada).

## 13. Riscos que permanecem

- CRS: geometrias candidatas precisam de CRS confirmado e revisado.
- Licenca: varias fontes com licenca a confirmar permanecem bloqueadas para uso publico direto.
- Circularidade: bases de servico urbano exigem isolamento de fonte independente.
- Landslide vs flood: Petropolis/Charter referem-se a deslizamento; nao confundir com flood extent.
- Fonte textual sem geometria nao fecha patch-level.

## 14. Proximo passo recomendado

Executar manualmente os downloads marcados como requer_download_manual (CEMADEN, ANA/HidroWeb, MapBiomas, IBGE, APAC, Copernicus EMS, IPPUC), salvar o bruto na quarentena e re-rodar este script; e solicitar formalmente as fontes em falha/bloqueio (DRM-RJ). Nenhuma fonte e promovida a label nesta passada.

## Guardrails preservados

- Sem treino supervisionado; sem label binario; sem positivo formal; sem negativo formal.
- Sem ground truth operacional patch-level nesta passada.
- unknown nao vira negativo; ausencia de evidencia nao vira classe 0.
- Curitiba nao vira negativo formal.
- Evidencia externa nao vira label automaticamente.
- DINOv2 nao prova inundacao.
- Fonte textual sem geometria nao fecha patch-level.
- Landslide scar nao prova flood extent.
- Geometria de suscetibilidade nao e geometria de evento observado.
- Download de fonte nao significa validacao operacional.
