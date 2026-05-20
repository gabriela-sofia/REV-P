# Protocolo C — Ficha de Geocodificação Manual

## Identificação

- **geocoding_target_id**: [GEOCODING_TARGET_ID]
- **observed_event_id**: [OBSERVED_EVENT_ID]
- **Região**: [REGIAO]
- **Data de preenchimento**: [DATA_DE_PREENCHIMENTO]
- **Revisor (função)**: [REVISOR_FUNCAO]

---

## 1. Localidade

- **Nome da localidade**: [LOCALIDADE]
- **Conforme citado na fonte**: [LOCALIDADE_CONFORME_FONTE]
- **Tipo de localidade**: [TIPO_DE_LOCALIDADE]
  - Controles: MUNICIPALITY / NEIGHBORHOOD / COMMUNITY / STREET / RIVER / DISTRICT / TECHNICAL_AREA / OCCURRENCE_POINT / UNKNOWN

---

## 2. Fonte

- **Nome da fonte**: [FONTE]
- **URL**: [URL]
- **Data de acesso**: [DATA_DE_ACESSO]
- **Tipo de documento**: [TIPO_DE_DOCUMENTO]
- **Evidência descrita na fonte**: [EVIDENCIA_DESCRITA]

---

## 3. Geometria

- **Tipo de geometria esperada**: [GEOMETRIA_ESPERADA]
  - Controles: POINT / LINE / POLYGON / AREA_APPROXIMATION / UNKNOWN
- **Fonte da geometria**: [FONTE_DA_GEOMETRIA]
  - Exemplos: shapefile IBGE, portal GEORecife, IPPUC, base ANA hidrografia, mapa técnico DRM-RJ
- **URL da fonte de geometria**: [URL_FONTE_GEOMETRIA]
- **Data de acesso à geometria**: [DATA_ACESSO_GEOMETRIA]
- **CRS esperado**: [CRS_ESPERADO]
  - Exemplo: SIRGAS 2000 (EPSG:4674); UTM zona 24S (EPSG:31984)
- **Coordenada ou polígono**: [COORDENADA_OU_POLIGONO]
  - ATENÇÃO: não preencher com coordenada estimada por buscador online sem verificação oficial; não preencher com centroide de município como representação de área afetada específica; não preencher com geometria aproximada de mapa genérico
  - Se coordenada não foi obtida de fonte oficial: deixar em branco e registrar status NOT_GEOCODED

---

## 4. Precisão espacial

- **Precisão espacial estimada**: [PRECISAO_ESPACIAL_ESTIMADA]
  - Controles: HIGH / MEDIUM / LOW / UNKNOWN
- **Justificativa de precisão**: [JUSTIFICATIVA_PRECISAO]

---

## 5. Licença e redistribuição

- **Licença da fonte de geometria**: [LICENCA]
- **Pode redistribuir**: [PODE_REDISTRIBUIR]
  - Se NÃO: geometria deve permanecer em `local_only/` — não versionar no repositório público
- **Revisão de licença concluída**: [REVISAO_LICENCA_CONCLUIDA]
  - Controles: SIM / NAO / PENDENTE

---

## 6. Status de geocodificação

- **geocoding_status**: [GEOCODING_STATUS]
  - Controles: NOT_GEOCODED / NEEDS_MANUAL_REVIEW / NEEDS_OFFICIAL_GEOMETRY / BLOCKED_PENDING_SOURCE / METHOD_REFERENCE_ONLY
- **Pode suportar patch-linking após revisão**: [PODE_SUPORTAR_PATCH_LINKING]
  - Controles: true / false
- **Esta geocodificação estabelece ground truth operacional sozinha**: NUNCA — cannot_establish_ground_truth_alone é sempre true

---

## 7. Decisão de uso

- **Decisão**: [DECISAO_DE_USO]
  - Opções: GEOCODIFICACAO_CONCLUIDA_PENDENTE_REVISAO_HUMANA / NECESSITA_FONTE_OFICIAL / NECESSITA_CONFIRMACAO_CRS / BLOQUEADO_POR_LICENCA / BLOQUEADO_SEM_EVIDENCIA_ESPACIAL
- **Pode entrar em patch-linking após revisão humana**: [PODE_ENTRAR_EM_PATCH_LINKING]
  - Controles: true / false

---

## 8. O que esta geocodificação não pode afirmar

- [NAO_PODE_AFIRMAR]
- Sempre incluir: ground truth operacional estabelecido; flood detection confirmado; flood prediction; flood label criado; training label criado; supervised training autorizado; patch validado como inundado; relação patch-evento confirmada; geometria aproximada equivale a geometria oficial

---

## 9. Próxima ação

- **Próxima ação**: [PROXIMA_ACAO]
- **Dependências abertas**: [DEPENDENCIAS_ABERTAS]
  - Listar dependency_ids do patch_linking_dependency_registry.csv que esta geocodificação contribui para resolver

---

## 10. Observações

[OBSERVACOES]
