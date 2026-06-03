# Protocolo C — Busca Externa e Solicitação Regional de Evidências

## 1. Motivação

As etapas anteriores do Protocolo C organizaram o que buscar e o que é necessário:

- **v1hl**: plano de aquisição de evidências observacionais — fontes-alvo por região, prioridades, força metodológica
- **v1hm**: pacote operacional de aquisição — intake, licença, proveniência, staging local
- **v1hn**: triagem de eventos candidatos por região — status, prioridade e backlog de fontes
- **v1ho**: dossiês de evidência por evento candidato — requisitos mínimos, lacunas e decisões de continuidade

A etapa v1hp transforma os dossiês de evidência em planos concretos de busca externa e solicitação por região. Para cada evento candidato e cada gate do Protocolo C com lacuna, esta etapa organiza as perguntas que precisam ser respondidas, as fontes que precisam ser consultadas, os pacotes de solicitação que precisam ser preparados e a prioridade com que isso deve ocorrer.

Esta etapa não executa a busca. Prepara as condições para que a busca e a solicitação sejam executadas de forma rastreável e auditável.

---

## 2. O que esta etapa faz

- Organiza perguntas de busca por região e por gate do Protocolo C
- Define fontes-alvo com modo de acesso (portal público ou solicitação formal)
- Cria pacotes de solicitação estruturados por região, evento e gate
- Registra prioridade de busca e bloqueadores ativos
- Conecta cada busca externa aos gates G1–G9
- Prepara a aquisição futura com rastreabilidade de licença e proveniência
- Documenta o que deve permanecer local-only se adquirido

---

## 3. O que esta etapa não faz

- **Não executa busca**: nenhum portal é consultado, nenhuma fonte é acessada
- **Não confirma evento**: eventos permanecem como EVENT_SEARCH_TARGET ou PENDING_SOURCE_REVIEW
- **Não declara ground truth operacional**: nenhum evento tem ground truth estabelecido
- **Não cria label supervisionado**: nenhum pixel é rotulado
- **Não treina modelo**: nenhum pipeline de aprendizado é executado
- **Não baixa raster**: nenhum GeoTIFF, shapefile, ZIP ou dado pesado é adquirido
- **Não executa Protocolo B**: multimodal, treinamento e predição permanecem bloqueados
- **Não avança multimodal**: multimodal permanece em hold
- **Não usa DINO como confirmação de evento**: embeddings DINO são review-only e não fecham gates G1–G9

---

## 4. Fontes externas por região

### Recife

**Registros de Defesa Civil — COMPDEC Recife**
Fonte-alvo para G1 (confirmação de evento), G2 (disponibilidade de fonte), G4 (alinhamento espacial) e G7 (revisão supervisora). Acesso via solicitação formal. Cobre eventos de 2021 e 2022. Licença UNKNOWN — deve ser clarificada antes de qualquer uso. Bundling de 2021 e 2022 em uma solicitação é estratégia recomendada.

**SGB/CPRM — Portal RIGeo**
Fonte-alvo para G1 (confirmação geológica), G5 (força de fonte observada/especializada), G6 (limitações e incerteza) e G8 (corroboração independente). Acesso público via portal RIGeo. Licença UNKNOWN — verificar termos de reutilização. Possível laudo pós-evento 2021.

**PE3D — Plataforma Geoespacial de Pernambuco**
Fonte de contexto espacial (G4 — baseline). Drenagem, hidrografia, topografia. Não é evidência de evento observado. Não fecha G1 nem G4 como evento. Portal público — verificar licença.

**Sentinel-2 L2A — Copernicus Data Space**
Fonte para G3 (alinhamento temporal). Acesso público, licença Copernicus Open. Dado bruto deve permanecer local-only. Período: maio–julho 2021, 2022. Cobertura de nuvem deve ser verificada.

**Copernicus GFM/CEMS**
Produto operacional com incerteza (G2, G4, G6). Produto algorítmico — não declara ground truth sozinho. Incerteza deve ser documentada. Licença pública mas redistribuição restrita.

### Petrópolis

**SGB/CPRM — Portal RIGeo**
Fonte-alvo prioritária para G1 e G8. CPRM publicou laudo pós-evento de fevereiro de 2022. Acesso público. Licença UNKNOWN — verificar termos. Separação entre inundação e deslizamento é requisito obrigatório ao usar este laudo.

**Defesa Civil Municipal Petrópolis**
Fonte para G1, G4, G7 e G8. Acesso via solicitação formal. Evento de fevereiro 2022 tem alta visibilidade institucional — probabilidade de documentação disponível é maior. Separação inundação-deslizamento obrigatória na revisão.

**Sentinel-2 L2A — Copernicus Data Space**
Fonte para G3. Período: fevereiro 2022. Risco de cobertura de nuvem significativo em fevereiro. SAR (Sentinel-1) pode ser necessário como complemento. Licença Copernicus Open.

### Curitiba

**Defesa Civil Municipal Curitiba**
Fonte-alvo para G1, G4 e G7. Acesso via solicitação formal. Defesa Civil Curitiba tem registros bem organizados. Bundling de 2022 e 2023 em uma solicitação. Licença UNKNOWN — clarificar uso acadêmico.

**GeoCuritiba — Portal GIS Municipal**
Fonte de contexto espacial baseline (rede pluvial, drenagem, topografia). Não é evidência de evento observado — não fecha G4 como evento. Acesso público — verificar licença. Contribui para entendimento territorial mas não para confirmação de evento.

**Sentinel-2 L2A — Copernicus Data Space**
Fonte para G3. Contexto urbano com boa cobertura Sentinel. Licença Copernicus Open.

---

## 5. Perguntas de busca por gate

As perguntas abaixo guiam a busca de evidências para cada gate do Protocolo C. Elas não têm resposta nesta etapa — são instrumentos de planejamento.

### G1 — EVENT_CONFIRMATION
- Existe confirmação oficial ou documental de que o evento ocorreu na data e local indicados?
- Qual instituição registrou o evento?
- O relatório ou comunicado inclui data, localização e tipo de evento?
- Para Petrópolis: o documento distingue inundação de deslizamento?

### G2 — SOURCE_AVAILABILITY
- Existe fonte pública, portal ou produto operacional com cobertura do evento?
- A fonte está disponível para acesso sem solicitação formal?
- A fonte tem licença compatível com uso acadêmico?

### G3 — TEMPORAL_ALIGNMENT
- A fonte tem data compatível com a janela temporal do evento candidato?
- Existe imagem Sentinel-2 ou produto com data dentro de 14 dias do evento?
- A cobertura de nuvem é aceitável para a data disponível?

### G4 — SPATIAL_ALIGNMENT
- Existe geometria, polígono, mapa ou dado espacial da área afetada?
- O dado espacial tem CRS documentado?
- A área afetada tem sobreposição com os patches do corpus REV-P?
- A sobreposição foi ou pode ser verificada por revisor humano?

### G5 — SOURCE_STRENGTH
- A fonte é observação direta, documentação oficial, anotação especializada, produto operacional algorítmico ou apenas modelagem contextual?
- A força metodológica da fonte é suficiente para o gate que se quer fechar?

### G6 — UNCERTAINTY_AND_LIMITATIONS
- A fonte documenta resolução, escala, cobertura, limitações de método?
- O produto operacional (ex: GFM) inclui metadados de confiança ou incerteza?
- A incerteza está registrada de forma que possa ser reportada na cadeia metodológica?

### G7 — REVIEW_GATE
- A fonte pode ser enviada para revisão por pesquisador ou especialista?
- Quem seria o revisor adequado (geomorfólogo, engenheiro hidráulico, especialista em Sentinel)?
- O resultado da revisão pode ser documentado no registry de revisão supervisora?

### G8 — INDEPENDENT_CORROBORATION
- Existe segunda fonte independente que confirme o evento?
- As duas fontes são metodologicamente independentes?

### G9 — PROMOTION_DECISION
- Todos os gates G1–G8 foram satisfeitos com evidência documentada e revisão supervisora?
- A decisão de promoção pode ser registrada formalmente?

---

## 6. Pacotes de solicitação

Os pacotes organizam o que precisa ser solicitado por região e por evento candidato:

**REQUEST_CIVIL_DEFENSE_RECORDS**
Solicitação de registros de Defesa Civil: relatórios de ocorrências, mapas de área afetada, dados de registro de danos. Exige solicitação formal. Licença UNKNOWN por padrão.

**REQUEST_MUNICIPAL_GIS_LAYER**
Solicitação de camadas GIS municipais: hidrografia, drenagem, topografia, rede viária. Pode ser acesso público via portal. Não é evidência de evento observado.

**REQUEST_OFFICIAL_FLOOD_MAP**
Solicitação de mapa oficial de inundação: laudo geológico, shapefile de área afetada, produto operacional com metodologia documentada. CPRM/SGB, CEMS/GFM. Licença e incerteza devem ser documentadas.

**REQUEST_TECHNICAL_REPORT**
Solicitação de laudo técnico: CPRM, IPT, instituição estadual. Pode incluir análise de causa, geometria de ocorrência, dados de campo.

**REQUEST_EVENT_OCCURRENCE_RECORDS**
Solicitação de registros de ocorrência: atas, comunicados, boletins de ocorrência. Pode complementar confirmação de evento (G1).

**REQUEST_LICENSE_CLARIFICATION**
Solicitação de esclarecimento de licença: o que pode ser usado, em que formato, com que restrições. Precede qualquer uso de dado de licença UNKNOWN.

**PUBLIC_PORTAL_REVIEW_PACKAGE**
Pacote para revisão de portal público: Copernicus Data Space, RIGeo CPRM, PE3D, GeoCuritiba. Acesso sem solicitação formal — mas requer documentação de licença.

**METHOD_REFERENCE_PACKAGE**
Referência metodológica de datasets externos (Sen1Floods11, Kuro Siwo, UFO, Copernicus GFM): informa o design do Protocolo C sem substituir evidência local. Não aplicável diretamente a patches/eventos do REV-P.

**Solicitação não equivale a evidência recebida.** Um pacote de solicitação preparado documenta a intenção e os termos da busca — não garante que a fonte estará disponível, acessível ou com licença compatível. A cadeia de evidências começa quando a fonte é efetivamente recebida, registrada e triada.

---

## 7. Priorização regional

**HIGH**
Fonte que pode fechar gate de evento (G1), data (G3) ou geometria (G4) com vínculo possível com patches do corpus. Fonte oficial, documentada ou operacional com incerteza documentada. Requer ação imediata de busca ou solicitação.

**MEDIUM**
Fonte que pode fechar evento ou contexto, mas sem geometria verificada. Pode ser acesso público sem necessidade de solicitação formal. Requer ação após conclusão das buscas HIGH.

**LOW**
Fonte contextual ou de referência espacial sem capacidade de fechar gate de evento observado. Contribui para entendimento territorial mas não para promoção de referência.

**BLOCKED**
Fonte sem acesso, sem licença identificada, com restrição de sensibilidade ou sem relação clara com patches do corpus. Não adianta iniciar busca até bloqueio ser resolvido.

**METHOD_REFERENCE_ONLY**
Dataset acadêmico ou referência externa que informa o design metodológico sem ser aplicado diretamente a eventos/patches do REV-P. Não fecha gates.

---

## 8. Saída da etapa

- `datasets/regional_external_search_plan.csv` — plano de busca regional: metas, fontes-alvo, modo de acesso, prioridade e status por entrada
- `datasets/source_request_package_registry.csv` — pacotes de solicitação por região, evento e gate
- `datasets/gate_search_question_registry.csv` — perguntas de busca por gate e região, sem resposta nesta etapa
- `datasets/regional_request_priority_matrix.csv` — matriz de prioridade regional com bloqueadores e próximos passos

Esses quatro registros são metadata-only. Nenhuma busca foi executada. Nenhuma solicitação foi enviada. Nenhum dado foi adquirido.

A etapa v1hq inicia a primeira camada documental de eventos observados candidatos com base nos planos de busca estruturados aqui: 9 eventos (3 por região) com G1/G2/G3 fechados documentalmente e G4 em triagem espacial. Ground truth operacional não está estabelecido. Protocolo B permanece bloqueado. Multimodal permanece hold. Os dados externos brutos que precisam ser trazidos manualmente estão catalogados em `datasets/manual_external_evidence_needed_registry.csv`. Veja [`protocolo_c_referencias_observacionais_candidatas.md`](protocolo_c_referencias_observacionais_candidatas.md).

A etapa v1hr prepara as condições para patch-linking: geocodificação manual de 22 localidades, janelas temporais Sentinel metadata-only e registro de 48 dependências metodológicas. Veja [`protocolo_c_pre_ligacao_evento_patch.md`](protocolo_c_pre_ligacao_evento_patch.md).
