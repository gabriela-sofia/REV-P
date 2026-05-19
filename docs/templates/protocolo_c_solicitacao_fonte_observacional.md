# Template — Solicitação de fonte observacional para pesquisa acadêmica

**Uso**: Este template deve ser adaptado para cada solicitação formal a órgão público ou técnico. Não enviá-lo diretamente. Substituir todos os placeholders por informações reais antes do envio. Não inserir dados pessoais ou identificadores privados no repositório.

---

## Cabeçalho da solicitação

**Destinatário:** [INSTITUICAO]
**Assunto:** Solicitação de dados para pesquisa acadêmica — Análise de evidências de eventos de inundação/alagamento em [REGIAO]
**Data:** [DATA_DE_ENVIO]

---

## Identificação da pesquisa

A presente solicitação é feita no contexto de pesquisa acadêmica conduzida em [ORIENTADOR_OU_INSTITUICAO]. A pesquisa investiga metodologias de referência para análise estrutural de áreas urbanas com histórico de inundação e alagamento, com foco nas regiões de [REGIAO].

A abordagem metodológica é baseada em sensoriamento remoto, análise estrutural visual e organização de evidências auditáveis. **A pesquisa não pressupõe ground truth observado e não realiza detecção automática de inundação.** O objetivo desta fase é construir um conjunto de referência metodológica que permita avaliar criticamente fontes de evidência observacional.

---

## Dado solicitado

**Tipo de dado:** [TIPO_DE_DADO_SOLICITADO]

**Evento ou período de referência:** [EVENTO_OU_PERIODO]

**Justificativa acadêmica:** [JUSTIFICATIVA_ACADEMICA]

**Formato desejado:**
- [FORMATO_DESEJADO]
- Preferência por formatos abertos: GeoJSON, CSV georeferenciado, GeoTIFF, Shapefile com metadados
- CRS preferencial: SIRGAS2000 (EPSG:4674) ou UTM equivalente

---

## Comprometimentos da pesquisadora

- Os dados serão utilizados exclusivamente para fins acadêmicos e metodológicos
- Dados brutos não serão redistribuídos sem autorização explícita da instituição fornecedora
- A proveniência dos dados será documentada e citada conforme exigência da instituição
- O repositório público do projeto conterá apenas metadados seguros (identificadores, datas, atributos de licença); não conterá dados brutos, geometrias restritas ou conteúdo sensível
- Quaisquer resultados de análise derivados dos dados serão compartilhados com a instituição antes de publicação, se solicitado
- A pesquisa seguirá a legislação aplicável, incluindo normas de proteção de dados e direitos autorais

---

## Contato

**Responsável pela solicitação:** [CONTATO_DA_PESQUISADORA]
**Vínculo institucional:** [ORIENTADOR_OU_INSTITUICAO]

---

## Notas de uso deste template

- Este template é público no repositório REV-P como parte da documentação metodológica
- Não contém dados pessoais, URLs privadas ou informações de solicitações reais
- Deve ser preenchido localmente antes de qualquer envio
- O preenchimento e envio são ações manuais da pesquisadora, não automáticas
- Registrar o status de cada solicitação em `datasets/evidence_acquisition_tracker.csv`
