# Protocolo C — plano de aquisição de evidências observacionais

## 1. Motivação

O Protocolo C estabelece a hierarquia de fontes, define os gates (G1–G9) e os bloqueios necessários para uma decisão auditada sobre promoção ou rejeição de candidatos a ground reference. 

Esta etapa organiza como buscar, documentar e priorizar evidências observacionais reais que possam fechar as lacunas de:
- **evento** (confirmação, data, geometria);
- **fonte** (disponibilidade, originalidade, documentação);
- **temporalidade** (compatibilidade temporal entre evento e observação);
- **espacialidade** (compatibilidade espacial, resolução, CRS, escala);
- **força metodológica** (observação de campo, mapa oficial, produto operacional, contexto);
- **revisão humana** (auditoria independente, corroboração cruzada);
- **decisão de promoção** (apta ou bloqueada).

O REV-P não possui ground truth operacional observado. O objetivo desta etapa é transformar a metodologia em um plano de aquisição real — registrando fontes-alvo por região, priorizando as que podem fechar gates específicos, documentando a força e limitação de cada fonte, e deixando claro o que não pode (e nunca poderá) ser feito sem referência independente validada.

---

## 2. O que esta etapa busca

A aquisição de evidências observacionais para ground reference busca por:

1. **Eventos confirmados**
   - Data precisa
   - Geometria aproximada (ponto, polígono)
   - Fonte de confirmação (notícia, relatório oficial, registro de Defesa Civil)

2. **Mapas oficiais observados**
   - Produtos de mapeamento de inundação pós-evento
   - Documentação da metodologia e acurácia
   - Data de levantamento
   - Área de cobertura

3. **Produtos operacionais com incerteza documentada**
   - GFM/CEMS (Copernicus)
   - Estimativas de extensão com intervalos de confiança
   - Disclaimers sobre limitações algorítmicas
   - Acurácia esperada vs. observação

4. **Imagens pós-evento de alta resolução**
   - Sentinel-2, Landsat, ou RapidEye pós-evento
   - Data de aquisição conhecida
   - Cobertura de nuvem aceitável
   - Anotação humana estruturada (quando disponível)

5. **Anotações humanas/especializadas**
   - Marcações de hidrógrafos
   - Anotações de especialistas em sensoriamento remoto
   - Documentação do critério de marcação
   - Independência da anotação (sem viés de modelo)

6. **Registros de Defesa Civil**
   - Relatórios de evento
   - Áreas afetadas mapeadas
   - Datas de início/fim
   - Responsáveis e contatos

7. **Bases municipais e estaduais**
   - Cadastros de áreas de risco
   - Histórico de eventos
   - Dados de drenagem, hidrografia, topografia
   - CRS e metadados conhecidos

8. **Relatórios técnicos**
   - Análises pós-evento de órgãos de pesquisa (CPRM, INPE, universidades)
   - Documentação de metodologia e fonte de dados
   - Reconhecimento de limitações
   - Acesso e reprodutibilidade

9. **Fontes acadêmicas e datasets públicos**
   - Sen1Floods11, Kuro Siwo, UFO
   - Datasets de treinamento com anotação verificada
   - Documentação de acurácia e validação cruzada
   - Aplicabilidade metodológica ao REV-P

---

## 3. O que esta etapa não faz

Esta etapa é explicitamente **metadata-only** e **bloqueada de**:

- ❌ **Declarar ground truth operacional** — nenhuma fonte operacional algorítmica (GFM, DINO, produto modelado) se torna verdade observacional por conversão automática.

- ❌ **Criar label supervisionado** — não há criação de rótulos para treino de modelo.

- ❌ **Treinar modelo** — nenhuma rede neural é retreinada ou adaptada com base nesta etapa.

- ❌ **Executar Protocolo B** — não há detecção, segmentação ou predição de inundações.

- ❌ **Transformar DINO em evidência observacional** — embeddings estruturais DINO servem apenas para suporte metodológico de revisão, nunca como referência independente.

- ❌ **Avançar para multimodal** — multimodal permanece em **hold** até que a camada de ground reference esteja metodologicamente fechada e validada.

- ❌ **Baixar raster, copiar dados pesados ou executar pipelines espaciais pesados** — apenas planejamento e documentação.

---

## 4. Fontes-alvo por região

### 4.1 Recife

**Defesa Civil**
- Coordenadoria Municipal de Proteção e Defesa Civil (COMPDEC)
- Relatórios de evento 2021, 2022
- Áreas afetadas, datas, geometrias
- Contacto: COMPDEC Recife

**Prefeitura e órgãos municipais**
- Secretaria de Gestão da Cidade
- PE3D (plataforma de dados geoespaciais de Pernambuco)
- Camadas de drenagem, hidrografia, topografia
- Histórico de inundações

**SGB/CPRM/RIGeo**
- Serviço Geológico do Brasil (CPRM)
- Relatórios pós-evento de eventos de 2021, 2022
- Análise geomorfológica
- Dados públicos em RIGeo

**Sentinel-1/Sentinel-2 pós-evento**
- Imagens Sentinel-2 pós-evento 2021, 2022
- Análise visual de mudança de radiância
- Data precisa de aquisição
- Cobertura de nuvem

**GFM/CEMS (Copernicus)**
- Produtos de mapeamento de rápida resposta
- Limitações algorítmicas documentadas
- Data de processamento
- Intervalo de confiança de extensão

**Produtos acadêmicos locais**
- UFPE, UFRPE: análises de inundação 2021, 2022
- Dados de sensores de precipitação
- Modelos hidrológicos locais
- Datasets de validação cruzada

---

### 4.2 Petrópolis

**Defesa Civil**
- Defesa Civil Municipal
- Registros de evento fevereiro 2022 (principal)
- Áreas críticas mapeadas
- Geometria de deslizamento e inundação

**Prefeitura e órgãos municipais**
- Secretaria de Defesa Civil
- Camadas de hidrografia, declividade, ocupação urbana
- Histórico de eventos 2010, 2022

**SGB/CPRM/RIGeo**
- Relatório pós-evento CPRM fevereiro 2022
- Análise geomorfológica de deslizamento
- Relação entre evento de chuva e falhas de encosta
- Dados de geologia estrutural

**Sentinel-1/Sentinel-2 pós-evento**
- Imagens pós-fevereiro 2022
- Sentinel-1 pode detectar mudança de superfície pós-deslizamento
- Sentinel-2 para validação visual
- Nuvens frequentes, mas cobertura possível

**GFM/CEMS (Copernicus)**
- Produtos CEMS pós-fevereiro 2022 se disponíveis
- Detecção de áreas inundadas em vales
- Precisão limitada por resolução em encostas

**Produtos acadêmicos locais**
- UFRJ (Laboratório de Meteorologia): análise de chuva extrema
- Dados meteorológicos horários
- Modelos de infiltração e escorregamento
- Validação de cenários

---

### 4.3 Curitiba

**Defesa Civil**
- Defesa Civil Municipal
- Registros de eventos 2022, 2023
- Áreas de alagamento (urbano, não encosta)
- Geometrias de áreas críticas

**Prefeitura e órgãos municipais**
- GeoCuritiba (portal de dados geoespaciais)
- Camadas de rede pluvial, drenagem, topografia
- Mapeamento de áreas de risco
- CRS SIRGAS2000, metadados disponíveis

**SGB/CPRM/RIGeo**
- Dados de geologia superficial
- Contexto hidrogeomorfológico
- Análise de suscetibilidade (não é ground truth, mas contexto)

**Sentinel-1/Sentinel-2 pós-evento**
- Imagens Sentinel-2 pós-evento 2022, 2023
- Análise de mudança radiométrica em áreas urbanas
- Data de aquisição conhecida
- Resolução 10 m adequada para áreas alagadas urbanas

**GFM/CEMS (Copernicus)**
- Produtos CEMS pós-evento disponíveis
- Detecção de áreas alagadas urbanas com melhor acurácia
- Limitações em áreas com cobertura urbana densa

**Produtos acadêmicos locais**
- UFPR, PUC-PR: análises de drenagem urbana
- Modelos hidrológicos urbanos
- Dados de estações meteorológicas
- Validação de cenários de precipitação extrema

**GeoCuritiba / Portal municipal**
- Dados de operação de drenagem em tempo real (histórico)
- Registros de ativação de válvulas de proteção
- Relatórios de limpeza de canais pós-evento

---

## 5. Força metodológica das fontes

Cada tipo de fonte tem força, limitação e aplicabilidade específicas.

### Observação de campo

**Força:**
- Direta, independente de sensoriamento remoto
- Documenta inundação efetiva em tempo real
- Pode validar extensão e profundidade

**Limitação:**
- Cobertura limitada espacialmente
- Não sistemática
- Depende de acesso e segurança

**O que pode sustentar:**
- G1_EVENT_CONFIRMATION (com data e local)
- G8_INDEPENDENT_CORROBORATION

**O que não pode sustentar:**
- Cobertura spatial completa (sem mapa oficial associado)
- Detecção em áreas inacessíveis

---

### Mapa oficial observado

**Força:**
- Cobertura geográfica completa (ou documentada)
- Metodologia publicada
- Autor responsável (agência, pesquisa)
- Incerteza esperada reportada

**Limitação:**
- Pode ser produto de algoritmo/modelo (como GFM)
- Requer acesso ao método e dados originais para auditoria
- Temporal: data fixa, não atualização contínua

**O que pode sustentar:**
- G2_SOURCE_AVAILABILITY
- G3_TEMPORAL_ALIGNMENT
- G4_SPATIAL_ALIGNMENT
- G5_SOURCE_STRENGTH
- G6_UNCERTAINTY_AND_LIMITATIONS
- Parcialmente G7_HUMAN_REVIEW (se anotação humana foi parte do método)
- G8_INDEPENDENT_CORROBORATION (como corroboration, não como única fonte)

**O que não pode sustentar:**
- Verdade operacional automática sem validação externa

---

### Imagem de alta resolução anotada

**Força:**
- Pixel-level information
- Data exata de aquisição
- Visibilidade visual de mudança

**Limitação:**
- Anotação pode ter viés, erros sistemáticos
- Requer especialista qualificado
- Método de anotação deve ser documentado

**O que pode sustentar:**
- G3_TEMPORAL_ALIGNMENT (data fixa)
- G4_SPATIAL_ALIGNMENT (resolução espacial)
- G5_SOURCE_STRENGTH (se anotação independente, não de modelo)
- G8_INDEPENDENT_CORROBORATION (como validação cruzada)

**O que não pode sustentar:**
- Verdade operacional sem metodologia de anotação clara e auditada

---

### Produto operacional algorítmico (GFM, CEMS, produto modelado)

**Força:**
- Cobertura rápida, grande escala
- Operacional, disponível em tempo quase-real
- Útil para contexto e complementação

**Limitação:**
- Incerteza inerente ao algoritmo
- Não é observação, é estimativa
- Acurácia varia por região, contexto, condições

**O que pode sustentar:**
- G2_SOURCE_AVAILABILITY (como cobertura)
- G6_UNCERTAINTY_AND_LIMITATIONS (documentar limitações)
- Parcialmente G5_SOURCE_STRENGTH (como produto, não como verdade)

**O que não pode sustentar:**
- G1_EVENT_CONFIRMATION (sem triangulação com fonte observacional)
- G8_INDEPENDENT_CORROBORATION (não é independente do sensor/algoritmo)
- **Nunca**: verdade operacional sozinho

---

### Mapa modelado/suscetibilidade

**Força:**
- Contexto hidrogeomorfológico
- Baseado em princípios físicos
- Reproduzível de dados públicos

**Limitação:**
- Não é observação de evento específico
- Prediz onde pode haver inundação, não onde houve
- Acurácia local desconhecida

**O que pode sustentar:**
- G6_UNCERTAINTY_AND_LIMITATIONS (documentar que é modelo, não observação)
- Contexto metodológico (não métrica de validação)

**O que não pode sustentar:**
- Qualquer gate de verdade observacional
- Confirmação de evento específico

---

### Contexto hidrogeomorfológico (CPRM, SGB)

**Força:**
- Baseado em dados físicos estruturais
- Validado por geólogos
- Útil para explicar por que inundação ocorreu (não que ocorreu)

**Limitação:**
- Não específico para evento
- Não substitui observação
- Não detecta limite de inundação

**O que pode sustentar:**
- Contexto de interpretação
- Corroboration indireta (se a localização de inundação é consistente com geomorfologia)

**O que não pode sustentar:**
- Confirmação de evento
- Validação de extensão

---

### Suporte estrutural DINO (review-only)

**Força:**
- Consistência estrutural visual
- Detecta mudança visual entre antes/depois
- Reproduzível, determinístico

**Limitação:**
- Não é observação
- Não mede inundação especificamente
- Não pode ser interpretado como evidência independente

**O que pode sustentar:**
- Suporte metodológico de revisão humana
- Visualização para auditoria (não como métrica de validação)

**O que não pode sustentar:**
- **Nunca**, sozinho: confirmação, ground truth, validação
- Qualquer gate de evidência observacional

---

## 6. Prioridade de aquisição

Prioridades refletem utilidade metodológica imediata para fechar gates do Protocolo C.

### HIGH

Evento confirmado + data + geometria aproximada + fonte oficial/anotada/operacional forte + cobertura espacial conhecida.

**Exemplo:**
- Recife 2021, 2022: evento confirmado, data, Defesa Civil registra, Sentinel-2 disponível pós-evento, GFM/CEMS pode confirmar.
- Petrópolis fevereiro 2022: evento extremo confirmado (escala nacional), data precisa, CPRM relatório pós-evento, Sentinel-1/2 pós-evento, múltiplas fontes.

---

### MEDIUM

Evento confirmado + data, mas geometria/cobertura ainda pendente.

**Exemplo:**
- Curitiba 2022, 2023: eventos confirmados, data conhecida, mas cobertura de Sentinel pode ser limitada por nuvem; GFM pode estar disponível mas precisa verificação.
- Recife: relatórios históricos, mas data/geometria precisa requer consultoria com Defesa Civil ou CPRM.

---

### LOW

Contexto geral sem evento específico.

**Exemplo:**
- Dados topográficos, hidrografia geral, mapas de suscetibilidade regional — suportam interpretação, não validação de evento específico.

---

### METHOD_REFERENCE_ONLY

Datasets ou produtos usados como referência metodológica, sem aplicação direta aos patches REV-P atuais.

**Exemplo:**
- Sen1Floods11, Kuro Siwo, UFO — mostram como outros projetos construíram ground reference; aplicáveis para desenho de aquisição futura, não para validação imediata.

---

## 7. Relação com gates do Protocolo C

O Protocolo C define 9 gates. Esta etapa mapeia quais fontes podem fechar quais gates.

| Gate | Descrição | Fontes que podem contribuir | Força esperada |
|------|-----------|---------------------------|----------------|
| **G1_EVENT_CONFIRMATION** | Evento ocorreu, data/local | Defesa Civil, relatórios CPRM, notícias com data/geometry, observação de campo | Oficial ou observacional |
| **G2_SOURCE_AVAILABILITY** | Dados/fontes estão disponíveis | Inventário de fontes por região, acesso documentado | Qualquer fonte listada e acessível |
| **G3_TEMPORAL_ALIGNMENT** | Data da observação alinha com evento | Sentinel pós-evento, GFM data de processamento, anotação com data | Referência temporal clara |
| **G4_SPATIAL_ALIGNMENT** | Geometria, CRS, resolução apropriados | Imagem de alta resolução anotada, mapa oficial com CRS, Sentinel-2 10 m | Resolução ≥ 10 m, CRS documentado |
| **G5_SOURCE_STRENGTH** | Fonte é independente, não algorítmica | Observação de campo, anotação humana independente, mapa oficial anotado | Humana, não derivada de modelo |
| **G6_UNCERTAINTY_AND_LIMITATIONS** | Incerteza documentada, limitações claras | GFM com disclaimer, produto operacional com intervalo de confiança, anotação com critério declarado | Métrica de acurácia ou intervalo |
| **G7_HUMAN_REVIEW** | Auditoria humana independente | Especialista em sensoriamento remoto, hidrólogo, geógrafo | Documentação de revisão |
| **G8_INDEPENDENT_CORROBORATION** | Múltiplas fontes independentes convergem | 2+ de: Defesa Civil, relatório técnico, imagem anotada, observação de campo | Consistência entre fontes |
| **G9_PROMOTION_DECISION** | Decisão auditable de aptidão | Todas as fontes avaliadas, lacunas documentadas, decisão documentada | Resultado de auditoria |

---

## 8. Critério para passar ao multimodal

Multimodal só deve ser retomado quando:

1. **Camada de ground reference metodologicamente fechada**
   - Pelo menos um evento por região com evidência observacional validada
   - Metodologia de validação documentada e reproduzível
   - Gaps conhecidos e explícitos

2. **Lacunas por região documentadas**
   - Quais gates estão fechados, quais continuam abertos
   - Quais fontes foram adquiridas, quais permanecem como FUTURE_ACQUISITION
   - Riscos metodológicos identificados

3. **Aquisição futura planejada**
   - Roadmap de aquisição para fechar gaps restantes
   - Responsáveis identificados (CPRM, prefeitura, universidade)
   - Timeline e recurso estimados

4. **Claims permitidos/proibidos claramente definidos**
   - Multimodal só pode fazer: fusão de dados, análise estrutural, suporte de revisão
   - Multimodal nunca pode fazer: detecção de inundação, predição, label supervisionado, declaração de ground truth

5. **Protocolo B permanece bloqueado até referência forte real**
   - Detecção de inundação só com ground reference validada
   - Predição só após validação metodológica cruzada
   - Sem exceções

---

## Referências

- Matgen et al. (2011). Towards an automated SAR-based flood monitoring system. Remote Sensing of Environment.
- Tellman et al. (2021). Satellite imaging reveals increased proportion of population exposed to floods. Nature.
- Chini et al. (2019). Rapid Damage Mapping for the 2018 Sulawesi Earthquake and Tsunami using Synthetic Aperture Radar Data.
- CPRM/SGB documentation on post-event geological assessment.
- Copernicus Emergency Management Service (CEMS) methodology and disclaimers.

## Etapas subsequentes

A triagem de eventos candidatos (v1hn) e os dossiês de evidência (v1ho) complementam este plano com camadas operacionais mais específicas: a triagem prioriza eventos por região e conecta-os ao backlog de fontes; os dossiês especificam o pacote mínimo de evidências por evento, os requisitos críticos e as decisões de continuidade. Ground truth operacional, Protocolo B e multimodal permanecem bloqueados/hold. Veja [`protocolo_c_triagem_eventos_candidatos.md`](protocolo_c_triagem_eventos_candidatos.md) e [`protocolo_c_dossies_eventos_candidatos.md`](protocolo_c_dossies_eventos_candidatos.md).

