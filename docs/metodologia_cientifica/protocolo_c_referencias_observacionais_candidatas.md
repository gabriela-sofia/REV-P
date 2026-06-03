# Protocolo C — Referências Observacionais Candidatas

## 1. Motivação

As etapas anteriores do Protocolo C organizaram progressivamente a estrutura de busca e planejamento de aquisição de evidências observacionais:

- **v1hl**: plano de aquisição de evidências observacionais — fontes-alvo por região, prioridades, força metodológica e readiness regional
- **v1hm**: pacote operacional de aquisição — intake, licença, proveniência e staging local
- **v1hn**: triagem de eventos candidatos por região — status, prioridade de busca, backlog de fontes e escopo por patch
- **v1ho**: dossiês de evidência por evento candidato — requisitos mínimos, lacunas e decisões de continuidade
- **v1hp**: busca externa e solicitação regional — planos de busca, pacotes de solicitação formal, perguntas por gate e matriz de prioridade

A etapa **v1hq** transforma esse planejamento acumulado na primeira camada documental efetiva de eventos observados candidatos. Ela não executa busca, não baixa dados e não declara ground truth operacional. Ela organiza, por região, os eventos para os quais existe evidência documental rastreável — data ou janela temporal, fonte primária identificada e localidade mínima — e registra as lacunas que impedem avanço para os níveis seguintes.

Esta etapa é a primeira em que o Protocolo C pode dizer: _"há evidência documental de que este evento ocorreu, há fonte rastreável, há data"_. Ela não pode dizer: _"há ground truth operacional"_, _"há label"_, _"há validação patch-level"_.

---

## 2. O que esta etapa prova

A v1hq pode provar, para cada evento registrado:

- existência documental do evento — fonte primária pública identificada e acessível
- fonte primária ou secundária rastreável — URL, instituição, tipo de documento
- data ou janela temporal clara — pelo menos SHORT_WINDOW (dois a cinco dias) ou MULTI_DAY_WINDOW rastreável
- localidade mínima — município, bairro, rua, corredor de rio ou mapa técnico
- elegibilidade para revisão de fonte — o evento pode avançar para aquisição de fonte, se necessário
- lacunas metodológicas para futura ligação patch-evento

A v1hq **não prova**:

- que o evento afetou os patches do corpus Sentinel no nível pixel
- que há cobertura Sentinel sem nuvens sobre o evento
- que há geometria de área afetada georreferenciada e compatível com o corpus
- que há licença/proveniência adequada para uso operacional
- que há revisão supervisora ou especialista da evidência
- que há corroboração independente suficiente
- que há ground truth operacional
- que há label de inundação
- que há validade para treino supervisionado

---

## 3. Diferença entre os níveis de evidência

### Evento observado candidato

Evento documentado por pelo menos uma fonte primária rastreável, com data ou janela temporal identificada e localidade mínima (município ou mais detalhado). Pode ter G1, G2 e G3 fechados em nível documental. Ainda sem vínculo patch-level validado. Não é sinônimo de ground reference nem de ground truth.

### Referência observacional candidata

Evento observado com fonte rastreável, temporalidade documentada e localização de triagem espacial suficiente para avançar para revisão de fonte e possível ligação evento-patch no futuro. G4 pode estar parcialmente fechado em nível de triagem (município/bairro/rua/corredor/mapa técnico) — não é overlay patch-level. A designação como referência observacional candidata é um passo metodológico intermediário, não uma promoção operacional.

### Ground reference

Referência espacial e temporal mais forte, construída a partir de evidência observacional direta (laudo técnico, mapa observado, produto operacional com incerteza documentada), com licença/proveniência verificada, revisão supervisora ou especialista e compatibilidade com os patches do corpus confirmada por overlay real. O REV-P ainda não possui ground reference formalizada. A construção de ground reference exige gates G1–G9 do Protocolo C satisfeitos ou explicitamente avaliados.

### Ground truth operacional

Estado final de validação, bloqueado no projeto atual. Exige evidência observacional forte em nível patch-level, compatibilidade espacial e temporal confirmada, revisão por pesquisador ou especialista, licença/proveniência completa e decisão auditada de promoção. **Ground truth operacional não está estabelecido no REV-P.** Nenhuma etapa atual declara ground truth operacional.

### Label de treino

Anotação binária ou multi-classe associada a um patch ou pixel, usada em treinamento supervisionado. Exige, além de ground reference validada, definição de negativos confiáveis, splits temporais ou espaciais para controle de leakage, balanceamento entre classes e protocolo de validação supervisionada independente. **Nenhum label de treino foi criado no REV-P.** A v1hq não cria label e não é pré-condição suficiente para criação de label.

**Regra metodológica central:**
> Todo ground truth operacional pode ser tratado como referência, mas nem toda referência pode ser promovida a ground truth, e nenhuma referência vira label de treino sem protocolo supervisionado específico posterior.

---

## 4. Gates fechados nesta etapa

Os gates do Protocolo C que podem ser fechados em nível documental pela v1hq:

### G1 — EVENT_CONFIRMATION

Pode ser fechado em nível documental quando há fonte primária rastreável que confirma a ocorrência do evento — decreto, boletim, comunicado oficial, laudo técnico ou notícia institucional com data e localidade. Fechamento documental não equivale a confirmação de impacto patch-level.

### G2 — SOURCE_AVAILABILITY

Pode ser fechado quando há pelo menos uma fonte com URL pública e acessível, ou documento institucional identificado e rastreável. Fechamento não implica que o dado bruto foi adquirido localmente ou que sua licença foi verificada para uso operacional.

### G3 — TEMPORAL_ALIGNMENT

Pode ser fechado quando há data ou janela temporal documentada por fonte primária — suficiente para identificar o período de busca de asset Sentinel. Fechamento em nível documental não garante que o asset Sentinel correspondente tem baixa cobertura de nuvem ou está disponível no workspace local.

### G4 — SPATIAL_ALIGNMENT_TRIAGE

Nesta etapa, G4 é fechado apenas em nível de triagem espacial — município, bairro, logradouro, corredor de rio ou área descrita em laudo técnico. **Não é overlay patch-level.** O fechamento de G4 em triagem significa que há evidência textual ou cartográfica de que o evento ocorreu na mesma área geográfica geral dos patches do corpus — não que a sobreposição espacial foi computada ou verificada pixel a pixel. G4 pode ser PARTIAL quando a localização é regional demais para triagem municipal ou quando a fonte descreve área maior que a do corpus.

---

## 5. Gates que permanecem abertos após v1hq

Os seguintes gates permanecem abertos e não foram fechados nesta etapa:

- **G4 — SPATIAL_ALIGNMENT** em nível patch-level: overlay espacial real entre geometria do evento e patches do corpus
- **G5 — SOURCE_STRENGTH**: força metodológica da fonte — observação direta vs. produto algorítmico
- **G6 — UNCERTAINTY_AND_LIMITATIONS**: documentação completa de incerteza da fonte
- **G7 — REVIEW_GATE**: revisão supervisora ou especialista da evidência concreta
- **G8 — INDEPENDENT_CORROBORATION**: corroboração por segunda fonte independente
- **G9 — PROMOTION_DECISION**: decisão formal de promoção a referência operacional

Além dos gates, permanecem abertos:

- aquisição local de fonte bruta quando a licença permitir
- verificação de licença e proveniência para uso operacional
- overlay patch-level real
- georreferenciamento de áreas afetadas
- vinculação evento-patch no corpus DINO

---

## 6. Papel das fontes externas

### Fonte oficial municipal

Decreto, boletim, comunicado, portal de notícias da prefeitura ou documento da Defesa Civil municipal. Pode fechar G1, G2 e G3 em nível documental. Para G4 e G5, exige que a fonte contenha geometria, lista de localidades, mapa ou laudo com cobertura espacial rastreável.

### Fonte oficial estadual

Laudo técnico de órgão estadual (DRM-RJ, SGB/CPRM, INEA, APAC), decreto estadual, produto do sistema estadual de monitoramento. Força metodológica alta para G1, G5 e G8 — especialmente quando produzida por pesquisadores de campo.

### Relatório técnico

Laudo geológico, relatório de vistoria, estudo de danos ou documento técnico com metodologia explícita. Força alta para G5 e G6 quando documenta incerteza e metodologia. Exige separação cuidadosa de fenômenos (inundação, deslizamento, enxurrada, transbordamento) antes de uso operacional.

### Produto Copernicus (GFM, CEMS)

Produto operacional algorítmico com incerteza documentada. Pode apoiar G2 e G4 em triagem, mas é produto modelado — não observação direta. **Não declara ground truth sozinho.** Incerteza e metadados de confiança devem ser documentados antes de qualquer uso operacional.

### Artigo acadêmico

Publicação revisada por pares que analisa o evento. Pode apoiar G1, G5, G6 e G8. Metodologia deve ser separada de dados observacionais primários — o artigo cita fontes que precisam ser adquiridas separadamente.

### Notícia local

Cobertura jornalística sem fonte oficial primária. **Não é fonte primária isolada.** Pode ser usada como apoio para contextualização ou como pista de busca de fonte oficial, nunca como evidência principal para fechamento de gate.

### Base ou plataforma institucional

Portal GIS municipal, sistema de monitoramento, base de ocorrências. Força variável dependendo de quem mantém, metodologia de coleta e atualização. Exige verificação de licença e proveniência antes de qualquer uso operacional.

---

## 7. Papel do DINO e do Sentinel nesta etapa

### Sentinel

Imagens Sentinel-2 ou Sentinel-1 podem apoiar análise temporal e visual futura, sujeita à disponibilidade de asset no workspace local, cobertura de nuvem e alinhamento temporal com o evento. O Sentinel não é ground truth sozinho — é sensor que registra reflectância. A interpretação de inundação a partir de imagem Sentinel exige referência observacional independente, separação de confundidores e revisão supervisora. Não é executado nenhum pipeline Sentinel nesta etapa.

### DINOv2

Os embeddings DINO são review-only. O DINOv2 pode apoiar revisão estrutural de patches para comparação visual ou triagem, mas não fecha nenhum gate do Protocolo C. Em particular:

- Embedding não é label
- Cluster DINO não é classe de inundação
- Similaridade DINO não é confirmação de evento
- DINOv2 não fecha G1, G2, G3, G4, G5, G6, G7, G8 ou G9
- `dino_usage_status=SUPPORT_ONLY` para todos os eventos

---

## 8. Relação com Protocolo B e multimodal

O **Protocolo B** (treinamento supervisionado, predição, pipeline multimodal) permanece **BLOCKED** nesta etapa e em todo o estado atual do projeto. A v1hq não cria as condições necessárias para reabertura do Protocolo B — que exigiria ground reference validada, labels curados, protocolo supervisionado específico e decisão metodológica explícita.

O pipeline **multimodal** permanece em **HOLD**. A combinação de modalidades (Sentinel visual + SAR + fontes GIS + DINO) não está autorizada nesta etapa.

A v1hq prepara evidência documental para futura ground reference — não para treino, não para predição e não para reabertura de qualquer protocolo bloqueado.

---

## 9. Saída da etapa

A v1hq produz os seguintes artefatos auditáveis:

- `datasets/observed_event_reference_candidate_registry.csv` — 9 eventos observados candidatos (3 por região) com gates G1–G4 avaliados, status de ground truth e bloqueadores
- `datasets/observed_event_reference_gap_registry.csv` — lacunas metodológicas por evento: o que falta para avançar à ligação patch-evento e ground reference
- `datasets/observed_event_reference_decision_registry.csv` — decisão metodológica por evento: nível de decisão, razão, próximo passo e o que está bloqueado
- `datasets/manual_external_evidence_needed_registry.csv` — inventário de dados externos necessários por região: o que buscar manualmente, em qual formato, de qual instituição e com qual modo de aquisição
- `docs/metodologia_cientifica/protocolo_c_diagnostico_dados_externos_validos.md` — diagnóstico por região dos dados externos que precisam ser trazidos manualmente para avançar
- `docs/templates/protocolo_c_intake_fonte_observacional_manual.md` — template de intake manual de fonte observacional
- `docs/templates/protocolo_c_revisao_evento_observado.md` — template de revisão supervisora de evento observado
- `tests/test_revp_v1hq_observed_event_reference_candidate_audit.py` — testes de auditoria da camada v1hq

**Nenhum dado bruto foi baixado. Nenhum raster, shapefile, GeoTIFF ou arquivo pesado foi versionado. Dados externos brutos, quando adquiridos no futuro, devem permanecer em `local_only/` ou `local_runs/` e nunca ser commitados.**

---

## 10. Próxima etapa

A etapa v1hr prepara as condições para patch-linking sem executá-lo: organiza escopos de preflight evento–patch, alvos de geocodificação manual por localidade, janelas temporais Sentinel metadata-only e dependências metodológicas a resolver antes de qualquer overlay real. Veja [`protocolo_c_pre_ligacao_evento_patch.md`](protocolo_c_pre_ligacao_evento_patch.md).
