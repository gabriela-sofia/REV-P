# Protocolo C — Diagnóstico de Dados Externos Válidos

## 1. Objetivo

Este documento transforma a pesquisa de eventos observados candidatos em uma lista operacional de dados externos necessários por região. Para cada região do REV-P, o diagnóstico especifica exatamente quais documentos, mapas, séries e registros precisam ser trazidos manualmente para que um evento observado candidato possa avançar de evidência documental para ground reference.

O diagnóstico não executa aquisição. Não baixa arquivos. Não acessa portais. É um roteiro metodológico de priorização — o que buscar, de quem, em qual formato e com qual nível de detalhe.

---

## 2. O que pode ser considerado dado externo válido

Um dado externo é válido para fins metodológicos do REV-P quando satisfaz pelo menos um dos seguintes critérios:

- **Decreto oficial**: ato administrativo que declara situação de emergência ou calamidade, com data, localidade e número de ato rastreável
- **Boletim de Defesa Civil**: comunicado periódico com registro de ocorrências, área afetada, pontos de alagamento ou ações de resposta
- **Relatório técnico**: documento produzido por órgão técnico especializado com metodologia explícita, autoria institucional e data — laudos geológicos, relatórios de vistoria, diagnósticos de bacia
- **Mapa oficial observado**: mapa produzido por órgão público ou técnico com representação de área afetada, cobertura identificada e data de levantamento
- **Shapefile ou GeoJSON institucional**: geometria vetorial produzida ou publicada por órgão público ou técnico, com CRS documentado e fonte rastreável
- **Lista de ocorrências**: registro tabular ou descritivo de pontos, endereços, bairros ou localidades afetadas, com data e fonte institucional
- **Pontos de alagamento**: localização georreferenciada ou descritiva de pontos de acumulação de água, com data e fonte
- **Dados pluviométricos institucionais**: série de precipitação de estação identificada, com coordenadas, período e órgão responsável
- **Produto operacional com incerteza**: produto de detecção algorítmica (como Copernicus GFM) com metadados de confiança, data de produto e limitações documentadas
- **Foto oficial com legenda e localização**: imagem publicada por órgão público com legenda que identifique o local, data e contexto
- **Artigo acadêmico com metodologia clara**: publicação revisada por pares que analisa o evento com método explícito, fontes primárias identificadas e limitações documentadas
- **Base municipal, estadual ou federal rastreável**: sistema ou portal de dados governamentais com licença identificada e data de acesso

---

## 3. O que não basta sozinho

Os seguintes elementos não são suficientes isoladamente para fechar gates de ground reference ou sustentar evidência observacional:

- **Notícia sem fonte oficial primária**: cobertura jornalística pode ser pista de busca, não evidência primária
- **Chuva acumulada sem impacto observado**: dado pluviométrico sem registro de ocorrência ou área afetada não confirma inundação
- **NDWI, NDBI ou índice espectral isolado**: produto derivado de Sentinel não é observação direta — é inferência que exige referência independente
- **Embedding DINO ou cluster**: representação visual de encoder congelado não é observação, não fecha gate de evento, temporalidade, espacialidade ou ground truth
- **Mapa modelado sem observação de campo**: mapa de suscetibilidade, risco genérico ou produto de simulação não equivale a mapa de inundação observada
- **Suscetibilidade genérica**: cartografia de risco ou suscetibilidade prévia ao evento não confirma que o evento ocorreu naquela localidade
- **Descrição sem data**: relato sem período identificável não pode fechar G3
- **Descrição sem localidade**: relato sem referência espacial mínima não pode fechar G4 em triagem

---

## 4. Recife — dados externos prioritários

### Diagnóstico

O evento REC_2022_05_24_30 (fim de maio de 2022) tem fonte primária municipal forte — múltiplas publicações da Prefeitura do Recife, situação de emergência declarada, ações de mutirão documentadas. O que falta é a camada espacial concreta: geometria de áreas afetadas, lista oficial de pontos de alagamento, base de ocorrências sistematizada.

### Dados prioritários

| Item | Instituição | Formato esperado | Modo de acesso |
|------|-------------|-----------------|----------------|
| Decreto nº 35.669/2022 (situação de emergência) | Prefeitura do Recife / DOM | PDF ou texto | Portal público ou Diário Oficial Municipal |
| Boletins da Defesa Civil (28–31/05/2022) | COMPDEC Recife | PDF, tabela | Solicitação formal ou portal |
| Base/lista das 115 ocorrências críticas | COMPDEC Recife | Tabela, CSV, lista | Solicitação formal |
| Pontos de alagamento CTTU/SUSi/registros operacionais | CTTU, SUSi | CSV, GeoJSON, lista | Solicitação formal ou portal |
| Localização dos abrigos temporários ativados | Prefeitura / Assistência Social | Tabela | Solicitação formal ou portal |
| Lista de comunidades afetadas (Jardim Uchôa, Areias, Jardim Monte Verde, Milagres, Sítio dos Macacos, CAIC Barro) | COMPDEC / boletins | Texto, lista | Já parcialmente em fontes municipais |
| Dados CEMADEN/APAC para a janela 24–30/05/2022 | CEMADEN, APAC | CSV, NetCDF | Portal público CEMADEN |
| Delimitação do Rio Tejipió e áreas de transbordamento | PE3D, APAC, prefeitura | Shapefile, GeoJSON | Portal PE3D / solicitação formal |
| Fotos oficiais com legenda e URL | Prefeitura do Recife | JPEG com metadados | Já publicadas no portal da prefeitura |

### Status

Recife tem evento observado forte (G1/G2/G3 fechados documentalmente). A lacuna principal é a camada espacial concreta — geometria, pontos de alagamento georreferenciados e base de ocorrências — necessária para avançar de referência observacional candidata para ground reference. Sem essa camada, nenhum overlay patch-level pode ser executado.

---

## 5. Petrópolis — dados externos prioritários

### Diagnóstico

O evento PET_2022_02_15 (15 de fevereiro de 2022) é o mais forte do corpus — relatório técnico DRM-RJ/NADE/Thalweg existente, Copernicus Image of the Day publicada, artigo NHESS revisado por pares. O desafio metodológico de Petrópolis é a separação de fenômenos: o desastre de fevereiro/2022 envolveu deslizamentos de massa, inundações e enxurradas de forma sobreposta. Qualquer uso operacional exige separação rigorosa entre esses processos.

### Dados prioritários

| Item | Instituição | Formato esperado | Modo de acesso |
|------|-------------|-----------------|----------------|
| Relatório técnico DRM-RJ / NADE / Thalweg completo | DRM-RJ | PDF | Portal público DRM-RJ |
| Relatório técnico SGB/CPRM (se aplicável ao evento de fev/2022) | SGB/CPRM | PDF | Portal RIGeo / solicitação |
| Shapefiles ou GeoJSONs de feições de deslizamento de deslizamento | DRM-RJ, CPRM, defesa civil | Shapefile, GeoJSON | Solicitação formal |
| Geometria de alcances de inundação/enxurrada (separada de deslizamento) | DRM-RJ, INEA | Shapefile, GeoJSON | Solicitação formal |
| Áreas de risco remanescente pós-evento | DRM-RJ, Defesa Civil | Shapefile, GeoJSON | Solicitação formal ou portal |
| Relatórios individuais de vistoria de imóveis | Defesa Civil Municipal, DRM-RJ | PDF, tabela | Solicitação formal |
| Decreto de calamidade pública / emergência | Prefeitura de Petrópolis | PDF ou texto | Portal público |
| Dados CEMADEN/INEA de precipitação (14–16/02/2022) | CEMADEN, INEA | CSV | Portal público CEMADEN |
| Lista de localidades afetadas com separação de fenômeno | DRM-RJ / boletins / mídia oficial | Texto, tabela | Relatório técnico já disponível |
| Fotos oficiais ou mapas técnicos com localização | DRM-RJ, Defesa Civil | JPEG + legenda, PDF mapa | Relatório técnico já disponível |

### Status

Petrópolis é a região com maior potencial para ground reference espacial forte, especialmente para PET_2022_02_15. O relatório técnico DRM-RJ está publicamente disponível e contém mapeamento técnico. A lacuna principal é a separação metodológica entre inundação/transbordamento e deslizamento — o REV-P foca em inundação e alagamento, e qualquer uso operacional de dados de Petrópolis exige essa separação antes do overlay patch-level.

---

## 6. Curitiba — dados externos prioritários

### Diagnóstico

O evento CTB_2023_10_28_30 (outubro de 2023) é o mais forte de Curitiba — fonte municipal disponível, publicações de prefeitura documentando atendimento a famílias e limpeza pós-evento, referências ao Caximba e Tatuquara. Os eventos de Curitiba têm boa documentação temporal mas carecem de geometria concreta de área afetada.

### Dados prioritários

| Item | Instituição | Formato esperado | Modo de acesso |
|------|-------------|-----------------|----------------|
| Boletins/ocorrências da Defesa Civil (28–30/10/2023) | Defesa Civil Curitiba | PDF, tabela | Solicitação formal ou portal |
| Relatório do evento Caximba/Tatuquara | Defesa Civil / regionais | PDF, tabela | Solicitação formal |
| Registros de famílias atendidas/retiradas (versão pública) | Defesa Civil / FAS | Tabela | Solicitação formal (versão sem dado pessoal) |
| Fotos oficiais com legenda e localização | Prefeitura de Curitiba | JPEG + legenda | Já publicadas parcialmente no portal |
| Dados Simepar/Defesa Civil de precipitação (28–30/10/2023) | Simepar, Defesa Civil | CSV | Portal Simepar / solicitação |
| Mapas de áreas alagáveis e microdrenagem urbana | Ippuc, SMMA, GeoCuritiba | Shapefile, GeoJSON | Portal GeoCuritiba |
| Histórico de inundação por bacia | SMOP, Defesa Civil | Tabela, shapefile | Solicitação formal |
| Registros FAS/regionais (se versão pública disponível) | FAS Curitiba | Tabela | Solicitação formal |
| Ruas/localidades afetadas: Caximba, Cajuru, Miguel Pedro Abib, Alice Vilas Boas | Boletins, portal prefeitura | Texto, lista | Já parcialmente em fontes municipais |

### Status

Curitiba tem evento observado candidato com boa documentação temporal. A lacuna principal é a georreferenciamento — os pontos e localidades mencionados nas fontes municipais precisam ser convertidos em geometria para overlay patch-level futuro. Boletins e ocorrências da Defesa Civil são a prioridade de aquisição.

---

## 7. Onde os dados brutos devem ficar

Se um dado externo for adquirido localmente, deve ser armazenado exclusivamente em:

```
local_only/
└── protocolo_c/
    └── evidencias_observacionais/
        ├── recife/
        │   ├── 2022_05/
        │   ├── 2023_02/
        │   └── 2024_06/
        ├── petropolis/
        │   ├── 2022_02/
        │   ├── 2022_03/
        │   └── 2024_03/
        └── curitiba/
            ├── 2022_01/
            ├── 2023_10/
            └── 2024_02/

local_runs/
└── ground_reference_audit/
```

**Essas pastas não devem ser versionadas.** Nenhum arquivo dessas pastas deve entrar em `git add` ou `git commit`. O `.gitignore` bloqueia `local_only/` e `local_runs/`.

---

## 8. O que vai para o GitHub

O repositório público pode conter apenas metadados seguros sobre os dados externos:

- identificador da fonte (`source_id`, `observed_event_id`)
- URL pública da fonte
- data de acesso (quando aplicável)
- nome e tipo do arquivo local (sem path privado)
- tipo de licença ou status de licença
- status de aquisição local (`NOT_ACQUIRED`, `LOCAL_ONLY_REQUIRED`, `PUBLIC_REVIEW_REQUIRED`)
- nível de precisão espacial e temporal
- gates que a fonte pode fechar
- decisão metodológica sobre a fonte

---

## 9. O que não vai para o GitHub

Nunca versionar:

- raster pesado (GeoTIFF, TIF, TIFF, VRT)
- shapefile bruto (SHP, DBF, SHX, PRJ, CPG)
- geodatabase (GDB, GPKG)
- arquivo ZIP ou RAR de dado geoespacial
- base restrita ou com licença não verificada
- dado pessoal (nome, CPF, endereço de pessoa física)
- foto sem licença clara ou com dados pessoais visíveis
- PDF com redistribuição não explicitamente pública
- path privado do workspace local
- embedding (NPZ, NPY)
- qualquer arquivo de `local_only/` ou `local_runs/`
