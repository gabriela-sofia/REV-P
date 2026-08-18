# Auditoria de reprodutibilidade externa — marco MV1

> Auditoria externa de realidade, review-only. Verifica se dados, URLs, hashes, arquivos baixados, bloqueios técnicos e claims públicos são defensáveis para uma pessoa externa. Não cria label, não altera a decisão review-only, não libera treino. Bloqueadores são exclusivamente técnicos/metodológicos — **licença não é tratada como limitação**: os dados são públicos/institucionais.

## 1. Escopo

Esta auditoria cobre as fontes externas navegadas e baixadas no marco MV1 (artefatos `*_navegacao_mv1`, índice de arquivos baixados, log de navegação/downloads e integração com o marco). Foram verificados, com ferramentas reais nesta passada:

- existência, tamanho e SHA256 recalculado dos 8 arquivos em quarentena local (`local_only/evidencias_externas_quarentena/`);
- parseabilidade de GeoJSON (JSON + contagem de feições), PDFs (assinatura `%PDF`/`%%EOF`) e XLSX (abertura como pacote OOXML);
- disponibilidade externa atual de 10 fontes (URLs de origem e de download) via HEAD/GET leve, sem baixar massa pesada;
- ausência de bruto pesado em `outputs_public/`.

O hardening de reprodutibilidade do Codex ainda não existe no repositório no momento desta auditoria; será reauditado quando for criado.

## 2. Diferença entre auditabilidade e reprodutibilidade

**Auditável** = a proveniência está registrada e conferível: URL oficial, hash, tamanho, formato, instituição, data de acesso. O marco já é auditável.

**Reproduzível por pessoa externa** = alguém fora do projeto consegue obter o mesmo arquivo e conferir o hash sem acesso privilegiado. Isso depende de a URL ainda responder e de o arquivo bater com o hash registrado. Esta auditoria mede exatamente essa segunda propriedade. Como os binários ficam em `local_only` (fora do Git, por política), a forma reprodutível de publicar é um **manifesto URL + SHA256 + tamanho**, que permite re-download e verificação independentes.

## 3. Arquivos locais verificados

Os 8 arquivos baixados existem, têm tamanho idêntico ao registrado, hash SHA256 idêntico ao registrado e são parseáveis:

| Arquivo | Bytes | Hash confere | Parse |
|---|---|---|---|
| ARQ_IBGE_PET (GeoJSON) | 2.371 | sim | FeatureCollection, 1 feição |
| ARQ_IBGE_REC (GeoJSON) | 1.110 | sim | FeatureCollection, 1 feição |
| ARQ_IBGE_CUR (GeoJSON) | 1.773 | sim | FeatureCollection, 1 feição |
| ARQ_GEOCWB_BACIA (GeoJSON) | 804.561 | sim | FeatureCollection, 6 feições |
| ARQ_ANA_INVENTARIO (PDF) | 4.350.035 | sim | `%PDF`…`%%EOF` ok |
| ARQ_MAPBIOMAS_DHN250 (XLSX) | 9.177.772 | sim | OOXML válido, 54 entradas |
| ARQ_SGB_CARTA_PET (PDF) | 13.811.060 | sim | `%PDF`…`%%EOF` ok |
| ARQ_NHESS (PDF) | 7.765.136 | sim | `%PDF`…`%%EOF` ok |

Nenhum arquivo faltando; nenhum hash divergente; nenhum tamanho divergente.

## 4. URLs verificadas

Sondagem HEAD/GET leve (sem baixar corpo pesado):

- **Downloads que respondem:** IBGE (3 endpoints, 200), GeoCuritiba ArcGIS REST (200), ANA inventário (200, `Content-Length`=4.350.035 idêntico ao arquivo local), MapBiomas XLSX (206 a Range), SGB bitstream (200 via GET — o HEAD do DSpace devolve um stub HTML de 751 bytes, mas o GET entrega o PDF; o arquivo local de 13,8 MB confere por hash), NHESS PDF (206/200).
- **Páginas de origem que respondem:** IBGE downloads-geociencias (200), GeoCuritiba MapServer/54 (200), ANA HidroWeb (200), MapBiomas estatísticas (200), RIGEO/SGB handle/15692 (200), NHESS artigo (200).
- **Portais bloqueados:** CEMADEN (portal vivo, 206), APAC homepage (200) com `/boletins` 404, Copernicus Rapid Mapping (portal vivo, 200), DRM-RJ (**503 indisponível**).

## 5. Hashes conferidos

8 de 8 hashes SHA256 recalculados batem com os registrados no índice de arquivos baixados. Nenhum hash não confere. Isto torna os 8 arquivos verificáveis de forma independente por qualquer pessoa que re-baixe pela URL oficial.

## 6. Arquivos parseáveis

8 de 8 parseáveis: 4 GeoJSON abrem como `FeatureCollection` com contagem de feições coerente; 3 PDFs começam com `%PDF` e contêm `%%EOF`; o XLSX abre como pacote OOXML válido (54 entradas, `[Content_Types].xml` presente).

## 7. Origem pública/institucional das fontes

Todas as 10 fontes têm origem pública/institucional rastreável e URL oficial registrada: IBGE (federal), IPPUC/GeoCuritiba (municipal), ANA/SNIRH (federal), MapBiomas (rede pública de dados), SGB/CPRM e RIGEO (federal), NHESS/Copernicus Publications (periódico aberto), CEMADEN/MCTI (federal), APAC (estadual PE), Copernicus EMS (europeu), DRM-RJ/NADE (estadual RJ). Nenhuma fonte é classificada como inutilizável por questão de licença.

## 8. Fontes que exigem solicitação formal

- **CEMADEN** — o download pluviométrico depende de formulário que envia o link por e-mail. Confirmado: o portal está vivo, mas não há arquivo direto; o bloqueio é técnico (formulário/e-mail), não de licença. O papel de suscetibilidade é coberto por alternativa oficial (carta SGB).
- **DRM-RJ/NADE** — sem URL direta e portal retornando 503; requer solicitação formal. Alternativa oficial federal (SGB/CPRM) já baixada para o mesmo município/tema.

## 9. Fontes bloqueadas por portal ou ausência de arquivo direto

- **APAC** — homepage responde (200), mas `/boletins` retorna 404 e não há PDF/CSV estático nos menus; dados em PCD de tempo real. Bloqueio técnico (portal sem arquivo estático). Ângulo de chuva parcialmente coberto pelo inventário ANA.
- **Copernicus EMS Rapid Mapping** — portal vivo (200, aplicação JS), mas nenhum produto vetorial público de rapid mapping confirmado para Petrópolis (ID EMSR não confirmado — não inventar). Bloqueio técnico/metodológico. Mesmo que existisse, delimitaria deslizamento, não inundação.
- **CEMADEN/DRM-RJ** — ver seção 8.

## 10. O que pode ir para pacote externo

- **Prioridade alta:** manifesto URL + SHA256 + tamanho das 8 fontes — artefato leve e suficiente para reprodução pública (a pessoa externa re-baixa e confere o hash; não é preciso redistribuir binários).
- **Prioridade média:** os 4 GeoJSON leves (3 malhas IBGE + bacias GeoCuritiba) como referência de **contexto** territorial/hidrográfico, nunca como evento; e a aquisição da série ANA por estação do Capibaribe (ainda não baixada).
- **Prioridade baixa:** os PDFs/XLSX pesados (ANA, MapBiomas, SGB, NHESS) referenciados por URL+hash, mantidos em `local_only`.

## 11. O que não deve ir para Git por motivo técnico

Por política do projeto e por peso, **nenhum dos 8 binários externos** deve entrar no Git/`outputs_public`; permanecem em `local_only` e são referenciados por URL+hash. Além disso:

- **SIG vetorial completo da carta SGB (~1,8 GB)** — pesado demais; `nao_publicar`, manter fora do repositório e baixar sob demanda local.
- PDFs de 4–14 MB (ANA, MapBiomas, SGB, NHESS) — arquivos brutos não necessários para Git.

Confirmação: **0 brutos pesados em `outputs_public/`** (varredura por extensão e por tamanho >5 MB).

## 12. O que ainda não é reproduzível por pessoa externa

- **Série hidrológica ANA por estação** (Capibaribe) — exige seleção manual de estação/período no HidroWeb; não é arquivo direto.
- **CEMADEN** — exige formulário/e-mail.
- **Copernicus EMS** — produto vetorial público não confirmado para Petrópolis.
- **APAC** — sem arquivo estático.
- **DRM-RJ** — portal 503; requer solicitação formal.

Os 8 arquivos já baixados **são** reproduzíveis por terceiros (URLs respondem, hashes conferem); a limitação é que os binários não são redistribuídos no repositório — o manifesto URL+hash resolve isso.

## 13. Riscos para artigo/slides

- **Contexto apresentado como evento:** malhas IBGE, bacias GeoCuritiba, inventário ANA e cobertura MapBiomas são contexto territorial/hidrográfico, não evento observado. Nunca rotular como ocorrência.
- **Suscetibilidade como evento:** a carta SGB é suscetibilidade (PDF do mapa), não evento; e mistura deslizamento e inundação — risco landslide vs flood.
- **Fonte secundária como geometria:** o artigo NHESS é referência documental; suas figuras não são geometria auditável.
- **Charter 758 (FONTE_REC_002):** produto digitalizado candidato, não revisado; pode misturar cicatrizes de deslizamento com área de inundação — não assumir flood extent.

## 14. Recomendações objetivas para deixar público

1. Publicar um **manifesto de reprodutibilidade** (URL oficial + SHA256 + tamanho + formato + instituição + data de acesso) como o pacote externo principal — leve e suficiente.
2. Marcar explicitamente cada item como **contexto/suscetibilidade/secundário**, nunca evento, no material público.
3. Manter binários em `local_only`; não versionar; referenciar por hash.
4. Registrar os bloqueios como **técnicos/metodológicos** (formulário/e-mail, portal sem arquivo estático, 503, sem produto vetorial), sem mencionar licença como impedimento.
5. Sinalizar `nao_publicar` para o SIG de 1,8 GB e para produtos ainda não baixados/confirmados (CEMADEN, Copernicus, série ANA por estação).
6. Substituir, nos artefatos de navegação existentes, o critério/observação de "licença incerta" por critérios técnicos de reprodutibilidade — esta auditoria não usa licença como bloqueador.

## 15. Guardrails preservados

- evidência externa não vira label;
- download não vira validação operacional;
- suscetibilidade não vira evento observado;
- landslide scar não vira flood extent;
- texto sem geometria não fecha patch-level;
- Curitiba não vira negativo formal;
- não liberar treino;
- não declarar ground truth operacional.
</content>
