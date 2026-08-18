# Petrópolis — aquisição real do S2ID e teste do critério COBRADE

**Status**: `AQUISICAO_EXECUTADA_FILTRO_INAPLICAVEL`. Fecha o "próximo passo real
(fora desta rodada)" nº 1 de
`revp_petropolis_separacao_fenomeno_enchente_deslizamento.md`: *"Baixar S2ID
filtrado por Petrópolis + COBRADE 12200/12300/12400"*. O dado **foi baixado de
fato**. O filtro **foi aplicado de fato**. O resultado é que o critério não
separa nada em Petrópolis — não por falta de dado, mas porque o dado não usa
esses códigos.

## 1. O S2ID tem export público e programático — sem login

O documento anterior registrou que *"as únicas referências ao S2ID encontradas
em `PROJETO/outputs/external_validation/` são snapshots HTML da página do portal,
não exportações de dado"* e classificou a aquisição como fora de escopo. Nesta
rodada a aquisição foi tentada e **funcionou**.

Duas rotas públicas foram sondadas em `s2id.mi.gov.br` (JSF/PrimeFaces 5.2):

### Rota A — Série Histórica (`/paginas/series/`): **bloqueada, e o motivo é específico**

Formulário simples: slider de período (2003–2026) + seletor de UF + botão
Pesquisar. Dirigi o postback por dois caminhos independentes (POST JSF
reconstruído em Python, e o navegador real acionando `atualizarLinhaTempo()` +
`PF('widget_j_idt30').selectValue('RJ')` + clique no botão).

Com RJ/2022 o módulo responde `HTTP 200` com `partial-response` válido, mas a
`datatable` volta com **1 linha zerada**. Não é erro de sessão nem de ViewState:
com Brasil/2003–2026 o **mesmo** postback retorna **379 linhas** normalmente.

Diagnóstico real: **a Série Histórica só contém dados até 2016.** Os anos
distintos presentes na resposta completa são 2003…2016 — nada de 2017 em diante,
apesar de o slider oferecer até 2026 e de a própria página afirmar que cobre
"desde o ano de 2013". RJ aparece com linhas de 2003 a 2016 e nenhuma depois.

Bloqueio: **conteúdo desatualizado no servidor há ~10 anos**, não autenticação,
não captcha, não ausência de export. A UI oferece um período que o backend não
tem. Registrado como bloqueio específico, no mesmo padrão do caso PE3D.

### Rota B — Relatórios → Danos Informados (`/paginas/relatorios/`): **funcionou**

Painel *"Relatório Gerencial – Danos Informados"*, público, sem login, com:
filtro de período (`abas:sanfonas:j_idt74_input` / `dt_final_danos_input`),
**65 checkboxes de tipologia COBRADE** (`abas:sanfonas:cobrades3`), filtro de UF
(`abas:sanfonas:estadosDanosInformados`) e botão **Exportar CSV**
(`abas:sanfonas:j_idt95`).

POST reconstruído em Python (`s2id_danos_uf.py`), sem navegador:

```
HTTP 200 | text/csv | attachment;filename="Danos_Informados.csv" | 41.175 bytes | 65 cobrades
TOTAL RJ 2022: 145 registros
```

CSV real, registro-a-registro, 53 colunas: UF, Município, Registro, **Protocolo**,
**COBRADE**, Status, População, danos humanos (DH_*), materiais (DM_*),
ambientais (DA_*) e prejuízos públicos/privados (PEPL_*/PEPR_*).

**Limitação operacional encontrada**: janelas maiores que 1 ano com as 65
tipologias fazem o servidor devolver a página HTML em vez do CSV
(`Content-Type: text/html`, 252 KB, sem mensagem de erro). Contornado consultando
ano a ano. Registrado porque afeta qualquer reuso.

## 2. Registros reais de Petrópolis obtidos

Filtro: UF=RJ, **todas as 65 tipologias COBRADE** (não só as hidrológicas — para
garantir que nada fosse perdido pelo próprio filtro), período 01/01–31/12/2022.
Petrópolis (IBGE 3303906) tem **exatamente 3 registros**, todos com status
`Reconhecido`:

| Protocolo | Data do evento | COBRADE | Mortos | Desabrigados | Desalojados | Desaparecidos |
|---|---|---|---|---|---|---|
| RJ-F-3303906-**13214**-20220107 | 07/01/2022 | 13214 – Tempestade Local/Convectiva – Chuvas Intensas | 0 | 12 | 227 | 0 |
| RJ-F-3303906-**13214**-20220215 | **15/02/2022** | 13214 – Tempestade Local/Convectiva – Chuvas Intensas | **78** | 450 | 0 | 200 |
| RJ-F-3303906-**13214**-20220320 | 20/03/2022 | 13214 – Tempestade Local/Convectiva – Chuvas Intensas | 5 | 1.167 | 240 | 3 |

O registro de 15/02/2022 é o desastre-alvo: 78 mortos, 350 feridos, 200
desaparecidos, 120.000 outros afetados, 400 unidades habitacionais danificadas e
180 destruídas.

## 3. Aplicação do critério COBRADE decidido — e por que ele não separa

### 3.1 Correção nos códigos do critério

O critério registrado em `fase2_decisao_label_curitiba_petropolis.md` e repetido
em `revp_petropolis_separacao_fenomeno_enchente_deslizamento.md` usa esta tabela:

| Código no documento | Fenômeno afirmado |
|---|---|
| 12200 | Enxurrada |
| 12300 | Inundação gradual |
| 12400 | Alagamento |

A lista oficial servida pelo próprio S2ID (extraída dos rótulos dos 65
checkboxes) é:

| COBRADE oficial | Fenômeno |
|---|---|
| 12100 | Inundações |
| 12200 | Enxurradas |
| 12300 | **Alagamentos** |

Ou seja: **12300 é Alagamentos, não inundação gradual; inundação é 12100; e
12400 não existe** na tabela do S2ID. O critério do projeto estava com dois dos
três códigos errados. Nesta rodada usei os códigos oficiais (12100, 12200, 12300)
e ainda assim ampliei para todas as 65 tipologias, então o achado não depende
dessa correção.

### 3.2 O filtro aplicado dá zero

| Grupo | Códigos | Registros de Petrópolis 2022 |
|---|---|---|
| Hidrológico — INCLUIR | 12100, 12200, 12300 | **0** |
| Movimento de massa — EXCLUIR | 11321, 11331, 11332 | **0** |
| Meteorológico (gatilho) | **13214** | **3** |

**Os três registros caem fora das duas classes do critério.** A Defesa Civil de
Petrópolis classificou os eventos pelo **gatilho meteorológico** (chuva intensa),
não pelo **processo** (enchente ou deslizamento). O critério de separação
enchente/deslizamento é, portanto, **inaplicável a este dataset** — não por
lacuna de aquisição, mas porque a variável de separação não está preenchida com
os valores que o critério pressupõe.

Isto é diferente do que o documento anterior supunha. Ele supunha que o dado
existiria com COBRADE preenchido registro a registro e que bastaria filtrar. O
dado existe, tem COBRADE, e o COBRADE aponta para outro eixo de classificação.

### 3.3 Um segundo bloqueio, independente do primeiro

Mesmo que o COBRADE separasse o fenômeno, o registro do S2ID **não é pontual**:
a unidade é o **município** (Petrópolis inteira, população 296.044), sem
geometria, sem bairro, sem coordenada. Os campos são contagens agregadas
(mortos, desabrigados, unidades habitacionais) e valores em reais por setor.

Ou seja: o S2ID nunca poderia alimentar diretamente um modelo ponto-a-ponto como
o de Recife (que usa 278 pontos com lat/lon). Ele serve como **confirmação
administrativa federal de que houve desastre naquela data naquele município** —
tier de evidência real, mas de nível municipal.

A validação obrigatória prevista no critério ("amostra manual de 20–30 registros
por região antes de qualquer uso") ficou **prejudicada por N**: existem 3
registros no ano, não 20–30. Todos os 3 foram inspecionados integralmente, campo
a campo (saída completa em `dados/`).

## 4. Reavaliação de `region_registry.py`

A régua do roadmap: *"`region_registry.py` de Petrópolis passa de `insufficient`
pra `limited_evidence` só quando o filtro estiver aplicado e documentado."*

O filtro foi aplicado e está documentado. O resultado dele é **zero registros na
classe enchente**. Promover a região com base num filtro que retornou vazio seria
overclaim direto.

**`region_maturity` permanece `insufficient`, `model_version=None`.** O que muda
é o `status_note`: o bloqueio deixa de ser *"mistura enchente/deslizamento ainda
não separada — bloqueio documentado, não resolvido"* (que sugere dado faltando) e
passa a ser o bloqueio real e mais específico — o dado federal existe, foi
adquirido, e é municipal e classificado por gatilho meteorológico, o que o torna
incapaz de separar fenômeno **e** de ancorar pontos.

O registro `ground_reference_candidate_master_registry.csv` (36 linhas, todas
`MOVEMENT_OF_MASS`, fonte CPRM) continua sendo a única evidência de Petrópolis
com granularidade sub-municipal — e continua sendo só deslizamento.

## 5. O que ainda desbloquearia (não tentado nesta rodada)

1. **Defesa Civil municipal de Petrópolis (PMP)** — registro de atendimentos com
   endereço, equivalente ao que a SEDEC de Recife fornece (`data/raw/recife/
   seced_defesa_civil/`, 11 CSVs anuais que geraram os 141 pontos geocodificados
   de Recife). É esse tipo de arquivo que produz pontos; o S2ID não é.
2. **FIDE** (Formulário de Informações do Desastre) dos 3 protocolos acima —
   pode conter descrição textual das áreas atingidas. Não foi localizada rota
   pública de download do FIDE por protocolo; o módulo de Relatórios entrega
   agregados, não o formulário.

## 6. Artefatos

Em `PROJETO/local_runs/petropolis_s2id_aquisicao_cobrade/dados/` (git-ignored):

| Arquivo | Conteúdo |
|---|---|
| `s2id_danos_uf.py` | consulta reexecutável: `python s2id_danos_uf.py <UF> <ANO> <município>` |
| `s2id_danos_rj_2022_todas_cobrade.csv` | 145 registros do RJ em 2022, 65 tipologias, 53 colunas |
| `s2id_danos_PR_2021_todas_cobrade.csv` | 186 registros do PR em 2021 (checagem cruzada Curitiba) |
| `s2id_danos_PR_2022_todas_cobrade.csv` | 204 registros do PR em 2022 |
| `s2id_series_2022.py` | sonda da Série Histórica que evidenciou o corte em 2016 |

### Reproduzir

```bash
python s2id_danos_uf.py RJ 2022 Petropolis
```
