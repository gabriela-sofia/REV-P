# Petrópolis — critério de separação enchente/deslizamento (desbloqueio, não modelagem)

**Status**: decisão já existe, filtro ainda não aplicável por falta de dado
registro-a-registro. Etapa 5 de `revp_proxima_linhagem_programacao_pos_api.md`
— só o desbloqueio, sem seguir para feature física nem modelo nesta rodada.

## O que existe

O critério de separação já foi decidido em sessão anterior e está documentado
em `PROJETO/docs/fase2_decisao_label_curitiba_petropolis.md` (2026-04-30,
seção 3 "Decisão para Petrópolis"). Não foi refeito nem alterado aqui — só
confirmado como a fonte de verdade adotada por este roadmap. Resumo do
critério (COBRADE, Codificação Brasileira de Desastres):

| COBRADE | Fenômeno | Decisão |
|---|---|---|
| 12200 | Enxurrada (flash flood) | **INCLUIR** (enchente) |
| 12300 | Inundação gradual | **INCLUIR** (enchente) |
| 12400 | Alagamento | **INCLUIR** (enchente) |
| 11300 | Deslizamento/corrida de massa | **EXCLUIR** |
| Eventos mistos (deslizamento que entulhou canal e inundou) | — | **EXCLUIR ou incerto** |

Validação obrigatória já registrada: amostra manual de 20-30 registros por
região antes de qualquer uso, para confirmar que o campo do registro não
classificou deslizamento como inundação por erro de digitação/triagem.

## Por que o filtro não foi aplicado nesta rodada

O critério é sobre um campo (`COBRADE`) que existiria num dataset
registro-a-registro da Defesa Civil (S2ID e/ou Defesa Civil municipal de
Petrópolis, PMP). Esse dataset **não está presente localmente**:
`PROJETO/data/raw/petropolis/` só contém camadas geológicas/hidrológicas de
referência (SGB/CPRM, GeoINEA) — nenhum registro de ocorrência com COBRADE.
As únicas referências ao S2ID encontradas em `PROJETO/outputs/external_validation/`
são snapshots HTML da página do portal, não exportações de dado. Aplicar o
filtro exigiria uma aquisição nova (S2ID ou contato direto com a Defesa Civil
PMP) — fora do escopo desta rodada, que pede só o desbloqueio conceitual.

O registro `ground_reference_candidate_master_registry.csv` do REV-P (36
linhas, todas `MOVEMENT_OF_MASS`, fonte: relatórios técnicos CPRM
`ANEXO-*-CPRM_Relatório_Petrópolis_*`) já é, por construção, só deslizamento —
não há nele nenhum registro de enchente/enxurrada/alagamento misturado. Ou
seja: a mistura de fenômeno que motiva esta etapa está na fonte bruta ainda
não baixada (S2ID/Defesa Civil PMP), não no que já foi processado pelo REV-P.

## Efeito em `region_registry.py`

Nenhum. Petrópolis permanece `region_maturity="insufficient"`,
`model_version=None` — o critério de separação está decidido, mas não há
ainda dado registro-a-registro para aplicá-lo, então a régua do roadmap
("`region_registry.py` de Petrópolis passa de `insufficient` pra
`limited_evidence` só quando o filtro estiver aplicado e documentado") não
foi atingida.

## Próximo passo real (fora desta rodada)

1. Baixar S2ID filtrado por Petrópolis + COBRADE 12200/12300/12400 (prioridade
   já registrada como alta em `fase2_decisao_label_curitiba_petropolis.md`).
2. Ou contato direto com a Defesa Civil municipal de Petrópolis (PMP) para
   registros de fevereiro/março 2022 com COBRADE.
3. Só então aplicar o filtro já decidido e reavaliar `region_maturity`.

---

## Atualização 2026-07-26 — passo 1 EXECUTADO

O S2ID **foi adquirido de fato** (módulo público Relatórios > Danos Informados,
export CSV, sem login) e o filtro **foi aplicado de fato**. Resultado real, em
`revp_petropolis_s2id_aquisicao_real_cobrade.md`:

- Petrópolis tem **3 registros em 2022**, todos COBRADE **13214 – Tempestade
  Local/Convectiva – Chuvas Intensas** (inclui o de 15/02/2022, 78 mortos).
- **Zero** registros nas classes hidrológicas e **zero** em movimento de massa:
  a classificação é pelo **gatilho meteorológico**, não pelo processo. O filtro
  decidido retorna vazio — é **inaplicável**, não pendente.
- Dois dos três códigos da tabela acima estão errados: na lista oficial do S2ID,
  **12300 é Alagamentos** (não inundação gradual), **inundação é 12100**, e
  **12400 não existe**.
- Bloqueio adicional e independente: o registro S2ID é **municipal**, sem
  geometria nem coordenada — não ancora pontos mesmo se o COBRADE separasse.

`region_maturity` de Petrópolis **permanece `insufficient`** (a régua exige
filtro aplicado *com resultado*; o resultado foi zero). O `status_note` foi
atualizado para descrever este bloqueio real em vez de "dado faltando".
