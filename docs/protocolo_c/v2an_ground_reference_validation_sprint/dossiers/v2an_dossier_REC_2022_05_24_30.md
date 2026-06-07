# Dossie v2an - REC_2022_05_24_30

ground_truth_status=NOT_ESTABLISHED. Documento de validacao observacional,
nao e ground truth, label, classe, target nem predicao.

## Identificacao
- Candidate: REC_2022_05_24_30
- Regiao: Recife
- Evento: Chuvas extremas e alagamentos do fim de maio de 2022
- Fenomeno: flood_or_heavy_rain (ambiguo: false)
- Janela: 2022-05-24 a 2022-05-30 (7 dia(s))

## Fontes
- Primaria: Prefeitura do Recife (OFFICIAL_MUNICIPAL)
- Secundaria: Prefeitura do Recife; APAC; CEMADEN; Copernicus metodologia
- Forca da fonte: STRONG

## Ancoras espaciais
- Nivel espacial: NEIGHBORHOOD
- Mapa/laudo: false; bairro: true
- Geometria/coordenada: nao disponivel; revisao manual de geometria requerida.

## Gates
- G1 CLOSED | G2 CLOSED | G3 CLOSED | G4 CLOSED
- G4B CLOSED | G5 CLOSED | G7 CLOSED | G8 CLOSED
- G9 BLOCKED_PENDING_HUMAN_REVIEW

## Blockers
- Dominante: no explicit Sentinel crosswalk and human ground-reference review pending
- Sem crosswalk Sentinel explicito; sem geometria de evento; revisao humana pendente.

## Readiness
- Score: 95 | banda: HIGH
- Pode entrar em revisao humana de ground reference: true

## Decisao segura
- decision_status: ADVANCES_TO_HUMAN_GROUND_REFERENCE_REVIEW

## O que falta para ground reference real
- Geometria oficial do evento; crosswalk Sentinel explicito; revisao humana e adjudicacao reais; revisao de licenca.

## O que continua proibido
- Nao criar ground truth operacional; nao criar label/classe/target; nao treinar;
  nao abrir Protocolo B; nao gerar overlay; nao inferir data ou geometria.
