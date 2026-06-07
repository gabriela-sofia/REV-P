# Dossie v2an - CTB_2024_02_18_20

ground_truth_status=NOT_ESTABLISHED. Documento de validacao observacional,
nao e ground truth, label, classe, target nem predicao.

## Identificacao
- Candidate: CTB_2024_02_18_20
- Regiao: Curitiba
- Evento: Chuvas intensas de fevereiro de 2024
- Fenomeno: flood_or_heavy_rain (ambiguo: false)
- Janela: 2024-02-18 a 2024-02-20 (3 dia(s))

## Fontes
- Primaria: Prefeitura de Curitiba (OFFICIAL_MUNICIPAL)
- Secundaria: Banda B / recorrência de alagamentos no Cajuru
- Forca da fonte: STRONG

## Ancoras espaciais
- Nivel espacial: NEIGHBORHOOD
- Mapa/laudo: false; bairro: true
- Geometria/coordenada: nao disponivel; revisao manual de geometria requerida.

## Gates
- G1 CLOSED | G2 CLOSED | G3 CLOSED | G4 CLOSED
- G4B CLOSED | G5 CLOSED | G7 CLOSED | G8 OPEN
- G9 BLOCKED_PENDING_HUMAN_REVIEW

## Blockers
- Dominante: no explicit Sentinel crosswalk and human ground-reference review pending
- Sem crosswalk Sentinel explicito; sem geometria de evento; revisao humana pendente.

## Readiness
- Score: 85 | banda: HIGH
- Pode entrar em revisao humana de ground reference: true

## Decisao segura
- decision_status: ADVANCES_TO_HUMAN_GROUND_REFERENCE_REVIEW

## O que falta para ground reference real
- Geometria oficial do evento; crosswalk Sentinel explicito; revisao humana e adjudicacao reais; revisao de licenca.

## O que continua proibido
- Nao criar ground truth operacional; nao criar label/classe/target; nao treinar;
  nao abrir Protocolo B; nao gerar overlay; nao inferir data ou geometria.
