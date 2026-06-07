# Dossie v2an - PET_2022_02_15

ground_truth_status=NOT_ESTABLISHED. Documento de validacao observacional,
nao e ground truth, label, classe, target nem predicao.

## Identificacao
- Candidate: PET_2022_02_15
- Regiao: Petropolis
- Evento: Desastre hidrometeorológico de 15 de fevereiro de 2022
- Fenomeno: evento_misto (ambiguo: true)
- Janela: 2022-02-15 a 2022-02-16 (2 dia(s))

## Fontes
- Primaria: DRM-RJ / NADE / Thalweg (TECHNICAL_REPORT)
- Secundaria: NHESS / Copernicus Image of the Day
- Forca da fonte: STRONG

## Ancoras espaciais
- Nivel espacial: TECHNICAL_MAP
- Mapa/laudo: true; bairro: true
- Geometria/coordenada: nao disponivel; revisao manual de geometria requerida.

## Gates
- G1 CLOSED | G2 CLOSED | G3 CLOSED | G4 CLOSED
- G4B CLOSED | G5 CLOSED | G7 CLOSED | G8 CLOSED
- G9 BLOCKED_PENDING_HUMAN_REVIEW

## Blockers
- Dominante: phenomenon ambiguity (mixed flood/mass movement separation pending)
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
