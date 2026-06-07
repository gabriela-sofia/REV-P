# Dossie v2an - PET_2024_03_21_28

ground_truth_status=NOT_ESTABLISHED. Documento de validacao observacional,
nao e ground truth, label, classe, target nem predicao.

## Identificacao
- Candidate: PET_2024_03_21_28
- Regiao: Petropolis
- Evento: Chuvas do fim de março de 2024
- Fenomeno: desastre_hidrometeorologico (ambiguo: false)
- Janela: 2024-03-21 a 2024-03-28 (8 dia(s))

## Fontes
- Primaria: Prefeitura de Petrópolis (OFFICIAL_MUNICIPAL)
- Secundaria: Copernicus Global Flood Monitoring
- Forca da fonte: STRONG

## Ancoras espaciais
- Nivel espacial: PARTIAL
- Mapa/laudo: false; bairro: false
- Geometria/coordenada: nao disponivel; revisao manual de geometria requerida.

## Gates
- G1 CLOSED | G2 CLOSED | G3 CLOSED | G4 OPEN
- G4B OPEN | G5 CLOSED | G7 OPEN | G8 CLOSED
- G9 BLOCKED_PENDING_HUMAN_REVIEW

## Blockers
- Dominante: no specific spatial geometry (needs manual spatial evidence)
- Sem crosswalk Sentinel explicito; sem geometria de evento; revisao humana pendente.

## Readiness
- Score: 70 | banda: HIGH
- Pode entrar em revisao humana de ground reference: false

## Decisao segura
- decision_status: NEEDS_MORE_SPATIAL_EVIDENCE

## O que falta para ground reference real
- Geometria oficial do evento; crosswalk Sentinel explicito; revisao humana e adjudicacao reais; revisao de licenca.

## O que continua proibido
- Nao criar ground truth operacional; nao criar label/classe/target; nao treinar;
  nao abrir Protocolo B; nao gerar overlay; nao inferir data ou geometria.
