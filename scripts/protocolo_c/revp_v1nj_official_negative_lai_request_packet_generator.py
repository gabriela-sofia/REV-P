"""REV-P v1nj - LAI / Defesa Civil official request packet generator."""

from __future__ import annotations

import argparse
import json

from revp_v1ni_v1nn_common import DATASETS, DOCS, SCHEMAS, read_csv, write_doc, write_outputs


TARGETS = DATASETS / "official_negative_evidence_request_target_registry.csv"
DOC_PACKET = DOCS / "protocolo_c_pacote_lai_negativos_v1nj.md"
DOC_MODEL = DOCS / "modelo_pedido_lai_defesa_civil_petropolis_2022_v1nj.md"
OUT_MANIFEST = DATASETS / "official_negative_request_packet_manifest.csv"
OUT_QUESTIONS = DATASETS / "official_negative_request_question_bank.csv"
SCHEMA_MANIFEST = SCHEMAS / "official_negative_request_packet_manifest_schema.csv"
SCHEMA_QUESTIONS = SCHEMAS / "official_negative_request_question_bank_schema.csv"

MANIFEST_FIELDS = ["packet_item_id", "artifact", "artifact_type", "target_count", "intended_channel", "send_status", "raw_response_policy", "notes"]
QUESTION_FIELDS = ["question_id", "target_evidence_type", "official_channel", "question_text", "required_answer_format", "why_needed_for_c4", "forbidden_inference_if_absent", "priority"]


def question_bank() -> list[dict[str, str]]:
    questions = [
        (
            "Q1",
            "2022 inspection records with explicit negative outcome",
            "LAI/Prefeitura/Defesa Civil",
            "Solicito autos, laudos, fichas de vistoria ou tabelas de vistorias de 2022 em Petropolis cujo resultado tecnico tenha indicado sem risco, sem ocorrencia, sem instabilidade ou sem dano geologico.",
            "copy of document or anonymized table with date, place, phenomenon, result, issuing body",
        ),
        (
            "Q2",
            "place-specific no occurrence or stability statement",
            "Defesa Civil/Orgao tecnico municipal",
            "Para cada registro localizado, informar endereco, coordenada ou bairro, data da vistoria, fenomeno avaliado e resultado tecnico explicito.",
            "structured table preserving locality, date, phenomenon, and technical result",
        ),
        (
            "Q3",
            "technical authorship or issuing body",
            "Prefeitura/Defesa Civil/SGB/CPRM/DRM",
            "Indicar o orgao emissor, responsavel tecnico quando disponivel, numero do processo, auto ou laudo, e se o documento pode ser fornecido integralmente.",
            "metadata table plus document copy when allowed",
        ),
        (
            "Q4",
            "negative evidence scope limits",
            "LAI/Prefeitura",
            "Se existirem apenas listas de ocorrencias positivas, favor informar se ha documentos separados que registrem vistorias sem ocorrencia ou estabilidade; lista positiva sozinha nao atende ao pedido.",
            "explicit answer distinguishing positive-only list from negative or stability records",
        ),
        (
            "Q5",
            "anonymized response option",
            "LAI/Defesa Civil",
            "Caso haja dados pessoais ou sensiveis, solicito copia anonimizada ou tabela com localidade, data, fenomeno, resultado tecnico e orgao emissor preservados.",
            "anonymized document or table retaining scientific fields",
        ),
    ]
    return [
        {
            "question_id": f"OFFNEG_{qid}_V1NJ",
            "target_evidence_type": evidence_type,
            "official_channel": channel,
            "question_text": text,
            "required_answer_format": fmt,
            "why_needed_for_c4": "C4 requires formal negative evidence; silence, positive-only lists, and pseudo-absence cannot close the negative gate.",
            "forbidden_inference_if_absent": "Do not infer absence, stability, formal negative, label readiness, or training readiness from non-response or missing records.",
            "priority": "HIGH",
        }
        for qid, evidence_type, channel, text, fmt in questions
    ]


def build_manifest(target_count: int) -> list[dict[str, str]]:
    return [
        {
            "packet_item_id": "OFFNEG_PACKET_DOC_V1NJ",
            "artifact": "docs/metodologia_cientifica/protocolo_c_pacote_lai_negativos_v1nj.md",
            "artifact_type": "method_packet",
            "target_count": str(target_count),
            "intended_channel": "LAI/Prefeitura/Defesa Civil/orgao tecnico",
            "send_status": "NOT_SENT_BY_SCRIPT",
            "raw_response_policy": "responses must be stored only in local_runs/protocolo_c/official_negative_response_inbox",
            "notes": "Packet asks for evidence; it does not assert that documents exist.",
        },
        {
            "packet_item_id": "OFFNEG_MODEL_DOC_V1NJ",
            "artifact": "docs/metodologia_cientifica/modelo_pedido_lai_defesa_civil_petropolis_2022_v1nj.md",
            "artifact_type": "reusable_request_model",
            "target_count": str(target_count),
            "intended_channel": "LAI/Prefeitura/Defesa Civil/orgao tecnico",
            "send_status": "NOT_SENT_BY_SCRIPT",
            "raw_response_policy": "responses must be stored only in local_runs/protocolo_c/official_negative_response_inbox",
            "notes": "Direct reusable wording for the user.",
        },
    ]


def write_docs(target_count: int) -> None:
    write_doc(
        DOC_PACKET,
        "Protocolo C - pacote LAI para negativos formais v1nj",
        [
            f"Pacote preparado a partir de {target_count} alvos v1ni. O script nao envia solicitacao e nao presume que os documentos existam.",
            "O pedido busca comprovacao explicita de ausencia, estabilidade, sem ocorrencia, sem instabilidade ou sem dano geologico em Petropolis em 2022.",
            "Nao basta lista de ocorrencias positivas, silencio administrativo, ausencia em Diario Oficial ou falta de registro em base publicada.",
            "Dados sensiveis podem ser anonimizados, desde que preservem localidade, data, fenomeno avaliado, resultado tecnico e orgao emissor.",
        ],
    )
    write_doc(
        DOC_MODEL,
        "Modelo de pedido LAI / Defesa Civil - Petropolis 2022 v1nj",
        [
            "Solicito, para fins de pesquisa cientifica, autos, laudos, fichas de vistoria ou tabelas oficiais de vistorias realizadas em Petropolis em 2022 que tenham registrado resultado tecnico explicito sem risco, sem ocorrencia, sem instabilidade ou sem dano geologico relacionado a deslizamento, escorregamento, movimento de massa, encosta ou barreira.",
            "Para cada registro, solicito: data da vistoria, endereco, coordenada ou bairro, fenomeno avaliado, resultado tecnico, orgao emissor, responsavel tecnico quando disponivel, numero de processo/auto/laudo e copia integral quando juridicamente possivel.",
            "Caso haja dados pessoais ou sensiveis, aceito copia anonimizada ou tabela anonimizada, desde que mantenha localidade, data, fenomeno, resultado tecnico e orgao emissor.",
            "Esclareco que lista de ocorrencias positivas ou ausencia de registro nao substitui declaracao tecnica explicita de ausencia, estabilidade, sem ocorrencia, sem instabilidade ou sem dano geologico.",
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--emit-packet", action="store_true")
    args = parser.parse_args()
    if OUT_MANIFEST.exists() and OUT_QUESTIONS.exists() and not args.force:
        print(json.dumps({"stage": "v1nj", "status": "EXISTING_OUTPUTS_PRESERVED"}))
        return
    target_count = len(read_csv(TARGETS))
    manifest = build_manifest(target_count)
    questions = question_bank()
    if args.force or args.emit_packet:
        write_docs(target_count)
        write_outputs(
            [(OUT_MANIFEST, manifest, MANIFEST_FIELDS), (OUT_QUESTIONS, questions, QUESTION_FIELDS)],
            [(SCHEMA_MANIFEST, MANIFEST_FIELDS, "v1nj request packet manifest"), (SCHEMA_QUESTIONS, QUESTION_FIELDS, "v1nj request question bank")],
            [DOC_PACKET, DOC_MODEL],
        )
    print(json.dumps({"stage": "v1nj", "questions": len(questions), "send_status": "NOT_SENT_BY_SCRIPT"}, indent=2))


if __name__ == "__main__":
    main()
