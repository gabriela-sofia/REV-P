"""Tests — REV-P Protocol C v1tx-v1ub TCC Evidence Dossier Exporter.

Outputs redirected to tmp_path. No network. No real dataset writes.
"""
from __future__ import annotations
import csv, importlib, re, subprocess, sys
from pathlib import Path

ROOT    = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts" / "protocolo_c"
sys.path.insert(0, str(SCRIPTS))

import revp_v1tx_v1ub_tcc_dossier_common as D  # noqa: E402

v1tx = importlib.import_module("revp_v1tx_case_dossier_exporter")
v1ty = importlib.import_module("revp_v1ty_final_evidence_matrix")
v1tz = importlib.import_module("revp_v1tz_tcc_latex_table_fragments")
v1ua = importlib.import_module("revp_v1ua_tcc_narrative_draft_generator")
v1ub = importlib.import_module("revp_v1ub_tcc_dossier_claim_audit_bundle")

ALL = [v1tx, v1ty, v1tz, v1ua, v1ub]


def _read(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _redirect(monkeypatch, mod, tmp: Path) -> None:
    for name in dir(mod):
        if name.startswith(("OUT_", "SCHEMA_", "DOC", "IN_")):
            val = getattr(mod, name)
            if isinstance(val, Path):
                monkeypatch.setattr(mod, name, tmp / val.name)
    monkeypatch.setattr(mod, "DATASETS", tmp)


def _setup_inputs(tmp: Path, cid: str = "CASE_PET_X1"):
    _write_csv(tmp / "protocol_c_unified_evidence_case_index_v1tn.csv", [{
        "case_id": cid, "region": "PET", "hazard_type": "FLOOD_LANDSLIDE",
        "event_window": "2022-02-12 to 2022-02-20",
        "external_evidence_status": "EXTERNAL_SOURCE_ABSENT_LOCAL",
        "hydromet_status": "HYDROMET_CONTEXT_AVAILABLE_REVIEW_ONLY",
        "dino_status": "DINO_NOT_PRESENT_CONTEXT_ONLY",
        "patch_link_status": "PATCH_LINK_ABSENT",
        "protocol_c_state": "PROTOCOL_C_BLOCKED_INSUFFICIENT_EVIDENCE",
        "case_readiness_status": "CASE_CONTEXT_AVAILABLE_NEEDS_EXTERNAL_SOURCE",
    }], ["case_id", "region", "hazard_type", "event_window",
         "external_evidence_status", "hydromet_status", "dino_status",
         "patch_link_status", "protocol_c_state", "case_readiness_status"])
    _write_csv(tmp / "protocol_c_single_flow_review_export_v1ts.csv",
               [{"case_id": cid}], ["case_id"])
    _write_csv(tmp / "protocol_c_automated_supervisor_adjudication_v1tr.csv", [{
        "case_id": cid,
        "supervisor_decision": "AUTOMATED_SUPERVISOR_VALIDATED_FOR_REVIEW_ONLY_USE",
        "final_for_review_only_use": "true", "ready_for_tcc_discussion": "true",
    }], ["case_id", "supervisor_decision", "final_for_review_only_use",
         "ready_for_tcc_discussion"])
    _write_csv(tmp / "protocol_c_proof_of_review_only_validation_audit_v1tu.csv", [{
        "case_id": cid, "proof_status": "REVIEW_ONLY_PROOF_COMPLETE",
        "review_only_validation_status": "VALIDATED_FOR_REVIEW_ONLY_USE",
    }], ["case_id", "proof_status", "review_only_validation_status"])


def _run_chain(monkeypatch, tmp: Path):
    _setup_inputs(tmp)
    for mod in ALL:
        _redirect(monkeypatch, mod, tmp)
    for mod in ALL:
        mod.run()


# --- staged ---

def test_staged_empty_start():
    out = subprocess.run(["git", "diff", "--cached", "--name-only"],
                         cwd=ROOT, capture_output=True, text=True)
    assert out.stdout.strip() == ""


# --- common helpers ---

def test_latex_escape_specials():
    assert D.latex_escape("a_b%c&d") == r"a\_b\%c\&d"

def test_latex_table_row():
    assert D.latex_table_row(["a", "b"]).endswith(r"\\")

def test_scan_forbidden_claims_detects_event_validated():
    assert D.scan_forbidden_claims("o evento validado em campo") != []

def test_scan_forbidden_claims_detects_ground_truth():
    assert D.scan_forbidden_claims("this is ground truth data") != []

def test_scan_forbidden_claims_detects_two_letter_label():
    assert D.scan_forbidden_claims("revisao por " + "A" + "I") != []

def test_scan_forbidden_claims_clean():
    assert D.scan_forbidden_claims("revisao automatizada review-only com DINO") == []

def test_evidence_matrix_cells():
    cells = D.evidence_matrix_cells({
        "external_evidence_status": "EXTERNAL_CANDIDATE_PRESENT_REVIEW_ONLY",
        "hydromet_status": "HYDROMET_CONTEXT_AVAILABLE_REVIEW_ONLY",
        "dino_status": "DINO_NOT_PRESENT_CONTEXT_ONLY",
        "patch_link_status": "PATCH_LINK_ABSENT", "event_window": "w"})
    assert cells["external_present"] == "true"
    assert cells["hydromet_context"] == "true"
    assert cells["dino_context"] == "false"


# --- v1tx dossier ---

def test_v1tx_dossiers_created(monkeypatch, tmp_path):
    _setup_inputs(tmp_path)
    _redirect(monkeypatch, v1tx, tmp_path)
    v1tx.run()
    rows = _read(tmp_path / v1tx.OUT_DOS.name)
    assert len(rows) == 1
    assert rows[0]["identificacao"]

def test_v1tx_sections_created(monkeypatch, tmp_path):
    _setup_inputs(tmp_path)
    _redirect(monkeypatch, v1tx, tmp_path)
    v1tx.run()
    secs = _read(tmp_path / v1tx.OUT_SEC.name)
    assert len(secs) == len(D.DOSSIER_SECTION_KEYS)

def test_v1tx_fail_closed(monkeypatch, tmp_path):
    _redirect(monkeypatch, v1tx, tmp_path)
    v1tx.run()
    rows = _read(tmp_path / v1tx.OUT_DOS.name)
    assert rows and rows[0]["case_id"] == "FAIL_CLOSED_NO_CASES"


# --- v1ty matrix ---

def test_v1ty_matrix_created(monkeypatch, tmp_path):
    _setup_inputs(tmp_path)
    for mod in (v1ty,):
        _redirect(monkeypatch, mod, tmp_path)
    v1ty.run()
    rows = _read(tmp_path / v1ty.OUT_MTX.name)
    assert rows and rows[0]["validated_for_review_only_use"] == "true"

def test_v1ty_guardrails_zero(monkeypatch, tmp_path):
    _setup_inputs(tmp_path)
    _redirect(monkeypatch, v1ty, tmp_path)
    v1ty.run()
    for r in _read(tmp_path / v1ty.OUT_MTX.name):
        for f in D.scan_guardrails([r], "x"):
            assert False, f


# --- v1tz fragments ---

def test_v1tz_fragments_created(monkeypatch, tmp_path):
    _setup_inputs(tmp_path)
    for mod in (v1ty, v1tz):
        _redirect(monkeypatch, mod, tmp_path)
    v1ty.run(); v1tz.run()
    rows = _read(tmp_path / v1tz.OUT_FRG.name)
    assert any(r["latex_row"].endswith(r"\\") for r in rows)


# --- v1ua narrative ---

def test_v1ua_narrative_created(monkeypatch, tmp_path):
    _setup_inputs(tmp_path)
    _redirect(monkeypatch, v1ua, tmp_path)
    v1ua.run()
    rows = _read(tmp_path / v1ua.OUT_NAR.name)
    assert any(r["scope"] == "case" for r in rows)
    assert any(r["scope"] == "global" for r in rows)


# --- v1ub claim audit + bundle ---

def test_v1ub_claim_safe_when_clean(monkeypatch, tmp_path):
    _run_chain(monkeypatch, tmp_path)
    sci = {r["metric_key"]: r["metric_value"]
           for r in _read(tmp_path / v1ub.OUT_SUM.name)}
    assert sci["claim_violations"] == "0"
    assert sci["final_status"] == "TCC_DOSSIER_BUNDLE_CLAIM_SAFE_READY_FOR_TCC"

def test_v1ub_detects_forbidden_claim(monkeypatch, tmp_path):
    _run_chain(monkeypatch, tmp_path)
    # inject a forbidden operational claim into the narrative output
    nar = tmp_path / "protocol_c_tcc_narrative_draft_v1ua.csv"
    rows = _read(nar)
    rows[0]["narrative_text"] = "o evento validado operacionalmente como ground truth"
    _write_csv(nar, rows, list(rows[0].keys()))
    v1ub.run()
    sci = {r["metric_key"]: r["metric_value"]
           for r in _read(tmp_path / v1ub.OUT_SUM.name)}
    assert int(sci["claim_violations"]) >= 1
    assert sci["final_status"] == "TCC_DOSSIER_BUNDLE_CLAIM_VIOLATION_FAIL_CLOSED"

def test_v1ub_manifest_present(monkeypatch, tmp_path):
    _run_chain(monkeypatch, tmp_path)
    assert len(_read(tmp_path / v1ub.OUT_MAN.name)) == len(v1ub.SCANNED)


# --- terminology guard + guardrails ---

_A = "A" + "I"; _B = "I" + "A"; _C = "a" + "i"
_PATS = [
    re.compile(r"\b" + _A + r"\b"), re.compile(r"\b" + _B + r"\b"),
    re.compile(_C + "_"), re.compile("_" + _C + "_"), re.compile("_" + _C + r"\b"),
    re.compile("(?i)" + "autonomous " + _A), re.compile("(?i)assistida por " + _B),
    re.compile(r"\b" + "LL" + "M" + r"\b"),
    re.compile(r"(?i)\b" + "cla" + "ude" + r"\b"), re.compile("(?i)" + "chat" + "gpt"),
]


def _dossier_versioned_files() -> list[Path]:
    out: list[Path] = []
    out += list(SCRIPTS.glob("revp_v1tx_*.py"))
    out += list(SCRIPTS.glob("revp_v1t[y-z]_*.py"))
    out += list(SCRIPTS.glob("revp_v1u[a-b]_*.py"))
    out += list(SCRIPTS.glob("revp_v1tx_v1ub_*.py"))
    out += list((ROOT / "datasets").glob("protocol_c_*_v1tx.csv"))
    out += list((ROOT / "datasets").glob("protocol_c_*_v1t[y-z].csv"))
    out += list((ROOT / "datasets").glob("protocol_c_*_v1u[a-b].csv"))
    out += list((ROOT / "datasets" / "schemas").glob("protocol_c_*_v1tx_schema.csv"))
    out += list((ROOT / "datasets" / "schemas").glob("protocol_c_*_v1t[y-z]_schema.csv"))
    out += list((ROOT / "datasets" / "schemas").glob("protocol_c_*_v1u[a-b]_schema.csv"))
    out += list((ROOT / "docs" / "metodologia_cientifica").glob("revp_v1tx_*.md"))
    out += list((ROOT / "docs" / "metodologia_cientifica").glob("revp_v1t[y-z]_*.md"))
    out += list((ROOT / "docs" / "metodologia_cientifica").glob("revp_v1u[a-b]_*.md"))
    out += [Path(__file__)]
    # exclude the refactor findings report (documents old terms by design)
    return [f for f in out if "terminology_refactor_findings" not in f.name]


def test_no_forbidden_labels_in_dossier_layer():
    offenders: list[str] = []
    for f in _dossier_versioned_files():
        for line in f.read_text(encoding="utf-8").splitlines():
            for pat in _PATS:
                if pat.search(line):
                    offenders.append(f"{f.name}: {line.strip()[:80]}")
    assert offenders == [], f"forbidden labels: {offenders[:5]}"


def test_chain_guardrails_zero(monkeypatch, tmp_path):
    _run_chain(monkeypatch, tmp_path)
    for f in tmp_path.glob("protocol_c_*_v1tx.csv"):
        assert D.scan_guardrails(_read(f), f.name) == []
    for f in tmp_path.glob("protocol_c_*_v1u[a-b].csv"):
        assert D.scan_guardrails(_read(f), f.name) == []


def test_dino_allowed_in_dossier():
    for pat in _PATS:
        assert not pat.search("DINO representation context review-only")


def test_staged_empty_end():
    out = subprocess.run(["git", "diff", "--cached", "--name-only"],
                         cwd=ROOT, capture_output=True, text=True)
    assert out.stdout.strip() == ""
