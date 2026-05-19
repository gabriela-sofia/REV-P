"""Tests for revp_v1gu_embedding_structural_evidence_package.py (corrected).

Covers: guardrails, allowed/forbidden claims, similarity computation,
top-k neighbors, intra/inter-region rate, regional centroids,
medoids/outliers, blocker document generation, corpus manifest index,
NPZ key priority, export functions.

v1gu uses canonical_patch_id (from v1fu manifest) as the authoritative
patch identifier. Corpus manifests (v1ge/v1fx/v1fz) are searched before
filesystem scan.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts" / "dino"
sys.path.insert(0, str(SCRIPTS_DIR))

from revp_v1gu_embedding_structural_evidence_package import (
    ALLOWED_CLAIMS,
    EMBEDDING_CORPUS_MANIFESTS,
    FIELD_PATCH_ID,
    FIELD_REGION,
    FORBIDDEN_CLAIMS,
    METHODOLOGICAL_GUARDRAILS,
    NPZ_EMBEDDING_KEYS,
    PHASE,
    SIMILARITY_METRIC,
    build_blocker_document,
    cosine_similarity,
    find_npz_for_patch,
    intra_inter_region_rate,
    load_corpus_manifest_index,
    load_embeddings_from_manifest,
    medoids_and_outliers,
    regional_centroids,
    similarity_matrix,
    top_k_neighbors,
    write_csv,
    write_json,
)


class TestGuardrails:
    def test_methodological_guardrails_locked(self) -> None:
        assert METHODOLOGICAL_GUARDRAILS["review_only"] is True
        assert METHODOLOGICAL_GUARDRAILS["supervised_training"] is False
        assert METHODOLOGICAL_GUARDRAILS["labels_created"] is False
        assert METHODOLOGICAL_GUARDRAILS["clustering_is_class"] is False
        assert METHODOLOGICAL_GUARDRAILS["dino_is_classifier"] is False

    def test_allowed_claims_present(self) -> None:
        assert "structural coherence" in ALLOWED_CLAIMS
        assert "exploratory similarity" in ALLOWED_CLAIMS
        assert len(ALLOWED_CLAIMS) > 0

    def test_forbidden_claims_present(self) -> None:
        assert "vulnerability prediction" in FORBIDDEN_CLAIMS
        assert "ground truth validation" in FORBIDDEN_CLAIMS
        assert len(FORBIDDEN_CLAIMS) > 0


class TestFieldMapping:
    def test_field_patch_id_is_canonical(self) -> None:
        assert FIELD_PATCH_ID == "canonical_patch_id"

    def test_field_region(self) -> None:
        assert FIELD_REGION == "region"


class TestSimilarityComputation:
    def test_cosine_identical(self) -> None:
        a = np.array([1.0, 0.0])
        assert np.isclose(cosine_similarity(a, a), 1.0)

    def test_cosine_orthogonal(self) -> None:
        a, b = np.array([1.0, 0.0]), np.array([0.0, 1.0])
        assert np.isclose(cosine_similarity(a, b), 0.0)

    def test_cosine_opposite(self) -> None:
        a, b = np.array([1.0, 0.0]), np.array([-1.0, 0.0])
        assert np.isclose(cosine_similarity(a, b), -1.0)

    def test_cosine_zero_vector(self) -> None:
        a, b = np.array([0.0, 0.0]), np.array([1.0, 0.0])
        assert cosine_similarity(a, b) == 0.0


class TestSimilarityMatrix:
    def test_shape(self) -> None:
        embeddings = {
            "PET_00001": np.array([1.0, 0.0]),
            "PET_00002": np.array([0.0, 1.0]),
            "CUR_00001": np.array([0.5, 0.5]),
        }
        matrix, ids = similarity_matrix(embeddings)
        assert len(matrix) == 3
        assert len(ids) == 3
        assert all(len(matrix[pid]) == 3 for pid in ids)

    def test_diagonal_ones(self) -> None:
        embeddings = {
            "PET_00001": np.array([1.0, 0.0]),
            "PET_00002": np.array([0.0, 1.0]),
        }
        matrix, ids = similarity_matrix(embeddings)
        for pid in ids:
            idx = ids.index(pid)
            assert np.isclose(matrix[pid][idx], 1.0)

    def test_symmetric(self) -> None:
        embeddings = {
            "PET_00001": np.array([1.0, 0.5]),
            "PET_00002": np.array([0.5, 1.0]),
        }
        matrix, ids = similarity_matrix(embeddings)
        assert np.isclose(matrix[ids[0]][1], matrix[ids[1]][0], atol=1e-6)


class TestTopKNeighbors:
    def test_ordering(self) -> None:
        embeddings = {
            "PET_00001": np.array([1.0, 0.0, 0.0]),
            "PET_00002": np.array([0.9, 0.1, 0.0]),
            "CUR_00001": np.array([0.0, 0.0, 1.0]),
        }
        neighbors = top_k_neighbors(embeddings, list(embeddings.keys()), k=2)
        p1 = neighbors["PET_00001"]
        assert p1[0]["rank"] == 1
        assert p1[0]["similarity"] >= p1[1]["similarity"]

    def test_excludes_self(self) -> None:
        embeddings = {
            "PET_00001": np.array([1.0, 0.0]),
            "PET_00002": np.array([0.0, 1.0]),
        }
        neighbors = top_k_neighbors(embeddings, list(embeddings.keys()), k=5)
        for pid, nbs in neighbors.items():
            for nb in nbs:
                assert nb["patch_id"] != pid

    def test_k_clipped_to_available(self) -> None:
        embeddings = {
            "PET_00001": np.array([1.0, 0.0]),
            "PET_00002": np.array([0.0, 1.0]),
        }
        neighbors = top_k_neighbors(embeddings, list(embeddings.keys()), k=10)
        assert all(len(nbs) == 1 for nbs in neighbors.values())


class TestIntraInterRegionRate:
    def _make_manifest(self, patch_region_pairs: list[tuple[str, str]]) -> list[dict[str, str]]:
        return [{"canonical_patch_id": pid, "region": region}
                for pid, region in patch_region_pairs]

    def test_all_intra(self) -> None:
        manifest = self._make_manifest([("PET_00001", "Petropolis"), ("PET_00002", "Petropolis")])
        neighbors = {
            "PET_00001": [{"patch_id": "PET_00002", "similarity": 0.9}],
            "PET_00002": [{"patch_id": "PET_00001", "similarity": 0.9}],
        }
        result = intra_inter_region_rate(manifest, ["PET_00001", "PET_00002"], neighbors)
        assert result["intra_region_rate"] == 1.0
        assert result["inter_region_rate"] == 0.0

    def test_all_inter(self) -> None:
        manifest = self._make_manifest([("PET_00001", "Petropolis"), ("CUR_00001", "Curitiba")])
        neighbors = {
            "PET_00001": [{"patch_id": "CUR_00001", "similarity": 0.5}],
            "CUR_00001": [{"patch_id": "PET_00001", "similarity": 0.5}],
        }
        result = intra_inter_region_rate(manifest, ["PET_00001", "CUR_00001"], neighbors)
        assert result["intra_region_rate"] == 0.0


class TestRegionalCentroids:
    def _make_manifest(self, patch_region_pairs: list[tuple[str, str]]) -> list[dict[str, str]]:
        return [{"canonical_patch_id": pid, "region": region}
                for pid, region in patch_region_pairs]

    def test_single_region(self) -> None:
        embeddings = {
            "PET_00001": np.array([1.0, 0.0]),
            "PET_00002": np.array([0.0, 1.0]),
        }
        manifest = self._make_manifest([("PET_00001", "Petropolis"), ("PET_00002", "Petropolis")])
        centroids = regional_centroids(embeddings, manifest)
        assert "Petropolis" in centroids
        assert centroids["Petropolis"]["n_patches"] == 2

    def test_multiple_regions(self) -> None:
        embeddings = {
            "PET_00001": np.array([1.0, 0.0]),
            "CUR_00001": np.array([0.0, 1.0]),
        }
        manifest = self._make_manifest([("PET_00001", "Petropolis"), ("CUR_00001", "Curitiba")])
        centroids = regional_centroids(embeddings, manifest)
        assert len(centroids) == 2


class TestMedoidsAndOutliers:
    def _make_manifest(self, patch_region_pairs: list[tuple[str, str]]) -> list[dict[str, str]]:
        return [{"canonical_patch_id": pid, "region": region}
                for pid, region in patch_region_pairs]

    def test_single_patch(self) -> None:
        embeddings = {"PET_00001": np.array([1.0, 0.0])}
        manifest = self._make_manifest([("PET_00001", "Petropolis")])
        medoids = medoids_and_outliers(embeddings, manifest)
        assert medoids["Petropolis"]["medoid"] == "PET_00001"
        assert medoids["Petropolis"]["outliers"] == []
        assert "single patch region" in medoids["Petropolis"]["info"]

    def test_multiple_patches(self) -> None:
        embeddings = {
            "PET_00001": np.array([1.0, 0.0]),
            "PET_00002": np.array([1.0, 0.0]),
            "PET_00003": np.array([0.0, 1.0]),
        }
        manifest = self._make_manifest([
            ("PET_00001", "Petropolis"),
            ("PET_00002", "Petropolis"),
            ("PET_00003", "Petropolis"),
        ])
        medoids = medoids_and_outliers(embeddings, manifest)
        assert "medoid" in medoids["Petropolis"]
        assert "outliers" in medoids["Petropolis"]
        assert medoids["Petropolis"]["n_patches"] == 3


class TestBlockerDocument:
    def test_write_json_creates_file(self, tmp_path: Path) -> None:
        data = {"blocker_code": "NO_NPZ_EMBEDDINGS_FOUND", "n_embeddings_missing": 128}
        output_file = tmp_path / "blocker.json"
        write_json(output_file, data)
        assert output_file.exists()
        with output_file.open() as f:
            loaded = json.load(f)
        assert loaded["blocker_code"] == "NO_NPZ_EMBEDDINGS_FOUND"

    def test_write_csv_creates_file(self, tmp_path: Path) -> None:
        rows = [
            {"canonical_patch_id": "CUR_00038", "region": "Curitiba",
             "embedding_status": "MISSING", "blocker": "NO_NPZ_FILE_FOUND_IN_SEARCH_DIRS"},
        ]
        output_file = tmp_path / "patch_status.csv"
        write_csv(output_file, rows, ["canonical_patch_id", "region", "embedding_status", "blocker"])
        assert output_file.exists()
        with output_file.open() as f:
            data = list(csv.DictReader(f))
        assert len(data) == 1
        assert data[0]["canonical_patch_id"] == "CUR_00038"


class TestFindNpz:
    def test_find_npz_returns_none_when_no_files(self) -> None:
        result = find_npz_for_patch("NONEXISTENT_PATCH_12345")
        assert result is None

    def test_find_npz_uses_corpus_index_first(self, tmp_path: Path) -> None:
        npz_path = tmp_path / "DINO_V1FU_SENTINEL_00001.npz"
        arr = np.array([1.0, 0.0], dtype="float32")
        np.savez_compressed(npz_path, cls_embedding=arr)
        corpus_index = {"CUR_00038": npz_path}
        result = find_npz_for_patch("CUR_00038", corpus_index)
        assert result == npz_path

    def test_find_npz_returns_none_when_corpus_index_empty(self) -> None:
        result = find_npz_for_patch("CUR_00038", corpus_index={})
        assert result is None


class TestNpzKeyPriority:
    def test_cls_embedding_key_loaded(self, tmp_path: Path) -> None:
        npz_path = tmp_path / "patch.npz"
        arr = np.array([0.5, 0.5], dtype="float32")
        np.savez_compressed(npz_path, cls_embedding=arr)
        manifest = [{"canonical_patch_id": "CUR_00038", "region": "Curitiba"}]
        corpus_index = {"CUR_00038": npz_path}
        embeddings, missing = load_embeddings_from_manifest(manifest, corpus_index)
        assert "CUR_00038" in embeddings
        assert missing == []
        assert np.allclose(embeddings["CUR_00038"], arr)

    def test_patch_mean_embedding_key_loaded(self, tmp_path: Path) -> None:
        npz_path = tmp_path / "patch.npz"
        arr = np.array([0.3, 0.7], dtype="float32")
        np.savez_compressed(npz_path, patch_mean_embedding=arr)
        manifest = [{"canonical_patch_id": "PET_00016", "region": "Petropolis"}]
        corpus_index = {"PET_00016": npz_path}
        embeddings, missing = load_embeddings_from_manifest(manifest, corpus_index)
        assert "PET_00016" in embeddings
        assert missing == []

    def test_missing_patch_reported(self) -> None:
        manifest = [{"canonical_patch_id": "CUR_00001", "region": "Curitiba"}]
        embeddings, missing = load_embeddings_from_manifest(manifest, corpus_index={})
        assert "CUR_00001" in missing
        assert embeddings == {}

    def test_npz_keys_order(self) -> None:
        assert NPZ_EMBEDDING_KEYS[0] == "cls_embedding"
        assert NPZ_EMBEDDING_KEYS[1] == "patch_mean_embedding"


class TestCorpusManifestIndex:
    def test_returns_empty_when_manifests_missing(self) -> None:
        index, audit = load_corpus_manifest_index()
        assert isinstance(index, dict)
        assert len(audit) == len(EMBEDDING_CORPUS_MANIFESTS)
        # When no manifest files exist, all show exists=False
        not_found = [a for a in audit if not a["exists"]]
        assert len(not_found) > 0

    def test_audit_has_required_fields(self) -> None:
        _, audit = load_corpus_manifest_index()
        for record in audit:
            assert "manifest" in record
            assert "exists" in record
            assert "n_rows" in record
            assert "n_success_rows" in record
            assert "n_npz_resolved" in record

    def test_corpus_manifests_constant_has_three_entries(self) -> None:
        assert len(EMBEDDING_CORPUS_MANIFESTS) == 3

    def test_index_populated_from_mock_manifest(self, tmp_path: Path) -> None:
        npz_file = tmp_path / "embeddings" / "DINO_V1FU_SENTINEL_00001.npz"
        npz_file.parent.mkdir()
        arr = np.array([1.0, 0.0], dtype="float32")
        np.savez_compressed(npz_file, cls_embedding=arr)

        manifest_csv = tmp_path / "dino_expanded_embedding_manifest_v1ge.csv"
        with manifest_csv.open("w", encoding="utf-8", newline="") as f:
            import csv as _csv
            w = _csv.DictWriter(f, fieldnames=["patch_id", "dino_input_id", "region", "embedding_path", "success"])
            w.writeheader()
            w.writerow({
                "patch_id": "CUR_00038",
                "dino_input_id": "DINO_V1FU_SENTINEL_00001",
                "region": "Curitiba",
                "embedding_path": "embeddings/DINO_V1FU_SENTINEL_00001.npz",
                "success": "SUCCESS",
            })

        import importlib
        import revp_v1gu_embedding_structural_evidence_package as mod
        orig = mod.EMBEDDING_CORPUS_MANIFESTS[:]
        mod.EMBEDDING_CORPUS_MANIFESTS[:] = [manifest_csv]
        try:
            index, audit = load_corpus_manifest_index()
            assert "CUR_00038" in index
            assert audit[0]["n_success_rows"] == 1
            assert audit[0]["n_npz_resolved"] == 1
        finally:
            mod.EMBEDDING_CORPUS_MANIFESTS[:] = orig


class TestBlockerDocumentFull:
    def test_blocker_document_has_corpus_audit(self, tmp_path: Path) -> None:
        manifest = [
            {"canonical_patch_id": "CUR_00038", "region": "Curitiba"},
            {"canonical_patch_id": "PET_00016", "region": "Petropolis"},
        ]
        corpus_audit = [{"manifest": "fake/path.csv", "exists": False,
                         "n_rows": 0, "n_success_rows": 0, "n_npz_resolved": 0}]
        build_blocker_document(manifest, ["CUR_00038", "PET_00016"], tmp_path, corpus_audit)
        blocker_file = tmp_path / "embedding_structural_evidence_blocker_v1gu.json"
        assert blocker_file.exists()
        data = json.loads(blocker_file.read_text(encoding="utf-8"))
        assert "corpus_manifest_audit" in data
        assert "corpus_manifests_checked" in data
        assert "corpus_manifests_missing" in data
        assert "n_npz_resolved_from_manifests" in data

    def test_blocker_code_when_manifests_missing(self, tmp_path: Path) -> None:
        manifest = [{"canonical_patch_id": "CUR_00038", "region": "Curitiba"}]
        corpus_audit = [{"manifest": "fake/v1ge.csv", "exists": False,
                         "n_rows": 0, "n_success_rows": 0, "n_npz_resolved": 0}]
        build_blocker_document(manifest, ["CUR_00038"], tmp_path, corpus_audit)
        data = json.loads((tmp_path / "embedding_structural_evidence_blocker_v1gu.json").read_text(encoding="utf-8"))
        assert data["blocker_code"] == "CORPUS_MANIFESTS_NOT_FOUND_EMBEDDINGS_NOT_EXTRACTED"

    def test_blocker_document_without_corpus_audit(self, tmp_path: Path) -> None:
        manifest = [{"canonical_patch_id": "CUR_00038", "region": "Curitiba"}]
        build_blocker_document(manifest, ["CUR_00038"], tmp_path)
        assert (tmp_path / "embedding_structural_evidence_blocker_v1gu.json").exists()

    def test_v1fu_corpus_separate_from_executed_corpus(self) -> None:
        # v1fu manifest has 128 patches with pixel_read_status=NOT_READ__FUTURE_DINO_ENCODING_ONLY
        # The executed corpus (v1ge/v1fx/v1fz) is separate and may have fewer patches
        # This test confirms the distinction is preserved in the blocker document
        corpus_audit = [
            {"manifest": "v1ge/manifest.csv", "exists": True,
             "n_rows": 64, "n_success_rows": 0, "n_npz_resolved": 0},
        ]
        manifest_64 = [{"canonical_patch_id": f"CUR_{i:05d}", "region": "Curitiba"} for i in range(64)]
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            build_blocker_document(manifest_64, [f"CUR_{i:05d}" for i in range(64)],
                                   Path(td), corpus_audit)
            data = json.loads((Path(td) / "embedding_structural_evidence_blocker_v1gu.json").read_text(encoding="utf-8"))
        assert data["blocker_code"] == "CORPUS_MANIFESTS_EXIST_BUT_NO_NPZ_ON_DISK"
        assert data["n_patches_in_manifest"] == 64


class TestPhaseAndMetadata:
    def test_phase_constant(self) -> None:
        assert PHASE == "v1gu"

    def test_similarity_metric(self) -> None:
        assert SIMILARITY_METRIC == "cosine"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
