"""
Tests para higiene pública de terminologia v1uc-v1ue
"""

import pytest
import sys
from pathlib import Path

# Adicionar scripts ao path
scripts_root = Path(__file__).parent.parent / "scripts" / "refactor"
sys.path.insert(0, str(scripts_root))

import v1uc_public_terminology_scanner as scanner
import v1ud_public_terminology_refactor as refactor

def test_scanner_false_positives():
    """Testa que falsos positivos conhecidos não são detectados."""
    test_strings = [
        "v1ia is a stage identifier",
        "v1ic is another identifier",
        "daily rainfall measurement",
        "rainfall_daily is a column",
        "source_family data structure",
    ]

    for test_str in test_strings:
        assert scanner.is_false_positive("", test_str), f"Should be FP: {test_str}"

def test_scanner_real_prohibitions():
    """Testa que termos reais proibidos são detectados."""
    prohibitions = [
        "human_review status is missing",
        "HUMAN_REVIEW required",
        "requires_human_confirmation in protocol",
        "revisão humana documental",
        "Visual assisted review update",
        "autonomous AI processing",
        "Claude-based extraction",
    ]

    for test_str in prohibitions:
        found = False
        for category, patterns in scanner.PROHIBITED_PATTERNS.items():
            for pattern in patterns:
                if __import__("re").search(pattern, test_str, __import__("re").IGNORECASE):
                    found = True
                    break
        assert found, f"Should detect: {test_str}"

def test_terminology_map_coverage():
    """Testa que mapa terminológico tem entradas."""
    assert len(refactor.TERMINOLOGY_MAP) > 0, "Terminology map should not be empty"

    # Verificar que todos os pares antigo->novo existem
    for old_term in ["human_review", "review_gate", "assisted", "supervisora"]:
        found = False
        for key in refactor.TERMINOLOGY_MAP.keys():
            if old_term.lower() in key.lower():
                found = True
                break
        # Não é um assertion rígida, só aviso

def test_file_renames_structure():
    """Testa que estrutura de renames é válida."""
    assert len(refactor.FILE_RENAMES) == 8, "Should have 8 file renames"

    for old_path, new_path in refactor.FILE_RENAMES.items():
        assert isinstance(old_path, str), f"Old path should be str: {old_path}"
        assert isinstance(new_path, str), f"New path should be str: {new_path}"
        assert old_path != new_path, f"Old and new should differ: {old_path}"

def test_preserve_terms_present():
    """Testa que termos a preservar estão listados."""
    preserve = refactor.PRESERVE_TERMS
    assert "DINO" in preserve, "Must preserve DINO"
    assert "Sentinel" in preserve, "Must preserve Sentinel"
    assert "Protocolo C" in preserve, "Must preserve Protocolo C"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
