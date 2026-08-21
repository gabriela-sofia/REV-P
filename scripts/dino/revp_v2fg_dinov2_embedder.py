"""REV-P v2fg -- `Dinov2Embedder`: extrator de embedding visual DINOv2 768D.

Camada de REPRESENTAÇÃO da governança visual (v2fg). Produz o vetor; não
decide nada. A decisão (domínio, medoid territorial, gate OOD) fica em
`revp_v2fg_dinov2_governance_engine.py`.

Fronteira metodológica (nunca cruzada por este módulo):
  * O embedding NÃO entra no modelo físico de Firth nem em seus coeficientes.
  * O embedding NÃO é rótulo, alvo, classe ou confirmação de evento.
  * Similaridade não é validação operacional de inundação/deslizamento.

Backend real: `transformers.AutoModel` + `AutoImageProcessor` sobre
`facebook/dinov2-with-registers-base` (mesmo backbone já usado em
v1qj/v1r8 -- ver `datasets/dino_*_embeddings_*.csv`, coluna `model_name`).
Vetor = token CLS de `last_hidden_state`, com L2-normalização (mesma regra
do executor v1qj, `_embed(..., l2=True)`).

Modo MOCK: determinístico, derivado do SHA-256 do arquivo, e existe SÓ para
teste. Precisa de opt-in explícito (`mock=True` ou
`REVP_DINOV2_ALLOW_MOCK=true`) e todo vetor produzido sai marcado com
`backend="mock"`. O pipeline de corpus (v2fg) recusa linha `mock` — mock
nunca vira corpus científico nem mascara ausência de dado real.
"""
from __future__ import annotations

import hashlib
import math
import os
import struct
from pathlib import Path
from typing import Any

MODEL_NAME = "facebook/dinov2-with-registers-base"
EMBEDDING_DIM = 768

#: Tolerância de norma L2 aceita como "unitário" (float32 acumulado em 768D).
L2_TOLERANCE = 1e-4

BACKEND_TORCH = "torch"
BACKEND_MOCK = "mock"
BACKEND_UNAVAILABLE = "unavailable"


def _env_true(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def l2_normalize(vec: list[float]) -> list[float] | None:
    """Normaliza para norma 1. Retorna None se a norma for 0/não-finita --
    vetor degenerado nunca vira embedding válido."""
    norm = math.sqrt(sum(float(x) * float(x) for x in vec))
    if not math.isfinite(norm) or norm <= 0.0:
        return None
    return [float(x) / norm for x in vec]


def l2_norm(vec: list[float]) -> float:
    return math.sqrt(sum(float(x) * float(x) for x in vec))


def is_unit_norm(vec: list[float], tolerance: float = L2_TOLERANCE) -> bool:
    return abs(l2_norm(vec) - 1.0) <= tolerance


class Dinov2Embedder:
    """Extrai embeddings 768D L2-normalizados com `facebook/dinov2-with-registers-base`.

    Parâmetros
    ----------
    model_path:
        Caminho local dos pesos. Default: `REVP_DINO_MODEL_PATH` (mesma
        variável já usada por v1pp/v1qj) e, se vazia, o próprio
        `MODEL_NAME` do Hugging Face.
    device:
        `"cpu"`, `"cuda"`, `"cuda:0"`... Default: `REVP_DINOV2_DEVICE` e,
        se vazia, `cuda` quando `torch.cuda.is_available()`, senão `cpu`.
    allow_download:
        Se False (default, igual ao resto do pipeline DINO do projeto), o
        `from_pretrained` roda com `local_files_only=True` -- fail-closed,
        sem baixar peso em silêncio. Espelha `REVP_DINO_ALLOW_DOWNLOAD`.
    mock:
        Opt-in explícito do backend determinístico de teste. Ver docstring
        do módulo.
    """

    def __init__(
        self,
        model_path: str | None = None,
        device: str | None = None,
        allow_download: bool | None = None,
        l2_normalize_output: bool = True,
        mock: bool = False,
    ) -> None:
        self.model_name = MODEL_NAME
        self.embedding_dim = EMBEDDING_DIM
        self.model_path = model_path if model_path is not None else (
            os.environ.get("REVP_DINO_MODEL_PATH", "").strip() or MODEL_NAME)
        self.allow_download = (
            _env_true("REVP_DINO_ALLOW_DOWNLOAD", False) if allow_download is None else allow_download)
        self.l2_normalize_output = l2_normalize_output
        self._mock = bool(mock) or _env_true("REVP_DINOV2_ALLOW_MOCK", False)
        self._requested_device = device if device is not None else os.environ.get("REVP_DINOV2_DEVICE", "").strip()
        self._device = self._requested_device or None
        self._bundle: Any | None = None
        self._load_error: str = ""
        self._backend = BACKEND_MOCK if self._mock else BACKEND_UNAVAILABLE

    # -- introspecção -------------------------------------------------- #

    @property
    def backend(self) -> str:
        """`"torch"`, `"mock"` ou `"unavailable"`. Só é definitivo depois de
        `available()`/`embed_image()` -- o carregamento é preguiçoso."""
        return self._backend

    @property
    def device(self) -> str:
        return self._device or "cpu"

    @property
    def is_mock(self) -> bool:
        return self._backend == BACKEND_MOCK

    @property
    def load_error(self) -> str:
        return self._load_error

    def available(self) -> bool:
        """True se há backend capaz de produzir vetor. Em modo mock retorna
        True, mas `is_mock` fica True junto -- o chamador é obrigado a
        distinguir."""
        if self._mock:
            self._backend = BACKEND_MOCK
            return True
        return self._load() is not None

    def describe(self) -> dict[str, Any]:
        """Proveniência do extrator, para carimbar em manifesto/auditoria."""
        return {
            "model_name": self.model_name,
            "model_path_configured": self.model_path,
            "embedding_dim": self.embedding_dim,
            "l2_normalized": self.l2_normalize_output,
            "backend": self._backend,
            "device": self.device,
            "allow_download": self.allow_download,
            "is_mock": self.is_mock,
            "load_error": self._load_error,
        }

    # -- backend real --------------------------------------------------- #

    def _resolve_device(self) -> str:
        if self._requested_device:
            return self._requested_device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _load(self) -> Any | None:
        if self._mock:
            self._backend = BACKEND_MOCK
            return None
        if self._bundle is not None:
            return self._bundle
        if self._load_error:
            return None
        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel

            self._device = self._resolve_device()
            local_only = not self.allow_download
            processor = AutoImageProcessor.from_pretrained(self.model_path, local_files_only=local_only)
            model = AutoModel.from_pretrained(self.model_path, local_files_only=local_only)
            model.eval()
            model.to(self._device)
            self._bundle = (processor, model, torch)
            self._backend = BACKEND_TORCH
            return self._bundle
        except Exception as exc:  # backend ausente/pesos ausentes: fail-closed
            self._load_error = f"{type(exc).__name__}: {exc}"
            self._backend = BACKEND_UNAVAILABLE
            self._device = self._requested_device or "cpu"
            return None

    # -- extração ------------------------------------------------------- #

    def embed_image(self, image_path: str | Path) -> list[float] | None:
        """Retorna o vetor 768D (L2-normalizado por default) ou None.

        None significa literalmente "não foi possível extrair" -- nunca é
        substituído por vetor sintético fora do modo mock explícito.
        """
        path = Path(image_path)
        if not path.exists():
            return None
        if self._mock:
            return self._mock_vector(path)
        bundle = self._load()
        if bundle is None:
            return None
        processor, model, torch = bundle
        try:
            from PIL import Image

            with Image.open(path) as raw:
                img = raw.convert("RGB")
            inputs = processor(images=img, return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
            if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
                vec = outputs.last_hidden_state[0, 0, :].detach().cpu().tolist()
            elif hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                vec = outputs.pooler_output[0].detach().cpu().tolist()
            else:
                return None
        except Exception as exc:
            self._load_error = f"{type(exc).__name__}: {exc}"
            return None
        return self._finalize(vec)

    def embed_images(self, image_paths: list[str | Path]) -> dict[str, list[float] | None]:
        """Extração em lote. Chave = `str(path)`; valor None onde falhou."""
        return {str(p): self.embed_image(p) for p in image_paths}

    def _finalize(self, vec: list[float]) -> list[float] | None:
        vec = [float(x) for x in vec]
        if len(vec) != self.embedding_dim:
            self._load_error = f"dimensao_inesperada: {len(vec)} != {self.embedding_dim}"
            return None
        if not all(math.isfinite(x) for x in vec):
            self._load_error = "vetor_com_valor_nao_finito"
            return None
        if not self.l2_normalize_output:
            return vec
        return l2_normalize(vec)

    # -- mock determinístico (só teste) --------------------------------- #

    def _mock_vector(self, path: Path) -> list[float] | None:
        """Vetor determinístico derivado do SHA-256 do arquivo.

        NÃO é embedding DINOv2. Existe para exercitar contrato/dimensão/
        normalização em teste sem pesos locais. Todo consumidor deve checar
        `is_mock` antes de tratar o vetor como evidência.
        """
        digest = hashlib.sha256(path.read_bytes()).digest()
        raw: list[float] = []
        counter = 0
        while len(raw) < self.embedding_dim:
            block = hashlib.sha256(digest + struct.pack(">I", counter)).digest()
            for i in range(0, len(block), 4):
                if len(raw) >= self.embedding_dim:
                    break
                (word,) = struct.unpack(">I", block[i:i + 4])
                raw.append((word / 0xFFFFFFFF) * 2.0 - 1.0)
            counter += 1
        return self._finalize(raw)


__all__ = [
    "MODEL_NAME", "EMBEDDING_DIM", "L2_TOLERANCE",
    "BACKEND_TORCH", "BACKEND_MOCK", "BACKEND_UNAVAILABLE",
    "Dinov2Embedder", "l2_normalize", "l2_norm", "is_unit_norm",
]
