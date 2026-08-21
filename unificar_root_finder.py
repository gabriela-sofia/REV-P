#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Corrige a causa raiz da fragilidade do repo: centenas de scripts descobrem a
raiz do projeto contando "quantas pastas eu tenho que subir a partir de mim
mesmo" (`Path(__file__).resolve().parents[2]`, hardcoded por arquivo). Isso
quebra silenciosamente toda vez que um arquivo muda de profundidade na arvore
- foi exatamente isso que aconteceu com 21 scripts durante a reorganizacao de
outputs_public/data/ (achado e corrigido antes de ir pro seu repositorio).

Este script troca cada uma dessas contagens fixas por uma busca que sobe a
arvore de pastas ate achar a raiz de verdade (onde estao `.git` e
`environment.yml`) - passa a funcionar em qualquer profundidade, para sempre.
Nao muda o NOME de nenhuma constante (ROOT, REPO_ROOT, PROJECT_ROOT, etc.) nem
o que vem depois dela (`/ "outputs_public" / ...`) - so a forma de achar o
ponto de partida.

So mexe em constantes que hoje realmente apontam pra raiz do repo (confirmado
lendo o disco antes de trocar, arquivo por arquivo). Constantes que apontam
pra uma pasta local (tipo "a propria pasta da etapa") ficam como estao -
trocar essas seria errado.

Roda a partir da RAIZ do repo:

    python unificar_root_finder.py

Depois: `git status`, confere, roda a suite de testes se quiser (`python -m
pytest tests -q`), e commit + push do jeito de sempre.
"""
import re
import subprocess
from pathlib import Path

PAT = re.compile(r'Path\(__file__\)\.resolve\(\)\.parents\[(\d+)\]')
NEW_EXPR = (
    'next(_p for _p in (Path(__file__).resolve(), *Path(__file__).resolve().parents) '
    'if (_p / ".git").is_dir() and (_p / "environment.yml").is_file())'
)


def is_true_root(candidate: Path) -> bool:
    return (candidate / ".git").is_dir() and (candidate / "environment.yml").is_file()


def main(repo_root: Path):
    if not (repo_root / ".git").is_dir():
        raise SystemExit(f"ERRO: {repo_root} nao parece ser a raiz do repo REV-P.")

    touched = []
    skipped_local = []
    for path in repo_root.rglob("*.py"):
        if ".git" in path.parts:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, PermissionError):
            continue

        matches = list(PAT.finditer(text))
        if not matches:
            continue

        resolved_file = path.resolve()
        n_fixed = 0

        def _replace(m):
            nonlocal n_fixed
            n = int(m.group(1))
            candidate = resolved_file.parents[n] if n < len(resolved_file.parents) else None
            if candidate is not None and is_true_root(candidate):
                n_fixed += 1
                return NEW_EXPR
            skipped_local.append((str(path.relative_to(repo_root)), m.group(0)))
            return m.group(0)

        rebuilt = PAT.sub(_replace, text)
        if n_fixed:
            path.write_text(rebuilt, encoding="utf-8")
            touched.append(str(path.relative_to(repo_root)))

    print(f"arquivos corrigidos: {len(touched)}")
    print(f"ocorrencias mantidas como estavam (ancora local, nao e raiz do repo): {len(skipped_local)}")
    if skipped_local:
        print("  (revisao manual opcional, nenhuma foi tocada):")
        for f, expr in skipped_local[:20]:
            print("   ", f, "::", expr)
        if len(skipped_local) > 20:
            print(f"    ... e mais {len(skipped_local) - 20}")

    print("\nPronto. Confira com `git status` e `python -m pytest tests -q`, depois commit + push.")


if __name__ == "__main__":
    main(Path.cwd())
