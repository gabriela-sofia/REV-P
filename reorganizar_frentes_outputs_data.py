#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reorganiza outputs_public/data/ em duas frentes de trabalho:

  outputs_public/data/linha_causal/       <- as 11 pastas susc_20* (SUSC-20, o produto final)
  outputs_public/data/linhagem_anterior/  <- as 26 pastas susc_17*/18*/19*/curitiba_lead*
                                              (pipeline exploratorio pre-causal)

Roda a partir da RAIZ do repo:

    python reorganizar_frentes_outputs_data.py

Usa `git mv` (preserva o historico de cada arquivo) e atualiza toda referencia
de codigo/doc/teste que aponta pros caminhos antigos, em todo o repositorio -
inclusive dentro das proprias pastas movidas (um script podia reconstruir o
proprio caminho inteiro a partir da raiz do repo, nao so um caminho relativo a
ele mesmo - isso tambem e corrigido). Testado ponta a ponta 3 vezes em clones
limpos: as 37 pastas saem do lugar certo, chegam no lugar certo, nenhuma
referencia fica orfa e a suite de testes roda igual antes e depois (mesma
contagem de passou/pulou, sem nenhum teste novo quebrando).

Depois de rodar: revise com `git status`, confira que nada te surpreende, e
faca o commit + push do jeito de sempre.
"""
import re
import subprocess
import sys
from pathlib import Path

CAUSAL = [
    "susc_20a_aquisicao_eventos_reais_recife",
    "susc_20b_engenharia_features_fisico_hidrologicas_recife",
    "susc_20c_modelagem_validacao_estatistica_rigorosa_recife",
    "susc_20d_motor_inferencia_local_mvp_recife",
    "susc_20e_api_contrato_inferencia_recife",
    "susc_20f_pipeline_geoprocessamento_sob_demanda_recife",
    "susc_20g_hand_twi_dinfinity_generico",
    "susc_20h_sentinel2_water_candidates",
    "susc_20i_janelas_evento_2023_2026",
    "susc_20j_sentinel1_sar_water_candidates",
    "susc_20k_siac156_curitiba_flood_candidates",
]
LEGACY = [
    "susc_17c5_geometry_to_patch_linkage_resolver",
    "susc_17c_strong_reference_acquisition_canary",
    "susc_17d_validacao_tecnica_evidencia_observacional",
    "susc_17e_prontidao_calibracao_observacional_exploratoria",
    "susc_17f_extracao_fisica_topografica_canarios_observacionais",
    "susc_17g_extracao_direta_features_fisicas_canarios",
    "susc_17h_calibracao_observacional_forte_somente_revisao",
    "susc_17i_ampliacao_regional_amostra_observacional",
    "susc_18a_execucao_referencia_observacional_regional",
    "susc_18b_execucao_geometrias_regionais_separacao_fenomeno",
    "susc_18c_aquisicao_geometria_oficial_curitiba",
    "susc_18d_protocolo_externo_curitiba",
    "susc_18e2_execucao_controlada_sentinel1_curitiba",
    "susc_18e_footprint_tecnico_sar_curitiba",
    "susc_18f_ingestao_validacao_footprint_sar_curitiba",
    "susc_18g_recuperacao_compactacao_vetorial_sar_curitiba",
    "susc_18h_consolidacao_mestre_cadeia_observacional",
    "susc_19a_matriz_multimodal_escalavel_por_patch",
    "susc_19b_auditoria_lacunas_territoriais",
    "susc_19c_avaliacao_observacional_review_only",
    "susc_19d_diagnostico_divergencias_score_v6_review_only",
    "susc_19e_pacote_comunicacao_cientifica_review_only",
    "susc_19f_execucao_mapbiomas_gee_ingestao_territorial",
    "susc_curitiba_leada_diario_oficial_tentativa",
    "susc_curitiba_leadb_ana_estacoes_reais",
    "susc_curitiba_leadc_global_flood_database",
]
FRENTE_DIR = {**{n: "linha_causal" for n in CAUSAL}, **{n: "linhagem_anterior" for n in LEGACY}}
ALL_NAMES = CAUSAL + LEGACY
TEXT_EXT = {".py", ".md", ".json", ".yml", ".yaml", ".txt", ".csv", ".cfg", ".ini"}

# fixes manuais: constantes "DATA"/"DATA_ROOT" genericas que so referenciam
# pastas legado nestes 3 arquivos (confirmado por leitura de codigo - nenhuma
# delas referencia nenhuma pasta susc_20*).
MANUAL_CONST_FIXES = [
    ("scripts/suscetibilidade/susc_18h_consolidacao_common.py",
     'DATA_ROOT = ROOT / "outputs_public" / "data"',
     'DATA_ROOT = ROOT / "outputs_public" / "data" / "linhagem_anterior"'),
    ("scripts/suscetibilidade/susc_19f_mapbiomas_common.py",
     'DATA = ROOT / "outputs_public" / "data"',
     'DATA = ROOT / "outputs_public" / "data" / "linhagem_anterior"'),
    ("scripts/suscetibilidade/susc_19e_comunicacao_cientifica_common.py",
     'DATA = ROOT / "outputs_public" / "data"',
     'DATA = ROOT / "outputs_public" / "data" / "linhagem_anterior"'),
]

# fixes cosmeticos: mencoes soltas em prosa (docs/README) que citam o nome da
# pasta sem o caminho completo na frente - nao quebra nada, so deixa preciso.
MANUAL_DOC_FIXES = [
    ("README.md",
     "`susc_20c_modelagem_validacao_estatistica_rigorosa_recife/reports/RELATORIO_v12_master.md`",
     "`outputs_public/data/linha_causal/susc_20c_modelagem_validacao_estatistica_rigorosa_recife/reports/RELATORIO_v12_master.md`"),
    ("README.md",
     "`susc_20j_sentinel1_sar_water_candidates/`",
     "`outputs_public/data/linha_causal/susc_20j_sentinel1_sar_water_candidates/`"),
    ("outputs_public/model/ESTADO_DO_MODELO.md",
     "`susc_20e_api_contrato_inferencia_recife/`",
     "`outputs_public/data/linha_causal/susc_20e_api_contrato_inferencia_recife/`"),
    ("docs/tcc_exports/planejamento_entrega01/esboco_telas_minimas_produto_v1.md",
     "(`susc_20e_api_contrato_inferencia_recife/`)",
     "(`outputs_public/data/linha_causal/susc_20e_api_contrato_inferencia_recife/`)"),
    ("docs/metodologia_cientifica/PLANO_ACAO_produto_v1.md",
     "(`susc_20e_api_contrato_inferencia_recife/`)",
     "(`outputs_public/data/linha_causal/susc_20e_api_contrato_inferencia_recife/`)"),
    ("docs/metodologia_cientifica/PLANO_ACAO_produto_v1.md",
     "`susc_20f_pipeline_geoprocessamento_sob_demanda_recife/`",
     "`outputs_public/data/linha_causal/susc_20f_pipeline_geoprocessamento_sob_demanda_recife/`"),
]


def sh(*args, cwd):
    r = subprocess.run(args, cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"FALHOU: {' '.join(args)}\n{r.stdout}\n{r.stderr}")
    return r.stdout


def main(repo_root: Path):
    data_dir = repo_root / "outputs_public" / "data"
    if not data_dir.is_dir():
        print(f"ERRO: {data_dir} nao existe. Rode este script a partir da raiz do repo REV-P.")
        sys.exit(1)

    moved, skipped = [], []
    for name in ALL_NAMES:
        src = data_dir / name
        dst = data_dir / FRENTE_DIR[name] / name
        if dst.exists():
            skipped.append(name)
            continue
        if not src.exists():
            print(f"AVISO: origem nao encontrada, pulando: {src}")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        sh("git", "mv", str(src.relative_to(repo_root)), str(dst.relative_to(repo_root)), cwd=repo_root)
        moved.append(name)
    if skipped:
        print(f"{len(skipped)} pastas ja estavam no lugar novo (script ja rodou antes?) - puladas.")
    print(f"movidas {len(moved)}/{len(ALL_NAMES)} pastas")

    # NOTA IMPORTANTE: NAO pulamos arquivos que estao dentro das pastas movidas.
    # Um script dentro de uma pasta pode reconstruir o proprio caminho completo a
    # partir da raiz do repo (em vez de um caminho relativo a ele mesmo) - isso
    # tambem precisa ser corrigido, senao ele aponta pra um lugar que nao existe
    # mais. Os regexes abaixo sao seguros de rodar em qualquer arquivo (inclusive
    # os que ja estao no lugar novo) porque o negative lookahead evita mexer de
    # novo em quem ja tem o segmento da frente.
    changed_files = []
    for path in repo_root.rglob("*"):
        if not path.is_file() or ".git" in path.parts or path.suffix not in TEXT_EXT:
            continue
        rel = path.relative_to(repo_root).as_posix()
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, PermissionError):
            continue
        orig = text
        for name in ALL_NAMES:
            frente = FRENTE_DIR[name]
            if name not in text:
                continue
            text = re.sub(
                r'("outputs_public"\s*/\s*"data"\s*/\s*)("%s")' % re.escape(name),
                r'\1"%s" / \2' % frente, text)
            text = re.sub(
                r'outputs_public/data/(?!linha_causal/|linhagem_anterior/)%s' % re.escape(name),
                f'outputs_public/data/{frente}/{name}', text)
            text = re.sub(
                r'outputs_public\\\\data\\\\(?!linha_causal\\\\|linhagem_anterior\\\\)%s' % re.escape(name),
                f'outputs_public\\\\data\\\\{frente}\\\\{name}', text)
            # caso D: referencia cruzada entre pastas de etapa via parents[N] +
            # join direto (ex.: sys.path.insert ou outra pasta lida a partir de
            # dentro de uma pasta de etapa), sem passar pelas strings literais
            # "outputs_public"/"data" no mesmo trecho - ex.:
            #   Path(__file__).resolve().parents[3] / "susc_20f_..." / "scripts"
            text = re.sub(
                r'(\.parents\[\d+\]\s*/\s*)("%s")' % re.escape(name),
                r'\1"%s" / \2' % frente, text)
        if text != orig:
            path.write_text(text, encoding="utf-8")
            changed_files.append(rel)
    print(f"referencias automaticamente corrigidas em {len(changed_files)} arquivos")

    # Dentro de CADA pasta movida, qualquer script que use
    # `Path(__file__).resolve().parents[N]` pra subir alem da propria pasta da
    # etapa (ex.: pra achar a raiz do repo) fica errado em exatamente +1, porque
    # inserimos um nivel novo (linha_causal/ ou linhagem_anterior/) no meio do
    # caminho. Deteta isso comparando o destino atual de parents[N] com o limite
    # da propria pasta da etapa, e corrige subindo N em 1.
    root_pat = re.compile(r'(Path\(__file__\)\.resolve\(\)\.parents)\[(\d+)\]')
    depth_fixed = []
    for name in ALL_NAMES:
        frente = FRENTE_DIR[name]
        base = data_dir / frente / name
        if not base.is_dir():
            continue
        for path in base.rglob("*.py"):
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, PermissionError):
                continue

            def _bump(m):
                n = int(m.group(2))
                candidate = path.resolve().parents[n] if n < len(path.resolve().parents) else None
                if candidate is None:
                    return m.group(0)
                try:
                    candidate.relative_to(base.resolve())
                    return m.group(0)  # ainda dentro da propria pasta da etapa, nao mexe
                except ValueError:
                    pass
                return f"{m.group(1)}[{n + 1}]"

            new_text = root_pat.sub(_bump, text)
            if new_text != text:
                path.write_text(new_text, encoding="utf-8")
                depth_fixed.append(str(path.relative_to(repo_root)))
    if depth_fixed:
        print(f"profundidade de parents[N] corrigida em {len(depth_fixed)} arquivo(s) dentro das pastas movidas:")
        for f in depth_fixed:
            print("  ", f)

    for relpath, old, new in MANUAL_CONST_FIXES:
        p = repo_root / relpath
        t = p.read_text(encoding="utf-8")
        if old in t:
            p.write_text(t.replace(old, new, 1), encoding="utf-8")
            print("fix manual aplicado:", relpath)

    for relpath, old, new in MANUAL_DOC_FIXES:
        p = repo_root / relpath
        t = p.read_text(encoding="utf-8")
        if old in t:
            p.write_text(t.replace(old, new, 1), encoding="utf-8")
            print("fix de doc aplicado:", relpath)

    # .gitignore tem uma regra de bloqueio geral (`outputs_public/data/*`) com
    # excecoes ponto a ponto pras 37 pastas antigas (`!outputs_public/data/<nome>/`).
    # Sem corrigir isso aqui, o `git mv` ainda funciona (arquivo ja rastreado
    # continua rastreado onde quer que va), mas qualquer ARQUIVO NOVO criado
    # dentro de uma pasta movida (ex.: um resultado novo que um script gerar no
    # futuro) fica invisivel pro git pra sempre, sem nenhum aviso. Corrige as
    # excecoes pra apontar pro caminho novo (com linha_causal/ ou
    # linhagem_anterior/ no meio).
    gitignore = repo_root / ".gitignore"
    if gitignore.is_file():
        t = gitignore.read_text(encoding="utf-8")
        orig = t
        for name in ALL_NAMES:
            frente = FRENTE_DIR[name]
            t = t.replace(f"!outputs_public/data/{name}/\n", f"!outputs_public/data/{frente}/{name}/\n")
            t = t.replace(f"!outputs_public/data/{name}/**\n", f"!outputs_public/data/{frente}/{name}/**\n")
        if t != orig:
            gitignore.write_text(t, encoding="utf-8")
            print(".gitignore atualizado: excecoes das 37 pastas apontam pro caminho novo")
        else:
            print("AVISO: nenhuma regra de .gitignore bateu com o padrao esperado - confira manualmente.")

    print("\nPronto. Confira com `git status`, depois commit + push do jeito de sempre.")


if __name__ == "__main__":
    main(Path.cwd())
