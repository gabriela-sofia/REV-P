"""Figura 2 do Documento de Planejamento — quatro desafios, com dados reais do projeto.

Fontes lidas (todas artefatos versionados do projeto):
  (a) PROJETO/local_runs/recife_modelo_v12_extracao_final/dataset_v12_final.csv
      REV-P/outputs_public/data/susc_20k_siac156_curitiba_flood_candidates/
            registries/v20n_dataset_curitiba_features_v2.csv
      REV-P/local_runs/mod-uk-01-firth/resumo_rnl.json      (UK, 7.476 pontos)
      REV-P/local_runs/mod-serra-01/resultado.json          (CEMS serra/planicie)
  (b) dataset_v12_final.csv  (HAND D-infinity de Recife)
  (c) REV-P/local_runs/mod-neg-01/resumo.json               (aplicacao cruzada)
  (d) results/v20q_walk_forward_cutoffs.csv                 (walk-forward Curitiba)

Uso:  python fig/make_fig2.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"font.family": "serif", "font.serif": ["DejaVu Serif"],
                     "font.size": 6.2, "axes.linewidth": 0.6, "axes.labelsize": 6.2,
                     "xtick.labelsize": 5.4, "ytick.labelsize": 5.4,
                     "legend.fontsize": 5.0, "pdf.fonttype": 42})

# --- raizes: ajustar apenas estas duas linhas se o repositorio for movido -------
REVP = os.environ.get("REVP_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))
PROJ = os.environ.get("PROJETO_ROOT", os.path.abspath(os.path.join(REVP, "..", "PROJETO")))

REC = os.path.join(PROJ, "local_runs", "recife_modelo_v12_extracao_final", "dataset_v12_final.csv")
CUR = os.path.join(REVP, "outputs_public", "data", "susc_20k_siac156_curitiba_flood_candidates",
                   "registries", "v20n_dataset_curitiba_features_v2.csv")
SERRA = os.path.join(REVP, "local_runs", "mod-serra-01", "resultado.json")
NEG = os.path.join(REVP, "local_runs", "mod-neg-01", "resumo.json")

rec = pd.read_csv(REC)
cur = pd.read_csv(CUR)
cur_u = cur.drop_duplicates(subset=["observation_unit_key"]) if "observation_unit_key" in cur else cur
serra = json.load(open(SERRA))
neg = json.load(open(NEG))

C_POS = "#1f4e79"
C_NEG = "#c0504d"
C_AC = "#7f7f7f"

fig, ax = plt.subplots(2, 2, figsize=(3.45, 1.44))

# ---- (a) composicao da amostra por conjunto, escala log -----------------------
# CEMS: contagens do modelo de encosta (classe 0/1; classe 2 = massa fica fora).
cems_serra_n, cems_serra_pos = 4475, 1835
cems_plan_n, cems_plan_pos = 20774, 9134

nomes = ["REC", "CTB", "PET", "UK", "SER", "PLA"]
pos = [int((rec.label == 1).sum()), int((cur_u.label == 1).sum()), 1,
       3738, cems_serra_pos, cems_plan_pos]
neg_ = [int((rec.label == 0).sum()), int((cur_u.label == 0).sum()), 0,
        3738, cems_serra_n - cems_serra_pos, cems_plan_n - cems_plan_pos]

a = ax[0, 0]
x = np.arange(len(nomes))
w = 0.36
a.bar(x - w / 2, [max(v, 0.6) for v in pos], w, color=C_POS, label="positivo")
a.bar(x + w / 2, [max(v, 0.6) for v in neg_], w, color=C_NEG, label="negativo")
a.set_yscale("log")
a.set_ylim(0.6, 2e7)
for i, (pp, nn) in enumerate(zip(pos, neg_)):
    a.text(i - w / 2, max(pp, 0.6) * 1.35, f"{pp}", ha="center", fontsize=3.8, rotation=90, va="bottom")
    a.text(i + w / 2, max(nn, 0.6) * 1.35, f"{nn}", ha="center", fontsize=3.8, rotation=90, va="bottom")
a.set_xticks(x)
a.set_xticklabels(nomes)
a.set_ylabel("pontos")
a.set_title("(a) amostra por conjunto", fontsize=6.2, pad=3)

# ---- (b) heterogeneidade: variavel disponivel por fonte -----------------------
# 1 = disponivel na cadeia atual; 0 = ausente. Fatos documentados em
# ext_analogos_de_petropolis_v1.md (chuva so no piloto UK e nas duas regioes
# brasileiras, nenhuma delas no ajuste fluvial), ext_chuva_fonte_unica_recife_v1.md
# (chuva unica em Recife, 2026-08-16), ext_resolucao_unica_30m_v2.md (declividade
# e TWI derivadas para Nivel 1/Sen1Floods11+UFO via ter06, 2026-08-14 -- antes
# vazias por decisao do ds01) e ext_balanco (mecanismo na origem).
a = ax[0, 1]
fontes = ["Recife", "Curitiba", "UK/EA", "CEMS", "Nível 1"]
variaveis = ["elev", "decl", "HAND", "TWI", "chuva", "mec."]
M = np.array([
    [1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 0],
    [1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 0, 1],
    [1, 1, 1, 1, 0, 0],
])
a.imshow(M, cmap=matplotlib.colors.ListedColormap(["#DCDCDC", C_POS]), vmin=0, vmax=1, aspect="auto")
a.set_xticks(range(len(variaveis)))
a.set_xticklabels(variaveis, fontsize=4.3)
a.set_yticks(range(len(fontes)))
a.set_yticklabels(fontes, fontsize=4.6)
a.set_xticks(np.arange(-.5, len(variaveis), 1), minor=True)
a.set_yticks(np.arange(-.5, len(fontes), 1), minor=True)
a.grid(which="minor", color="white", linewidth=0.6)
a.tick_params(which="minor", length=0)
a.set_title("(b) variável disponível por fonte", fontsize=6.2, pad=3)

# ---- (c) o AUC pode medir o criterio, nao o fenomeno --------------------------
a = ax[1, 0]
ap = neg["aplicacao_cruzada"]
proprio = [neg["modelos"]["exclusao"]["auc_cv"], neg["modelos"]["observacao"]["auc_cv"]]
cruzado = [ap["exclusao_em_observacao"]["auc"], ap["observacao_em_exclusao"]["auc"]]
x = np.arange(2)
w = 0.34
a.bar(x - w / 2, proprio, w, color=C_AC, label="no próprio dado")
a.bar(x + w / 2, cruzado, w, color=C_POS, label="no dado do outro")
a.axhline(0.5, color="k", ls="--", lw=0.6)
for i in range(2):
    a.text(i - w / 2, proprio[i] + .012, f"{proprio[i]:.3f}", ha="center", fontsize=4.8)
    a.text(i + w / 2, cruzado[i] + .012, f"{cruzado[i]:.3f}", ha="center", fontsize=4.8)
a.set_xticks(x)
a.set_xticklabels(["neg. por\nexclusão", "neg. por\nobservação"])
a.set_ylim(0.45, 1.30)
a.set_ylabel("AUC")
a.legend(frameon=False, handlelength=0.8, loc="upper center", ncol=2, columnspacing=0.5, borderpad=0.05, fontsize=4.6, handletextpad=0.3)
a.set_title("(c) efeito da definição de negativo", fontsize=6.2, pad=3)

# ---- (d) a regua de HAND muda com a classe de relevo --------------------------
a = ax[1, 1]
cs = serra["caracterizacao"]
vals_pos = [cs["planicie"]["hand_m"]["pos"], cs["serra"]["hand_m"]["pos"]]
vals_neg = [cs["planicie"]["hand_m"]["neg"], cs["serra"]["hand_m"]["neg"]]
x = np.arange(2)
w = 0.34
a.bar(x - w / 2, vals_pos, w, color=C_POS, label="positivo")
a.bar(x + w / 2, vals_neg, w, color=C_NEG, label="negativo")
a.set_yscale("log")
a.set_ylim(0.15, 400)
for i in range(2):
    a.text(i - w / 2, vals_pos[i] * 1.18, f"{vals_pos[i]:.2f}", ha="center", fontsize=4.8)
    a.text(i + w / 2, vals_neg[i] * 1.18, f"{vals_neg[i]:.2f}", ha="center", fontsize=4.8)
a.set_xticks(x)
a.set_xticklabels(["planície\n(97 AOIs)", "serra\n(22 AOIs)"])
a.set_ylabel("HAND (m)")
a.set_title("(d) a régua muda com o relevo", fontsize=6.2, pad=2)

for axx in ax.ravel():
    axx.spines[["top", "right"]].set_visible(False)
    axx.tick_params(width=0.6, length=2)

plt.tight_layout(pad=0.25, w_pad=0.7, h_pad=0.9)
out = os.path.join(os.path.dirname(__file__), "fig2_datasets.pdf")
plt.savefig(out, bbox_inches="tight", pad_inches=0.01)
print("REC", pos[0], neg_[0], "| CTB", pos[1], neg_[1], "| UK", pos[3], neg_[3],
      "| CEMS serra", pos[4], neg_[4], "| CEMS plan", pos[5], neg_[5])
print("(c)", proprio, cruzado, "| (d)", vals_pos, vals_neg)
