# SUSC — Roadmap após SUSC-03

> Todos os marcos abaixo preservam a política review-only do REV-P. Nenhum deles, por si só, cria ground truth de ocorrência ou desbloqueia treinamento supervisionado. A matriz de suscetibilidade ≠ ocorrência confirmada.

Estado atual: **SUSC-03 concluído** — matriz `susc_features_by_patch_v1.csv` migrada (300×72), validada e perfilada dentro do REV-P.

## Ordem recomendada

### SUSC-04 — Auditoria de proveniência das features incertas
Enumerar de forma definitiva as features com origem incerta (consolidando os flags do manifesto: `requires_provenance_audit`, `computation_script_found=false`, `public_source_known=false`, `raw_source_available=false`). Para cada uma: localizar/documentar fonte pública, script de computação e janela temporal. Resolver a estimativa "~11" do SUSC-02 com a contagem real. Sem alterar dados.

### SUSC-05 — Formalização científica das direções esperadas das features
Consolidar, por feature, a `expected_direction_for_flood_susceptibility` com citação metodológica (SPGAM, Baixo Jaguaribe, literatura). Produz uma tabela de hipóteses direcionais auditável — hipótese, não resultado.

### SUSC-06 — Baseline SPGAM/GAM interpretável por região
Ajustar um modelo aditivo generalizado **interpretável** (não supervisionado contra ocorrência) usando os condicionantes físicos por região, como referência metodológica comparável ao SPGAM. Sem usar labels heurísticos como verdade.

### SUSC-07 — Validação evento-real/documental por overlay
Sobrepor evidência documental/hidrológica (Defesa Civil, CPRM/SGB, séries de cota, charters) aos patches para checar consistência da suscetibilidade. É o caminho para qualquer noção futura de referência — ainda não ground truth de treino.

### SUSC-08 — Expansão DINO embeddings para todos os patches elegíveis
Estender a extração DINOv2 (com registers) aos patches elegíveis, como representação latente complementar por patch. Embeddings ficam fora do Git (pesados); apenas manifests/stats versionados.

### SUSC-09 — Score multimodal v6
Combinar condicionantes físicos, evidência óptica/SAR e embeddings DINO em um score multimodal v6, comparado ao v5 e ao baseline SPGAM. Score, não rótulo verdadeiro.

### SUSC-10 — Comparação SPGAM vs score v6 vs DINO features
Análise comparativa final (concordância, divergência, sensibilidade) entre o baseline SPGAM, o score v6 e as features DINO, com limitações explícitas e sem overclaim preditivo.

---

> Disclaimer obrigatório: A matriz SUSC-03 é um artefato tabular review-only de atributos associados à suscetibilidade urbana a enchentes. Ela não constitui ground truth de ocorrência, não desbloqueia treinamento supervisionado e não autoriza afirmações de evento observado por patch.
