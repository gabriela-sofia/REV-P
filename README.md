# REV-P

O REV-P organiza uma cadeia auditável de evidências para análise físico-ambiental urbana em três regiões brasileiras: Recife, Petrópolis e Curitiba. O projeto usa imagens Sentinel-2, fontes territoriais externas e representações visuais auto-supervisionadas (DINOv2 congelado) para estruturar corpus, inspeções e relatórios destinados à revisão humana. Não há classificador supervisionado, rótulo de treinamento ou predição operacional nesta entrega.

---

## O que o projeto faz

O REV-P constrói e audita uma base de evidências estruturais e territoriais:

- organiza **59 patches territoriais/contextuais** distribuídos entre as três regiões (Recife 18, Petrópolis 27, Curitiba 14);
- mantém um manifesto de **128 assets Sentinel candidatos** como inventário de entrada do pipeline visual;
- extrai embeddings com encoder **DINOv2 congelado** — 12 embeddings reais (4 por região, 768 dimensões) — e aplica análise estrutural: similaridade, k-NN, PCA, medoids, outliers;
- aplica o **Protocolo C** para organizar evidências externas candidatas, distinguindo: evidência contextual, referência temporal, referência candidata e ground truth operacional (este último ainda não estabelecido em nenhuma região);
- produz **figuras, tabelas, métricas e relatórios auditáveis** para apoio à revisão e à entrega acadêmica;
- registra em manifests e registries os estados metodológicos, bloqueios, lacunas e decisões, mantendo rastreabilidade completa.

---

## O que o projeto ainda não faz

O REV-P opera em modo **review-only**. Nesta versão:

- não declara ground truth operacional em nível de patch em nenhuma região;
- não cria rótulos binários de treinamento;
- não treina classificador supervisionado;
- não usa DINOv2 como detector ou preditor de inundação;
- não transforma evidência contextual em validação automática de evento observado.

Essas restrições são escolhas metodológicas, não falhas técnicas. A separação entre evidência contextual, referência candidata e rótulo operacional é parte central do design do projeto.

---

## Estrutura do repositório

```text
REV-P/
├── configs/          # Templates e checklists de execução local
├── datasets/         # Registries CSV/JSON, schemas e tabelas auxiliares
├── docs/             # Documentação técnica e metodológica
├── manifests/        # Manifests auditáveis de corpus, preflight e validação
├── outputs_public/   # Artefatos públicos finais
│   ├── figures/      # Figuras finais para artigo e apresentação
│   ├── tables/       # Tabelas consolidadas de corpus, evidência e análise
│   ├── metrics/      # Métricas descritivas e de QA
│   ├── logs_summary/ # Logs resumidos de execução e testes
│   ├── execution_reports/  # Relatórios de execução, auditoria e rastreabilidade
│   └── model/        # Declaração sobre ausência de modelo operacional
├── scripts/          # Scripts de extração, análise, QA e orquestração
├── tests/            # Testes automatizados do pipeline
└── requirements.txt  # Dependências Python
```

---

## Cadeia metodológica

O pipeline segue esta sequência:

1. **Corpus territorial** — definição dos 59 patches urbanos por região com documentação de linhagem;
2. **Pipeline Sentinel-first** — inventário dos 128 assets candidatos, preflight e controle de qualidade de entrada;
3. **Embeddings DINOv2** — extração com encoder congelado, análise de similaridade, vizinhança, PCA e detecção de outliers;
4. **Protocolo C** — organização de evidências externas candidatas por região (fontes oficiais, meteorológicas, cartográficas), separação de tipos de referência, adjudicação de gates;
5. **Auditoria e entrega** — geração de figuras, tabelas, métricas, relatórios e manifests para submissão e revisão.

Cada etapa tem testes automatizados, registries auditáveis e documentação metodológica em `docs/metodologia_cientifica/`.

---

## Dados e artefatos locais

Arquivos pesados não são versionados no GitHub:

- GeoTIFFs Sentinel originais (10–200 MB cada);
- shapefiles e GeoJSONs brutos;
- embeddings DINOv2 (`.npz`);
- dados de elevação (PE3D/MDE);
- logs completos e caches locais.

O repositório público contém manifests, registries, hashes, tabelas resumidas, figuras derivadas e relatórios leves — o suficiente para verificar a cadeia metodológica sem reproduzir os dados brutos.

---

## Estado atual

| Item | Estado |
|---|---|
| Corpus territorial | 59 patches (Recife 18, Petrópolis 27, Curitiba 14) |
| Assets Sentinel candidatos | 128 |
| Embeddings DINOv2 reais | 12 (4/região, 768D, encoder congelado) |
| Protocolo C — Recife | Referência candidata validada (pontuação 0.76) |
| Protocolo C — Curitiba | Referência temporal validada (pontuação 0.70) |
| Protocolo C — Petrópolis | Referência contextual validada (pontuação 0.55) |
| Ground truth operacional | Não declarado em nenhuma região |
| Treinamento supervisionado | Bloqueado por restrições metodológicas |
| Modelo operacional entregue | Não — ver `outputs_public/model/` |

---

## Como reproduzir os relatórios públicos

Preparação do ambiente:

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/Mac
# .\.venv\Scripts\Activate.ps1  # Windows PowerShell
pip install -r requirements.txt
```

Execução dos testes:

```bash
python -m pytest tests -q
```

Validação dos artefatos públicos:

```bash
python scripts/repository/build_outputs_public_delivery.py --validate-only
```

Geração completa dos artefatos públicos:

```bash
python scripts/repository/build_outputs_public_delivery.py
python scripts/repository/build_outputs_public_delivery.py --finalize
```

O índice principal dos artefatos está em:
`outputs_public/execution_reports/final_delivery_artifact_index.md`

---

## Limitações

- Sem ground truth operacional patch-level: a ausência não é contornada por proxy nem por ausência de evidência negativa.
- Corpus de embeddings pequeno: 12 vetores reais são suficientes para análise estrutural exploratória, não para validação estatística de desempenho.
- Geometria de evento observado ausente em Curitiba e Petrópolis: a sobreposição patch-evento não foi executada por falta de geometria oficial.
- Separação de fenômeno pendente em Petrópolis 2022: inundação e deslizamento coexistem nas fontes; sem separação, a geocodificação controlada permanece bloqueada.
- Fontes externas oficiais não respondidas: 12 solicitações formais a instituições (COMPDEC, DRM-RJ, Defesa Civil, CPRM) estão pendentes de resposta.

---

## Próximos passos

- Obter geometria oficial de evento em Recife (COMPDEC) e Petrópolis (DRM-RJ, PKG_FR_PET_001).
- Resolver separação de fenômeno em Petrópolis 2022 com produto oficial.
- Ampliar o corpus de embeddings DINOv2 além dos 12 vetores atuais.
- Executar sobreposição patch-evento assim que geometria oficial estiver disponível.
- Definir protocolo de label supervisionado após ground truth estabelecido em pelo menos uma região.
