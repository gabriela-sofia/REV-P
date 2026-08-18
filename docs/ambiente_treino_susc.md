# Ambiente de treino da linha causal SUSC

**Status**: guia operacional, não é gate nem decisão científica. Documenta um
problema real de ambiente encontrado e resolvido em 04-05/08/2026, e a
estrutura escolhida pra evitar que ele se repita a cada script novo.

---

## O problema real, na ordem em que apareceu

A linha causal SUSC (Firth penalizado sobre features físico-hidrológicas, e
agora um candidato interpretável mais forte, EBM/`ExplainableBoostingClassifier`)
depende de duas bibliotecas — `firthlogist` e `interpret` — que têm restrições
de ambiente reais, não hipotéticas:

1. **Nenhuma das duas publica versão compatível com Python ≥3.11.** Testado em
   05/08/2026 no ambiente `(base)` real da autora (Python 3.13.13, via
   miniconda): `pip install interpret firthlogist` falhou — `interpret` listou
   todas as versões existentes exigindo `Python <3.11`; `firthlogist` não
   encontrou nenhuma versão instalável. Não é erro de configuração, é o teto
   real dessas duas libs no momento em que este documento foi escrito.
2. **`firthlogist` chama uma API privada do scikit-learn removida na 1.6**
   (`BaseEstimator._validate_data`). Isso já tinha sido encontrado e resolvido
   no relatório do SUSC-20M (23/07/2026), com um shim de 3 linhas escrito
   direto no script daquela rodada. Reproduzido de novo, de forma
   independente, em 05/08/2026 (Python 3.10.12, scikit-learn 1.7.2) — ou seja,
   é uma incompatibilidade estável entre as duas bibliotecas, não um acidente
   de uma máquina específica.

Nenhum dos dois problemas é sobre capacidade de hardware. Pra registrar isso
com números reais: no mesmo dia, um treino de Firth rodou em **0,056s** e um
treino de EBM em **25,3s**, os dois no tamanho real do dataset primário de
Curitiba (n=1458, 5 features), num sandbox de teste com só 2 núcleos e 3,8GB
de RAM. Na máquina real da autora (10 núcleos, 15,5GB RAM, 560GB livres em
disco), a demanda de processamento é irrelevante — o teto é de versão de
biblioteca, não de máquina.

---

## A decisão: ambiente dedicado, separado do `(base)` e da linha DINO

Duas linhas de dependência coexistem no repositório, e não devem se misturar:

| Linha | Status | Onde | Python |
|---|---|---|---|
| DINOv2/embeddings | Histórica, encerrada como candidata a feature (PLANO_ACAO, decisão 2026-08-01/02) | `requirements.txt` (raiz) | sem restrição conhecida |
| Causal SUSC (Firth + EBM) | **Ativa** | `environment.yml` (conda) ou `requirements-susc.txt` (pip) | **obrigatório <3.11** |

Instalar as duas linhas juntas não tem benefício e aumenta a chance de
conflito de versão (foi exatamente misturar tudo no `(base)` que expôs o
problema #1 acima). Trate como dois ambientes de verdade, não como um
`requirements.txt` só.

---

## Como criar o ambiente

**Com conda (recomendado, já é o que a autora usa)**:

```bash
cd REV-P
conda env create -f environment.yml
conda activate revp-susc
```

**Com venv/pip puro** (precisa de um Python 3.9 ou 3.10 já instalado e
acessível como `python`):

```bash
cd REV-P
python -m venv .venv-susc
source .venv-susc/bin/activate   # Linux/Mac
# .\.venv-susc\Scripts\Activate.ps1  # Windows PowerShell
pip install -r requirements-susc.txt
```

## Como verificar que funciona de verdade (não só que instalou)

```bash
python -m pytest tests/test_susc_ambiente_treino_setup.py -v
```

Cinco testes reais: versão de Python compatível, o shim do `firthlogist`
aplica e é idempotente, um Firth treina de verdade com o shim, um EBM treina
no tamanho real do dataset de Curitiba (dado sintético, nunca dado real do
projeto), e um diagnóstico geral não levanta exceção.

Diagnóstico rápido isolado, fora do pytest:

```bash
python scripts/suscetibilidade/susc_ambiente_compat_common.py
```

Imprime um JSON com versão de Python, versão do scikit-learn, se o shim
precisou ser aplicado, e se as duas libs importam.

---

## O que está em cada pacote, e por quê

| Pacote | Papel |
|---|---|
| `numpy`, `pandas`, `scipy` | manipulação de dado e testes estatísticos (Mann-Whitney já usa `scipy.stats`) |
| `scikit-learn` | métricas (AUC), validação cruzada, base do `interpret` |
| `firthlogist` | regressão logística com penalização de Firth — a rota causal primária |
| `interpret` | `ExplainableBoostingClassifier` (EBM) — candidato interpretável em avaliação, ainda não promovido a rota primária |
| `matplotlib` | plots dos relatórios |
| `pytest`, `pyyaml`, `tqdm` | já usados no resto do repositório |

Nota sobre dependências transitivas: instalar `interpret` traz `shap` junto
(dependência dele), mesmo sem ninguém pedir `shap` diretamente. Isso não é um
problema em si, mas é bom saber — `shap` já falhou por timeout de rede num
sandbox anterior (SUSC-20V), então se `pip install interpret` falhar num
ambiente novo, vale checar se o erro é especificamente do `shap`.

## O que não fazer aqui

Não instalar `xgboost`, `shap` isolado, ou qualquer outra lib nova neste
ambiente sem uma tarefa concreta que precise dela — é a mesma regra de "uma
tarefa por vez" que rege o resto do projeto. Este ambiente existe pra rodar
Firth (rota primária) e prototipar EBM (candidato em avaliação), nada além
disso por enquanto.
