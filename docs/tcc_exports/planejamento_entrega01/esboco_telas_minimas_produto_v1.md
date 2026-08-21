# Esboço de telas mínimas do produto — v1 (2026-08-16)

**Por que isto é um arquivo separado, e não uma figura no `main.tex`**: o
produto não é uma interface gráfica, é um contrato de API (Fig. 1 do plano,
coluna 4). "Tela" aqui não é layout de tela — é o *payload* que um cliente
recebe, que é o que de fato existe hoje (SUSC-20E/20F, `PLANO_ACAO_produto_v1.md`
Fase 5). A pergunta "vale criar imagem de protótipo de interface?" já tinha
ficado em aberto na v6 (`NOTA_versoes.md`); isto resolve a pergunta com a
resposta mínima que o estado real do produto permite, sem inventar uma UI que
não existe e sem gastar do orçamento de 3 páginas do documento de planejamento
com uma figura nova.

Três estados possíveis, cada um já implementado no contrato descrito em §II
("Serviço e camada de explicação") — este arquivo só torna visual o que o
texto já declara:

> **Nota de 20/08/2026 — os tres estados deixaram de ser esboco.** O contrato
> roda em `scripts/servico/svc02_contrato_inferencia.py`, e os tres estados aqui
> desenhados sao respostas reais gravadas em
> `local_runs/svc-02-contrato/respostas_demonstracao.json`. Duas diferencas em
> relacao ao esboco: os portoes sao cinco e nomeados na resposta, e Recife
> responde `ok` com maturidade `mvp_local` carregando o criterio de leitura que
> seu modelo nao atinge. Ver `ext_servico_contrato_inferencia_v1.md`.

## Estado 1 — `ok`

```
ENTRADA           geometria (AOI), CRS, período
                   ↓
SAÍDA   status: ok
        região: Recife | maturidade: mvp_local
        escore: 0.62   IC: [0.51, 0.73]
        variáveis usadas: hand_m, twi_dinf, rain_max_24h, rain_decay_index
        model_card: região, maturidade, limites de uso
        explicação: "HAND baixo (2.1 m) e TWI alto pesam a favor;
                      chuva recente pesa contra"
```

## Estado 2 — `insufficient_data`

```
ENTRADA           geometria fora da cobertura real, ou variável faltando
                   ↓
SAÍDA   status: insufficient_data
        gate que não fechou: <nome do gate> (ex.: DEM fora de cobertura)
        nenhum escore devolvido
```

## Estado 3 — `region_not_supported`

```
ENTRADA           geometria em Petrópolis (ou qualquer região sem modelo)
                   ↓
SAÍDA   status: region_not_supported
        motivo: sem modelo ajustado e validado para esta região
        nenhum escore por analogia
```

## O que isto não é

Não é mockup de tela para usuário final, não é decisão de design de produto
(cores, fluxo de navegação) e não substitui o contrato de API real
(`susc_20e_api_contrato_inferencia_recife/`). É só a tradução mínima, em três
estados, do que "serviço auditável" já significa no texto do plano — para uso
em conversa com o orientador ou em pôster, se for útil.
