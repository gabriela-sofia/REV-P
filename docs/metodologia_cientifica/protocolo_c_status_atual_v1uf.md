# Protocolo C — Status Atual após v1uf

**Data:** 2026-06-03
**Modo:** review_only / fail-closed

## Onde Estamos

Após a v1uf, o Protocolo C passou a contar com **evidência hidrometeorológica
oficial real** para alguns eventos, extraída de séries do INMET por estação e janela
temporal, com coordenadas resolvidas do catálogo oficial.

| Evento | Série oficial | Coord oficial | Precip no evento | Estado |
|--------|---------------|---------------|------------------|--------|
| PET_2024_03_21_28 | sim (A610) | sim | 265,8 mm | âncora temporal confirmada, mas bloqueado |
| PET_2022_02_15 | sim (A610) | sim | 2,8 mm (baixa) | bloqueado (misto + sem geometria) |
| REC_2022_05_24_30 | sim (A301) | sim | cobertura insuficiente | bloqueado (sem cobertura/geometria) |

## O Que Mudou

- **Agora existe evidência hidrometeorológica oficial** para PET_2022 e PET_2024
  (e tentativa documentada para REC_2022).
- Isso **melhora a ancoragem temporal** dos eventos.
- As coordenadas das estações vêm do **catálogo oficial INMET** (com hash), nunca
  inventadas.

## O Que NÃO Mudou (continua bloqueado)

- **Ainda não existe ground truth operacional.**
- **Ainda não existe ground reference.**
- **Ainda não existe label.**
- **Nenhum overlay foi executado.**
- **Nenhuma geometria de inundação foi obtida.**
- Eventos PET continuam bloqueados por **fenômeno misto** (inundação + deslizamento
  não separados).
- REC continua bloqueado por **cobertura insuficiente** e ausência de geometria.

## Princípios Reafirmados

- Estação oficial **não é** geometria de inundação.
- Precipitação forte **não é** label.
- Ausência de precipitação em estação **não nega** o evento.
- Evidência hidrometeorológica serve como **plausibilidade temporal**, não como verdade
  espacial patch-level.
- Ground reference exige **geometria observada + overlay + revisão supervisora**.
- Protocolo B permanece bloqueado.

## Próxima Etapa

A próxima etapa (`v1ug`) deve focar em **pacote de revisão humana + solicitações
formais para geometria observada**: pontos/polígonos/ocorrências georreferenciadas
junto a SGB/CPRM, DRM-RJ, Defesa Civil de Petrópolis, COMPDEC Recife e Cemaden, além
de preparar a adjudicação supervisora. Sem label até haver ground reference validado.
