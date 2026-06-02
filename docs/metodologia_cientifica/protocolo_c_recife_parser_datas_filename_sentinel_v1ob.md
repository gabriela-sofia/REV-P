# Protocolo C - Recife parser de datas em filenames Sentinel v1ob

O parser reconhece YYYYMMDD, YYYY-MM-DD, YYYY_MM_DD, timestamps Sentinel e nomes compactos Sentinel-2/Sentinel-1.

Datas extraidas de janelas de evento sao marcadas como EVENT_WINDOW_NOT_SCENE_DATE e nao podem ser aceitas como scene_date sem outra fonte.

Conflitos sao registrados separadamente e bloqueiam uso direto.
