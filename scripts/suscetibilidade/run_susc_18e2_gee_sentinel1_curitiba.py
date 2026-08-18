from __future__ import annotations

import susc_18e2_execucao_gee_common as s18e2


if __name__ == "__main__":
    metadata = s18e2.run_all_routes()
    print(
        "18E2 execucao controlada Sentinel-1 Curitiba: "
        f"autenticado={str(metadata.get('authenticated')).lower()} "
        f"gee_consultado={str(metadata.get('gee_queried')).lower()} "
        f"pre={metadata.get('pre_scene_count')} pos={metadata.get('post_scene_count')} "
        f"export={str(metadata.get('export_started')).lower()} "
        f"status18E2={metadata.get('status_18e2')}"
    )
    raise SystemExit(0)
