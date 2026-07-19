from __future__ import annotations

import sys

import susc_17g_extracao_fisica_common as s17g


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "extract":
        raise SystemExit(s17g.run_extraction_cli())
    raise SystemExit(s17g.run_all())
