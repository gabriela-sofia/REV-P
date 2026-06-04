#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v1uw_curitiba_common import parse_args, run_document_text_extractor
except ModuleNotFoundError:
    from revp_v1uw_curitiba_common import parse_args, run_document_text_extractor


if __name__ == "__main__":
    run_document_text_extractor(parse_args())
