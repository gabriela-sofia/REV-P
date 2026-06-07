#!/usr/bin/env python3
try:
    from scripts.protocolo_c.revp_v2an_common import parse_args, run_document_metadata_extractor
except ModuleNotFoundError:
    from revp_v2an_common import parse_args, run_document_metadata_extractor


if __name__ == "__main__":
    run_document_metadata_extractor(parse_args())
