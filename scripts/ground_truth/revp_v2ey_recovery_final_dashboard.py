from pathlib import Path
from revp_v2es_to_v2ey_common import parse_args, run_v2ey
args = parse_args()
run_v2ey(Path(args.repo_root), args.force)
