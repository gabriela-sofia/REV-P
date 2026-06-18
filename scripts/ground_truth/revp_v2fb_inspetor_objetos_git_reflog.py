from pathlib import Path
from revp_v2ez_to_v2ff_comum import parse_args, run_v2fb
args = parse_args()
run_v2fb(Path(args.repo_root), args.force)

