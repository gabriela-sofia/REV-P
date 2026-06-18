from pathlib import Path
from revp_v2ez_to_v2ff_comum import parse_args, run_integrated
args = parse_args()
run_integrated(Path(args.repo_root), args.force)

