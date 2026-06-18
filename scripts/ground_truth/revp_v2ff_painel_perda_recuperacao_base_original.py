from pathlib import Path
from revp_v2ez_to_v2ff_comum import parse_args, run_v2ff
args = parse_args()
run_v2ff(Path(args.repo_root), args.force)

