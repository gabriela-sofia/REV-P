from pathlib import Path
from revp_v2ez_to_v2ff_comum import parse_args, run_v2fd
args = parse_args()
run_v2fd(Path(args.repo_root), args.force)

