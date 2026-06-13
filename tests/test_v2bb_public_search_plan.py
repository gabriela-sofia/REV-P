"""v2bb public search plan tests."""

from pathlib import Path
import scripts.v2bb_public_geometry_retrieval_feed_builder as engine

ROOT = Path(__file__).resolve().parents[1]


def test_search_plan_has_direct_urls_and_all_diagnosed_targets():
    rows = engine.load_csv(ROOT / "datasets" / "v2bb_public_search_plan.csv")
    assert len(rows) >= 60
    direct = [row for row in rows if row["must_attempt_download"] == "true"]
    assert len(direct) == 4 and all(row["result_url"].startswith("https://") for row in direct)
    diagnosed = {row["target_id"] for row in engine.load_csv(ROOT / "datasets" / "v2ay_geometry_acquisition_targets.csv")}
    assert diagnosed <= {row["target_id"] for row in rows}
    assert all("license" not in key.lower() for row in rows for key in row)
