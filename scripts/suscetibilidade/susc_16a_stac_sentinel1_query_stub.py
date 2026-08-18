"""SUSC-16A Sentinel-1 STAC query stub (review-only).

This stub prints the query shape only. It does not authenticate, download, or
write raster data unless a future operator explicitly implements those steps.
"""

from __future__ import annotations

import json


def build_query(aoi_bbox, pre_window, post_window):
    return {
        "collections": ["sentinel-1-grd"],
        "bbox": aoi_bbox,
        "datetime": f"{pre_window[0]}/{post_window[1]}",
        "query": {"sar:instrument_mode": {"eq": "IW"}},
        "review_only": True,
    }


if __name__ == "__main__":
    print(json.dumps(build_query([-35.05, -8.20, -34.80, -7.90], ("YYYY-MM-DD", "YYYY-MM-DD"), ("YYYY-MM-DD", "YYYY-MM-DD")), indent=2))
