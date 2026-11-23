#!/usr/bin/env python3
"""
Join COGMark's cogzilla results with the GDAL / TiTiler baseline and print a
Markdown comparison table.

    python benches/compare_results.py [results_dir]

Reads `cogzilla_results.json` (written by `cargo run -p cogmark`) and
`gdal_results.json` (written by `gdal_oracle.py`). Standard library only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def fmt_us(us: float) -> str:
    if us >= 1000:
        return f"{us / 1000:.2f} ms"
    return f"{us:.1f} µs"


def main() -> int:
    results = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "results"
    cogzilla = {r["id"]: r for r in json.loads((results / "cogzilla_results.json").read_text())}
    oracle = json.loads((results / "gdal_results.json").read_text())
    baseline = {r["id"]: r for r in oracle["gdal"]}

    print("| ID | Workload | cogzilla | GDAL / TiTiler | Speedup |")
    print("| :--- | :--- | ---: | ---: | ---: |")
    for bid, c in cogzilla.items():
        b = baseline.get(bid)
        name = c.get("name", "")
        c_qps = c.get("extra", {}).get("qps")
        if b is None:
            print(f"| `{bid}` | {name} | {fmt_us(c['median_us'])} | — | — |")
        elif c_qps and b.get("extra", {}).get("qps"):
            b_qps = b["extra"]["qps"]
            print(f"| `{bid}` | {name} | {c_qps:,.0f} tiles/s | {b_qps:,.0f} tiles/s | {c_qps / b_qps:.1f}× |")
        else:
            speedup = b["median_us"] / c["median_us"]
            print(f"| `{bid}` | {name} | {fmt_us(c['median_us'])} | {fmt_us(b['median_us'])} | {speedup:.1f}× |")

    parity = oracle.get("parity") or {}
    if parity:
        print("\n**Pixel parity vs. GDAL**\n")
        print("```json")
        print(json.dumps(parity, indent=2))
        print("```")
    return 0


if __name__ == "__main__":
    sys.exit(main())
