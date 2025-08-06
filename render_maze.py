#!/usr/bin/env python3
"""Render a simple ASCII visualization of a generated maze from JSON data."""

from __future__ import annotations

import json
from pathlib import Path

# Path to the JSON file produced by ``generator.py``
JSON_PATH = Path("maze_layout.json")

SYMBOLS = {
    "start": "S",
    "end": "E",
    "checkpoint": "C",
    "normal": "#",
}


def symbol_for(name: str) -> str:
    lname = name.lower()
    if "start" in lname:
        return SYMBOLS["start"]
    if "end" in lname:
        return SYMBOLS["end"]
    if "checkpoint" in lname:
        return SYMBOLS["checkpoint"]
    return SYMBOLS["normal"]


def main() -> int:
    data = json.loads(JSON_PATH.read_text())
    pos_map: dict[tuple[int, int], str] = {}
    for item in data:
        x, y, _ = item["position_maze"]
        pos_map[(x, y)] = symbol_for(item["name"])

    if not pos_map:
        print("No placements to render.")
        return 0

    xs = [p[0] for p in pos_map]
    ys = [p[1] for p in pos_map]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    for y in range(max_y, min_y - 1, -1):
        row = []
        for x in range(min_x, max_x + 1):
            row.append(pos_map.get((x, y), "."))
        print("".join(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
