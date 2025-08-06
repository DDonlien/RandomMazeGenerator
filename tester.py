#!/usr/bin/env python3
"""Simple validator for the maze generator.

The script executes ``generator.py``'s :class:`MazeGenerator` with the
provided parameters and performs a series of sanity checks on the produced
placements:

* Exactly one start and one end track exist.
* The requested number of checkpoints is present.
* Total difficulty is within the supplied range.
* Every track fits within the build space bounds.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import generator


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate generated maze")
    parser.add_argument("csv", type=Path, help="Path to rail configuration CSV")
    parser.add_argument("min_difficulty", type=int, help="Minimum total difficulty")
    parser.add_argument("max_difficulty", type=int, help="Maximum total difficulty")
    parser.add_argument("build_size", type=int, help="Build space size (odd number)")
    parser.add_argument("safety_size", type=int, help="Safety zone size (odd number)")
    parser.add_argument("checkpoints", type=int, help="Number of checkpoints")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    gen = generator.MazeGenerator(
        csv_path=args.csv,
        difficulty_range=(args.min_difficulty, args.max_difficulty),
        build_space_size=args.build_size,
        safety_zone_size=args.safety_size,
        checkpoint_count=args.checkpoints,
        seed=args.seed,
    )
    gen.generate()

    total_diff = sum(pl.final_difficulty for pl in gen.placements)
    start_count = sum(1 for pl in gen.placements if pl.track.type == "start")
    end_count = sum(1 for pl in gen.placements if pl.track.type == "end")
    checkpoint_count = sum(1 for pl in gen.placements if pl.track.type == "checkpoint")

    assert start_count == 1, f"expected 1 start, got {start_count}"
    assert end_count <= 1, f"expected at most 1 end, got {end_count}"
    assert checkpoint_count <= args.checkpoints, (
        f"expected at most {args.checkpoints} checkpoints, got {checkpoint_count}"
    )
    assert args.min_difficulty <= total_diff <= args.max_difficulty, (
        f"total difficulty {total_diff} outside range {args.min_difficulty}-{args.max_difficulty}"
    )

    for pl in gen.placements:
        x, y, z = pl.position
        sx, sy, sz = pl.track.size
        for dx in range(sx):
            for dy in range(sy):
                for dz in range(sz):
                    cell = (x + dx, y + dy, z + dz)
                    assert gen.is_in_bounds(cell), f"{pl.track.name} out of bounds"

    print("All checks passed. Generated", len(gen.placements), "tracks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
