#!/usr/bin/env python3
"""Maze generation tool based on Requirement.md.

This module reads track definitions from a CSV file and procedurally
builds a maze layout.  The implementation follows the specification in
`Requirement.md` shipped with the repository.  Parameters that used to be
provided on the command line are stored as module level constants so they
can be tweaked directly in this file.

The actual placement algorithm tries to stay faithful to the rules but it
is intentionally conservative so the script can run without external game
engine dependencies.
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


# Configuration -------------------------------------------------------------

CSV_PATH = Path("rail_config.csv")
MIN_DIFFICULTY = 1
MAX_DIFFICULTY = 3
BUILD_SPACE_SIZE = 9
SAFETY_ZONE_SIZE = 3
CHECKPOINT_COUNT = 1
SEED: Optional[int] = None
OUTPUT_JSON = Path("maze_layout.json")

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

BlockPos = Tuple[int, int, int]
Rotation = int  # degrees around the vertical axis


def _hash_name(name: str) -> int:
    """Return a deterministic 32-bit hash for a track name."""
    return hash(name) & 0xFFFFFFFF


@dataclass
class Exit:
    """Definition for a track exit."""

    direction: str  # e.g. "X+", "Y-"
    relative_pos: BlockPos
    allowed_rotations: List[Rotation]

    def direction_vector(self) -> BlockPos:
        mapping = {
            "X+": (1, 0, 0),
            "X-": (-1, 0, 0),
            "Y+": (0, 1, 0),
            "Y-": (0, -1, 0),
            "Z+": (0, 0, 1),
            "Z-": (0, 0, -1),
        }
        if self.direction not in mapping:
            raise ValueError(f"Unknown direction: {self.direction}")
        return mapping[self.direction]


@dataclass
class Track:
    """Representation of a track module."""

    name: str
    index: int
    size: BlockPos
    difficulty: float
    exits: List[Exit]
    type: str = "normal"  # start, end, checkpoint or normal

    @classmethod
    def from_row(cls, row: Dict[str, str]) -> "Track":
        """Create a :class:`Track` from a CSV row.

        The cleaned ``rail_config.csv`` stores one exit per three columns,
        ``Exit1Dir``, ``Exit1Pos`` and ``Exit1Rot`` (up to three exits).  Pos
        values are comma separated coordinates and ``Rot`` may be either a
        JSON array or a simple integer.  A ``Type`` column classifies special
        rails such as start/end/checkpoint.
        """

        size = (int(row["SizeX"]), int(row["SizeY"]), int(row["SizeZ"]))
        exits: List[Exit] = []
        for i in range(1, 4):
            dir_key = f"Exit{i}Dir"
            dir_val = row.get(dir_key)
            if not dir_val:
                continue

            pos_key = f"Exit{i}Pos"
            pos_str = row.get(pos_key, "0,0,0")
            pos = tuple(int(p.strip()) for p in pos_str.split(","))

            rot_key = f"Exit{i}Rot"
            rot_str = (row.get(rot_key) or "[0]").strip()
            try:
                rot_list = [int(r) for r in json.loads(rot_str)]
            except json.JSONDecodeError:
                rot_list = [int(r.strip()) for r in rot_str.strip("[]").split(",") if r.strip()]

            exits.append(
                Exit(
                    direction=dir_val,
                    relative_pos=pos,
                    allowed_rotations=rot_list,
                )
            )

        track_type = row.get("Type", "normal")
        return cls(
            name=row["Name"],
            index=_hash_name(row["Name"]),
            size=size,
            difficulty=float(row["Difficulty"]),
            exits=exits,
            type=track_type.lower(),
        )


@dataclass
class Placement:
    """Placed track information."""

    track: Track
    position: BlockPos
    rotation: Rotation
    final_difficulty: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.track.name,
            "index": self.track.index,
            "position": self.position,
            "rotation": self.rotation,
            "difficulty": self.final_difficulty,
        }


# ---------------------------------------------------------------------------
# Maze generator implementation
# ---------------------------------------------------------------------------

class MazeGenerator:
    """Generate a maze given track definitions and constraints."""

    BLOCK_UNIT_IN_CM = 16  # 1 logical unit equals 16 centimeters

    def __init__(
        self,
        csv_path: Path,
        difficulty_range: Tuple[int, int],
        build_space_size: int,
        safety_zone_size: int,
        checkpoint_count: int,
        seed: Optional[int] = None,
    ) -> None:
        self.csv_path = Path(csv_path)
        self.min_difficulty, self.max_difficulty = difficulty_range
        self.build_space_size = build_space_size
        self.safety_zone_size = safety_zone_size
        self.checkpoint_count = checkpoint_count
        self.random = random.Random(seed)

        self.tracks: List[Track] = []
        self.by_type: Dict[str, List[Track]] = {
            "start": [],
            "end": [],
            "checkpoint": [],
            "single": [],
            "multi": [],
        }
        self.placements: List[Placement] = []
        self.placement_map: Dict[BlockPos, Placement] = {}

    # ------------------------------------------------------------------
    # Loading and categorising tracks
    # ------------------------------------------------------------------

    def load_tracks(self) -> None:
        """Load the track definitions from the CSV file."""
        with self.csv_path.open("r", encoding="utf8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                track = Track.from_row(row)
                self.tracks.append(track)
                if track.type == "start":
                    self.by_type["start"].append(track)
                elif track.type == "end":
                    self.by_type["end"].append(track)
                elif track.type == "checkpoint":
                    self.by_type["checkpoint"].append(track)
                else:
                    if len(track.exits) > 1:
                        self.by_type["multi"].append(track)
                    else:
                        self.by_type["single"].append(track)

        if not self.by_type["start"]:
            raise RuntimeError("No start tracks defined in CSV")
        if not self.by_type["end"]:
            raise RuntimeError("No end tracks defined in CSV")
        if not self.by_type["single"] and not self.by_type["multi"]:
            raise RuntimeError("No regular tracks defined in CSV")

    # ------------------------------------------------------------------
    # Utility helpers
    # ------------------------------------------------------------------

    @staticmethod
    def world_offset_z(safety_zone_size: int) -> int:
        """Compute world Z offset from safety zone size."""
        return (safety_zone_size - 1) // 2

    @staticmethod
    def to_world_coordinates(pos: BlockPos) -> Tuple[int, int, int]:
        """Convert block units to centimetres for UE5."""
        return tuple(p * MazeGenerator.BLOCK_UNIT_IN_CM for p in pos)

    def is_in_bounds(self, pos: BlockPos) -> bool:
        half = self.build_space_size // 2
        return all(-half <= p < half + 1 for p in pos)

    # ------------------------------------------------------------------
    # Generation algorithm
    # ------------------------------------------------------------------

    def generate(self) -> List[Placement]:
        """Generate a maze layout and return placements."""
        self.load_tracks()
        self.place_start()
        self.iterate_generation()
        return self.placements

    def place_start(self) -> None:
        """Place the starting track at a random valid position."""
        half = self.build_space_size // 2
        start_pos = (
            self.random.randint(-half, half),
            self.random.randint(-half, half),
            self.random.randint(-half, half),
        )
        start_track = self.random.choice(self.by_type["start"])
        placement = Placement(
            track=start_track,
            position=start_pos,
            rotation=0,
            final_difficulty=start_track.difficulty,
        )
        self.record_placement(placement)

    def record_placement(self, placement: Placement) -> None:
        self.placements.append(placement)
        # mark occupied cells according to track size
        x, y, z = placement.position
        sx, sy, sz = placement.track.size
        for dx in range(sx):
            for dy in range(sy):
                for dz in range(sz):
                    cell = (x + dx, y + dy, z + dz)
                    self.placement_map[cell] = placement

    # Difficulty management ------------------------------------------------

    def iterate_generation(self) -> None:
        target_total = self.random.randint(self.min_difficulty, self.max_difficulty)
        segment_target = target_total / (self.checkpoint_count + 1)
        current_total = self.placements[-1].final_difficulty
        current_segment = self.placements[-1].final_difficulty

        open_exits: List[Tuple[Placement, Exit]] = [
            (self.placements[0], ex) for ex in self.placements[0].track.exits
        ]

        last_difficulty = self.placements[0].final_difficulty
        checkpoints_placed = 0

        while open_exits:
            base = self.random.choice(open_exits)
            base_placement, base_exit = base
            next_pos = tuple(
                base_placement.position[i] + base_exit.direction_vector()[i]
                for i in range(3)
            )

            # decide track type depending on difficulty
            force_checkpoint = (
                checkpoints_placed < self.checkpoint_count
                and current_segment > segment_target
            )
            place_checkpoint = force_checkpoint or (
                checkpoints_placed < self.checkpoint_count
                and current_segment > segment_target * 0.5
                and self.random.random() < 0.5
            )

            force_end = current_total > target_total
            place_end = force_end or (
                current_total > target_total * 0.75 and self.random.random() < 0.75
            )

            if place_end:
                track_list = self.by_type["end"]
            elif place_checkpoint:
                track_list = self.by_type["checkpoint"] or self.by_type["multi"]
            else:
                # prefer multi exit tracks to allow expansion
                track_list = self.by_type["multi"] + self.by_type["single"]

            success = False
            for _ in range(20):  # limit number of attempts per step
                track = self.random.choice(track_list)
                rotation = self.random.choice(track.exits[0].allowed_rotations)
                if self.can_place(track, next_pos):
                    final_diff = track.difficulty * (1 + last_difficulty * 0.1)
                    placement = Placement(track, next_pos, rotation, final_diff)
                    self.record_placement(placement)
                    last_difficulty = final_diff
                    current_total += final_diff
                    current_segment += final_diff
                    open_exits.extend((placement, ex) for ex in track.exits)
                    open_exits.remove(base)
                    success = True
                    if place_checkpoint and track in self.by_type["checkpoint"]:
                        checkpoints_placed += 1
                        current_segment = 0.0
                    if place_end and track in self.by_type["end"]:
                        return
                    break
            if not success:
                # As a last resort try to close the path with an end piece
                if not any(pl.track.type == "end" for pl in self.placements):
                    end_track = self.random.choice(self.by_type["end"])
                    if self.can_place(end_track, next_pos):
                        final_diff = end_track.difficulty * (1 + last_difficulty * 0.1)
                        placement = Placement(end_track, next_pos, 0, final_diff)
                        self.record_placement(placement)
                        return

                # backtrack: remove base exit to avoid infinite loop
                open_exits.remove(base)

    def can_place(self, track: Track, pos: BlockPos) -> bool:
        x, y, z = pos
        sx, sy, sz = track.size
        for dx in range(sx):
            for dy in range(sy):
                for dz in range(sz):
                    cell = (x + dx, y + dy, z + dz)
                    if not self.is_in_bounds(cell) or cell in self.placement_map:
                        return False
        return True

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def export_json(self, path: Path) -> None:
        """Export placements to a JSON file with world and maze coordinates."""
        data = []
        offset_z = self.world_offset_z(self.safety_zone_size)
        for pl in self.placements:
            wx, wy, wz = self.to_world_coordinates(pl.position)
            data.append(
                {
                    "name": pl.track.name,
                    "index": pl.track.index,
                    "position_maze": pl.position,
                    "position_cm": (wx, wy, wz + offset_z * self.BLOCK_UNIT_IN_CM),
                    "rotation": pl.rotation,
                    "difficulty": pl.final_difficulty,
                }
            )
        path.write_text(json.dumps(data, indent=2), encoding="utf8")


def validate(gen: "MazeGenerator") -> None:
    """Run a series of sanity checks on ``gen.placements``.

    This mirrors the lightweight validation that previously lived in
    ``tester.py`` so that maze creation and verification happen in one
    place.
    """

    total_diff = sum(pl.final_difficulty for pl in gen.placements)
    start_count = sum(1 for pl in gen.placements if pl.track.type == "start")
    end_count = sum(1 for pl in gen.placements if pl.track.type == "end")
    checkpoint_count = sum(
        1 for pl in gen.placements if pl.track.type == "checkpoint"
    )

    assert start_count == 1, f"expected 1 start, got {start_count}"
    assert end_count <= 1, f"expected at most 1 end, got {end_count}"
    assert checkpoint_count <= gen.checkpoint_count, (
        f"expected at most {gen.checkpoint_count} checkpoints, got {checkpoint_count}"
    )
    assert gen.min_difficulty <= total_diff <= gen.max_difficulty, (
        f"total difficulty {total_diff} outside range {gen.min_difficulty}-{gen.max_difficulty}"
    )

    for pl in gen.placements:
        x, y, z = pl.position
        sx, sy, sz = pl.track.size
        for dx in range(sx):
            for dy in range(sy):
                for dz in range(sz):
                    cell = (x + dx, y + dy, z + dz)
                    assert gen.is_in_bounds(cell), f"{pl.track.name} out of bounds"


# ---------------------------------------------------------------------------
# Script entry point
# ---------------------------------------------------------------------------


def main() -> int:
    gen = MazeGenerator(
        csv_path=CSV_PATH,
        difficulty_range=(MIN_DIFFICULTY, MAX_DIFFICULTY),
        build_space_size=BUILD_SPACE_SIZE,
        safety_zone_size=SAFETY_ZONE_SIZE,
        checkpoint_count=CHECKPOINT_COUNT,
        seed=SEED,
    )
    gen.generate()
    validate(gen)
    gen.export_json(OUTPUT_JSON)
    print(f"Generated {len(gen.placements)} placements -> {OUTPUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
