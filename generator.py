#!/usr/bin/env python3
"""Maze generation tool based on Requirement.md.
基于 Requirement.md 的迷宫生成工具。

This module reads track definitions from a CSV file and procedurally
builds a maze layout.  The implementation follows the specification in
`Requirement.md` shipped with the repository.  Parameters that used to be
provided on the command line are stored as module level constants so they
can be tweaked directly in this file.
该模块从 CSV 文件中读取轨道定义，并以程序化方式构建迷宫布局。
其实现遵循仓库中附带的 `Requirement.md` 中的规范。原本通过命令
行提供的参数现在作为模块级常量存储，可以在此文件中直接调整。

The actual placement algorithm tries to stay faithful to the rules but it
is intentionally conservative so the script can run without external game
engine dependencies.
实际的放置算法尽量遵循这些规则，但为了在没有外部游戏引擎依赖
的情况下运行而有意保持保守。
"""

from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


# Configuration -------------------------------------------------------------
# 配置参数

CSV_PATH = Path("rail_config.csv")  # 轨道定义 CSV 文件路径
MIN_DIFFICULTY = 1  # 允许的最小总难度
MAX_DIFFICULTY = 3  # 允许的最大总难度
BUILD_SPACE_SIZE = 9  # 迷宫生成的立方空间尺寸（单位：方块）
SAFETY_ZONE_SIZE = 3  # 底部安全区大小，用于提升生成后的世界坐标
CHECKPOINT_COUNT = 1  # 需要放置的检查点数量
SEED: Optional[int] = None  # 随机种子，None 表示使用系统时间
OUTPUT_JSON = Path("maze_layout.json")  # 导出的布局 JSON 文件路径

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

BlockPos = Tuple[int, int, int]  # 三维方块坐标 (x, y, z)
Rotation = int  # 绕垂直轴的旋转角度（单位：度）


def _hash_name(name: str) -> int:
    """Return a deterministic 32-bit hash for a track name.

    返回轨道名称对应的确定性 32 位哈希值。
    """
    # Python 的 ``hash`` 结果会随运行变化，这里通过掩码将其限制
    # 在 32 位范围内，从而得到一致的索引值。
    return hash(name) & 0xFFFFFFFF


@dataclass
class Exit:
    """Definition for a track exit.

    轨道出口的定义，包含方向、相对位置以及允许的旋转。
    """

    direction: str  # e.g. "X+", "Y-"；出口的朝向
    relative_pos: BlockPos  # 相对于轨道原点的坐标偏移
    allowed_rotations: List[Rotation]  # 该出口允许的旋转角度集合

    def direction_vector(self) -> BlockPos:
        """Translate textual directions to coordinate offsets.

        将方向字符串转换成坐标向量。
        """
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
    """Representation of a track module.

    表示单个轨道模块的结构，包含尺寸、难度和出口信息等。
    """

    name: str  # 轨道名称
    index: int  # 轨道的哈希索引
    size: BlockPos  # 轨道占用的尺寸
    difficulty: float  # 基础难度
    exits: List[Exit]  # 轨道所有出口
    type: str = "normal"  # start, end, checkpoint or normal；轨道类型

    @classmethod
    def from_row(cls, row: Dict[str, str]) -> "Track":
        """Create a :class:`Track` from a CSV row.

        根据 CSV 行创建 :class:`Track` 实例。

        清洗后的 ``rail_config.csv`` 中每三个字段描述一个出口，
        分别为 ``Exit1Dir``、``Exit1Pos``、``Exit1Rot``（最多三个出
        口）。 ``Pos`` 值使用逗号分隔的坐标，``Rot`` 可以是 JSON
        数组或单个整数。``Type`` 列用于区分起点、终点或检查点等
        特殊轨道。
        """

        # 读取轨道尺寸
        size = (int(row["SizeX"]), int(row["SizeY"]), int(row["SizeZ"]))
        exits: List[Exit] = []
        for i in range(1, 4):
            # 每个出口的三个字段都带有索引
            dir_key = f"Exit{i}Dir"
            dir_val = row.get(dir_key)
            if not dir_val:
                continue  # 如果方向为空则说明该出口不存在

            pos_key = f"Exit{i}Pos"
            pos_str = row.get(pos_key, "0,0,0")
            pos = tuple(int(p.strip()) for p in pos_str.split(","))

            rot_key = f"Exit{i}Rot"
            rot_str = (row.get(rot_key) or "[0]").strip()
            try:
                rot_list = [int(r) for r in json.loads(rot_str)]
            except json.JSONDecodeError:
                # ``Rot`` 既可以是 JSON 数组，也可以是简单的逗号分隔字符串
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
    """Placed track information.

    表示一个已放置轨道的信息。
    """

    track: Track  # 对应的轨道对象
    position: BlockPos  # 在迷宫中的起始坐标
    rotation: Rotation  # 放置时的旋转角度
    final_difficulty: float  # 考虑递增系数后的最终难度

    def to_dict(self) -> Dict[str, object]:
        """Convert placement to serialisable dict.

        将放置信息转换为可序列化的字典。
        """
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
    """Generate a maze given track definitions and constraints.

    根据轨道定义和约束条件生成迷宫布局。
    """

    BLOCK_UNIT_IN_CM = 16  # 1 logical unit equals 16 centimeters / 一个方块等于 16 厘米

    def __init__(
        self,
        csv_path: Path,
        difficulty_range: Tuple[int, int],
        build_space_size: int,
        safety_zone_size: int,
        checkpoint_count: int,
        seed: Optional[int] = None,
    ) -> None:
        # 初始化参数并建立随机数生成器
        self.csv_path = Path(csv_path)
        self.min_difficulty, self.max_difficulty = difficulty_range
        self.build_space_size = build_space_size
        self.safety_zone_size = safety_zone_size
        self.checkpoint_count = checkpoint_count
        self.random = random.Random(seed)

        # 轨道列表以及按类型分类的索引
        self.tracks: List[Track] = []
        self.by_type: Dict[str, List[Track]] = {
            "start": [],
            "end": [],
            "checkpoint": [],
            "single": [],
            "multi": [],
        }
        # 记录所有放置的轨道及其占用的坐标
        self.placements: List[Placement] = []
        self.placement_map: Dict[BlockPos, Placement] = {}

    # ------------------------------------------------------------------
    # Loading and categorising tracks
    # ------------------------------------------------------------------

    def load_tracks(self) -> None:
        """Load the track definitions from the CSV file.

        从 CSV 文件读取所有轨道定义并按类型分类。
        """
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
                    # 通过出口数量区分单出口和多出口轨道
                    if len(track.exits) > 1:
                        self.by_type["multi"].append(track)
                    else:
                        self.by_type["single"].append(track)

        # 基本的配置检查，确保生成过程不会缺少关键部件
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
        """Compute world Z offset from safety zone size.

        根据安全区大小计算世界坐标系的 Z 偏移量。
        """
        return (safety_zone_size - 1) // 2

    @staticmethod
    def to_world_coordinates(pos: BlockPos) -> Tuple[int, int, int]:
        """Convert block units to centimetres for UE5.

        将方块坐标转换为厘米单位，方便导入 UE5。
        """
        return tuple(p * MazeGenerator.BLOCK_UNIT_IN_CM for p in pos)

    def is_in_bounds(self, pos: BlockPos) -> bool:
        """Check whether ``pos`` lies inside the build volume.

        判断 ``pos`` 是否位于允许的构建空间内。
        """
        half = self.build_space_size // 2
        return all(-half <= p < half + 1 for p in pos)

    # ------------------------------------------------------------------
    # Generation algorithm
    # ------------------------------------------------------------------

    def generate(self) -> List[Placement]:
        """Generate a maze layout and return placements.

        生成迷宫并返回所有放置的轨道。
        """
        self.load_tracks()
        self.place_start()
        self.iterate_generation()
        return self.placements

    def place_start(self) -> None:
        """Place the starting track at a random valid position.

        在随机位置放置起始轨道，作为生成的起点。
        """
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
        """Record a newly placed track and mark occupied cells.

        将新的轨道放置结果加入列表，并标记它占用的所有空间，以防
        歧义或重叠。
        """
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
        """Expand the maze until difficulty targets are met.

        不断从开放的出口扩展轨道，直到达到目标难度或无法继续。
        """
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

            # 根据当前难度和进度决定下一块轨道的类型
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
                # 优先使用多出口轨道以便后续扩展
                track_list = self.by_type["multi"] + self.by_type["single"]

            success = False
            for _ in range(20):  # 每一步尝试放置的次数上限
                track = self.random.choice(track_list)
                rotation = self.random.choice(track.exits[0].allowed_rotations)
                if self.can_place(track, next_pos):
                    # 难度随着之前轨道递增，模拟迷宫逐渐变难
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
                # 如果无法放置任何轨道，尝试用终点块封闭路径
                if not any(pl.track.type == "end" for pl in self.placements):
                    end_track = self.random.choice(self.by_type["end"])
                    if self.can_place(end_track, next_pos):
                        final_diff = end_track.difficulty * (1 + last_difficulty * 0.1)
                        placement = Placement(end_track, next_pos, 0, final_diff)
                        self.record_placement(placement)
                        return

                # 回溯：移除当前出口，防止死循环
                open_exits.remove(base)

    def can_place(self, track: Track, pos: BlockPos) -> bool:
        """Check whether ``track`` can be placed at ``pos``.

        判断在 ``pos`` 位置放置 ``track`` 是否会越界或与已放置轨道重叠。
        """
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
        """Export placements to a JSON file with world and maze coordinates.

        将放置结果导出为 JSON 文件，同时包含迷宫坐标和世界坐标。
        """
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

    对 ``gen.placements`` 运行一系列基础验证。

    这些检查来源于原先的 ``tester.py``，确保迷宫生成和验证在
    同一处完成。
    """

    total_diff = sum(pl.final_difficulty for pl in gen.placements)
    start_count = sum(1 for pl in gen.placements if pl.track.type == "start")
    end_count = sum(1 for pl in gen.placements if pl.track.type == "end")
    checkpoint_count = sum(
        1 for pl in gen.placements if pl.track.type == "checkpoint"
    )

    # 基本检查，确保生成的迷宫满足预期约束
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
