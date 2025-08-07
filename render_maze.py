#!/usr/bin/env python3
"""Render a simple ASCII visualization of a generated maze from JSON data.

读取 ``generator.py`` 生成的 JSON 文件，并在终端输出迷宫的 ASCII 示意图。
"""

from __future__ import annotations

import json
from pathlib import Path

# ``generator.py`` 输出的布局文件路径
JSON_PATH = Path("maze_layout.json")

# 各种轨道类型对应的字符符号，用于渲染
SYMBOLS = {
    "start": "S",        # 起点
    "end": "E",          # 终点
    "checkpoint": "C",   # 检查点
    "normal": "#",       # 普通轨道
}


def symbol_for(name: str) -> str:
    """根据轨道名称返回用于渲染的字符。"""

    lname = name.lower()
    if "start" in lname:
        return SYMBOLS["start"]
    if "end" in lname:
        return SYMBOLS["end"]
    if "checkpoint" in lname:
        return SYMBOLS["checkpoint"]
    return SYMBOLS["normal"]


def main() -> int:
    """读取 JSON 布局并以 ASCII 形式绘制迷宫。"""

    # 读取 JSON 数据并建立坐标到字符的映射
    data = json.loads(JSON_PATH.read_text())
    pos_map: dict[tuple[int, int], str] = {}
    for item in data:
        # ``position_maze`` 中记录了轨道在迷宫坐标系中的位置
        x, y, _ = item["position_maze"]
        pos_map[(x, y)] = symbol_for(item["name"])

    # 如果没有任何轨道，直接提示并退出
    if not pos_map:
        print("No placements to render.")
        return 0

    # 计算坐标范围，用于确定渲染区域
    xs = [p[0] for p in pos_map]
    ys = [p[1] for p in pos_map]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    # 按从上到下、从左到右的顺序输出迷宫
    for y in range(max_y, min_y - 1, -1):
        row = []
        for x in range(min_x, max_x + 1):
            # 获取当前位置的符号，若无轨道则输出 "."
            row.append(pos_map.get((x, y), "."))
        print("".join(row))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
