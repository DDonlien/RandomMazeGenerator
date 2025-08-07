"""
兼容性导入模块。

早期版本的工程通过 ``MazeGeneration`` 模块暴露迷宫生成器类。
为保持旧代码能够继续工作，此文件简单地从新的 ``generator`` 模块
中导入 :class:`MazeGenerator` 并在 ``__all__`` 中导出它。
"""

from generator import MazeGenerator  # 导入核心的迷宫生成器类

# 控制 ``from MazeGeneration import *`` 的导出内容，仅暴露 ``MazeGenerator``
__all__ = ["MazeGenerator"]
