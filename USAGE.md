# 脚本使用说明书

本项目提供了一些用于构建和查看随机迷宫的脚本。

## generator.py

根据 `rail_config.csv` 生成迷宫布局。

```bash
python generator.py
```

脚本会读取 `rail_config.csv` 中的轨道定义，并将迷宫布局写入 `maze_layout.json`。可在 `generator.py` 文件顶部调整允许的难度范围和构建空间大小，然后再运行脚本。

## MazeGeneration.py

`MazeGeneration.py` 包含一个替代的生成实验。同样依赖 `rail_config.csv` 并将生成结果打印到标准输出。

运行方式：

```bash
python MazeGeneration.py
```

## render_maze.py

将 `generator.py` 生成的 `maze_layout.json` 渲染成简单的控制台表示，便于快速查看。

```bash
python render_maze.py maze_layout.json
```

## 测试

本仓库使用 `pytest` 进行单元测试。运行全部测试：

```bash
pytest
```
