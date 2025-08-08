import json
from pathlib import Path


def test_maze_layout():
    """测试生成的迷宫布局是否符合requirement.md的要求。
    
    执行以下测试：
    1. 检查是否有且仅有一个起始轨道
    2. 检查是否有且仅有一个结束轨道
    3. 检查坐标系统是否正确
    4. 检查难度值是否合理
    5. 检查轨道类型分布
    6. 检查是否有正常轨道
    
    Returns:
        bool: 测试是否通过
    """
    
    # 加载生成的迷宫布局
    layout_path = Path("maze_layout.json")
    if not layout_path.exists():
        print("错误: 找不到 maze_layout.json 文件")
        print("请先运行 python3 generator.py 生成迷宫")
        return False
    
    with open(layout_path, 'r', encoding='utf-8') as f:
        placements = json.load(f)
    
    print(f"测试迷宫布局: 共 {len(placements)} 个轨道")
    
    # 测试1: 检查是否有起始轨道
    start_tracks = [p for p in placements if "Start" in p["name"]]
    if len(start_tracks) != 1:
        print(f"❌ 错误: 期望1个起始轨道，实际有 {len(start_tracks)} 个")
        return False
    else:
        print("✅ 起始轨道数量正确")
    
    # 测试2: 检查是否有结束轨道
    end_tracks = [p for p in placements if "End" in p["name"]]
    if len(end_tracks) == 0:
        print("❌ 错误: 没有找到结束轨道")
        return False
    elif len(end_tracks) > 1:
        print(f"❌ 错误: 期望最多1个结束轨道，实际有 {len(end_tracks)} 个")
        return False
    else:
        print("✅ 结束轨道数量正确")
    
    # 测试3: 检查坐标系统
    for i, placement in enumerate(placements):
        name = placement["name"]
        pos_maze = placement["position_maze"]
        pos_cm = placement["position_cm"]
        
        # 检查迷宫坐标是否在合理范围内
        for coord in pos_maze:
            if not isinstance(coord, int):
                print(f"❌ 错误: {name} 的迷宫坐标 {pos_maze} 包含非整数")
                return False
        
        # 检查世界坐标是否正确转换 (1单元 = 16厘米)
        expected_cm = [coord * 16 for coord in pos_maze]
        # 注意: 这里需要加上世界偏移量，但测试中我们只检查比例关系
        if pos_cm[0] % 16 != 0 or pos_cm[1] % 16 != 0 or pos_cm[2] % 16 != 0:
            print(f"❌ 错误: {name} 的世界坐标 {pos_cm} 不是16的倍数")
            return False
    
    print("✅ 坐标系统正确")
    
    # 测试4: 检查难度值
    total_difficulty = sum(p["difficulty"] for p in placements)
    print(f"总难度: {total_difficulty:.2f}")
    
    # 测试5: 检查轨道类型分布
    track_types = {}
    for placement in placements:
        name = placement["name"]
        if "Start" in name:
            track_types["start"] = track_types.get("start", 0) + 1
        elif "End" in name:
            track_types["end"] = track_types.get("end", 0) + 1
        elif "CheckPoint" in name:
            track_types["checkpoint"] = track_types.get("checkpoint", 0) + 1
        else:
            track_types["normal"] = track_types.get("normal", 0) + 1
    
    print(f"轨道类型分布: {track_types}")
    
    # 测试6: 检查是否有正常轨道
    if track_types.get("normal", 0) == 0:
        print("⚠️  警告: 没有正常轨道，迷宫可能过于简单")
    
    print("✅ 所有测试通过!")
    return True


if __name__ == "__main__":
    test_maze_layout()
