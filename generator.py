#!/usr/bin/env python3
"""简单的迷宫生成器 - 从数据导入开始"""

import csv
import json
from pathlib import Path
from typing import List, Dict, Tuple

# 基本配置
CSV_PATH = Path("rail_config.csv")
OUTPUT_JSON = Path("maze_layout.json")


class TrackManager:
    """轨道管理器 - 负责存储轨道数据"""
    
    def __init__(self):
        self.tracks: List[Dict] = []
    
    def load_from_csv(self) -> bool:
        """从CSV文件加载轨道数据"""
        if not CSV_PATH.exists():
            print(f"警告: 找不到 {CSV_PATH}，使用默认配置")
            self._load_default_tracks()
            return True
        
        try:
            # 清空现有数据
            self.tracks.clear()
            
            # 从CSV文件读取数据并填充到列表中
            with CSV_PATH.open("r", encoding="utf8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    track = self._parse_csv_row(row)
                    if track:
                        self.tracks.append(track)
            
            print(f"从CSV加载了 {len(self.tracks)} 个轨道")
            return True
            
        except Exception as e:
            print(f"读取CSV文件失败: {e}")
            print("使用默认轨道配置")
            self._load_default_tracks()
            return False
    
    def _parse_csv_row(self, row: Dict[str, str]) -> Dict:
        """解析CSV行数据"""
        try:
            # 基本信息
            name = row.get("Name", "")
            difficulty_str = row.get("Difficulty", "0").strip()
            # 处理空白的难度值
            difficulty = float(difficulty_str) if difficulty_str else 0.0
            
            size = (
                int(row.get("SizeX", "1")),
                int(row.get("SizeY", "1")), 
                int(row.get("SizeZ", "1"))
            )
            track_type = row.get("Type", "normal").lower()
            
            # 解析出口
            exits = []
            for i in range(1, 4):  # 最多3个出口
                dir_key = f"Exit{i}Dir"
                dir_val = row.get(dir_key)
                if not dir_val:
                    continue
                    
                pos_key = f"Exit{i}Pos"
                pos_str = row.get(pos_key, "0,0,0")
                pos = tuple(int(p.strip()) for p in pos_str.split(","))
                
                rot_key = f"Exit{i}Rot"
                rot_str = row.get(rot_key, "[0]").strip()
                try:
                    rotations = json.loads(rot_str)
                    if isinstance(rotations, int):
                        rotations = [rotations]
                except:
                    rotations = [0]
                
                # 将方向字符串转换为坐标向量
                direction_vector = self._direction_to_vector(dir_val)
                
                exits.append({
                    "direction": dir_val,           # 原始方向字符串
                    "direction_vector": direction_vector,  # 坐标向量
                    "pos": pos,                     # 出口位置
                    "rotations": rotations          # 允许的旋转角度
                })
            
            return {
                "name": name,
                "difficulty": difficulty,
                "size": size,
                "exits": exits,
                "type": track_type
            }
            
        except Exception as e:
            print(f"解析轨道数据失败: {e}")
            return None
    
    def _direction_to_vector(self, direction: str) -> Tuple[int, int, int]:
        """将方向字符串转换为坐标向量"""
        direction_mapping = {
            "X+": (1, 0, 0),    # 正X方向
            "X-": (-1, 0, 0),   # 负X方向
            "Y+": (0, 1, 0),    # 正Y方向
            "Y-": (0, -1, 0),   # 负Y方向
            "Z+": (0, 0, 1),    # 正Z方向
            "Z-": (0, 0, -1),   # 负Z方向
        }
        
        if direction not in direction_mapping:
            print(f"警告: 未知方向 {direction}，使用默认方向 (1, 0, 0)")
            return (1, 0, 0)
        
        return direction_mapping[direction]
    
    def get_all_tracks(self) -> List[Dict]:
        """获取所有轨道"""
        return self.tracks
    
    def get_tracks_by_type(self, track_type: str) -> List[Dict]:
        """获取指定类型的轨道"""
        return [track for track in self.tracks if track["type"] == track_type]

# 输出所有已加载的轨道，便于调试
print("=== 已加载的轨道列表 ===")
for idx, track in enumerate(self.tracks):
    print(f"[{idx}] {track}")


def main():
    """主函数"""
    print("=== 迷宫生成器 - 轨道管理器测试 ===")
    
    # 创建轨道管理器
    track_manager = TrackManager()
    
    # 加载轨道数据
    success = track_manager.load_from_csv()
    
    # 显示轨道信息
    tracks = track_manager.get_all_tracks()
    print(f"\n轨道统计:")
    print(f"  总计: {len(tracks)} 个轨道")
    
    # 按类型统计
    by_type = {}
    for track in tracks:
        track_type = track["type"]
        if track_type not in by_type:
            by_type[track_type] = []
        by_type[track_type].append(track)
    
    print(f"\n按类型分布:")
    for track_type, type_tracks in by_type.items():
        print(f"  {track_type}: {len(type_tracks)} 个")
    
    # 测试获取功能
    print(f"\n测试获取功能:")
    start_tracks = track_manager.get_tracks_by_type("start")
    print(f"  起始轨道: {len(start_tracks)} 个")
    
    end_tracks = track_manager.get_tracks_by_type("end")
    print(f"  结束轨道: {len(end_tracks)} 个")


if __name__ == "__main__":
    main()
