# =========================
# 配置 & 数据结构
# =========================

struct TrackExit:
    direction: string          # 如 "Y+", 表示相对朝向  :contentReference[oaicite:0]{index=0}
    relative_pos: tuple[int]   # 相对出口坐标偏移 (dx, dy, dz)  :contentReference[oaicite:1]{index=1}
    allowed_rotations: list[int]  # 允许自旋角度列表，如 [0,90,180,270]  :contentReference[oaicite:2]{index=2}

struct TrackDef:
    name: string               # 来自CSV的唯一名称  :contentReference[oaicite:3]{index=3}
    index: int                 # 基于name生成的hash  :contentReference[oaicite:4]{index=4}
    size: tuple[int]           # 模块占用尺寸 (sx, sy, sz) 单位：逻辑单元  :contentReference[oaicite:5]{index=5}
    base_difficulty: float     # 基础难度  :contentReference[oaicite:6]{index=6}
    exits: list[TrackExit]     # 出口列表  :contentReference[oaicite:7]{index=7}
    tags: set[string]          # 可选，用于分类（起点/终点/检查点/单出口/多出口等）

struct PlacedTrack:
    def: TrackDef
    pos: tuple[int]            # 放置在Build Space中的逻辑坐标 (x,y,z)
    facing: string             # 面向（由上一个出口决定）  :contentReference[oaicite:8]{index=8}
    spin: int                  # 自旋角度（从allowed_rotations随机）  :contentReference[oaicite:9]{index=9}

struct BuildParams:
    csv_path: string
    target_difficulty_range: tuple[int, int]  # (lower, upper)  :contentReference[oaicite:10]{index=10}
    build_space_size: int                     # 立方体奇数边长（单位：逻辑单元）  :contentReference[oaicite:11]{index=11}
    safety_zone_size: int                     # 安全边界奇数边长  :contentReference[oaicite:12]{index=12}
    checkpoint_count: int                     # 期望检查点数量  :contentReference[oaicite:13]{index=13}

struct DifficultyState:
    segment_target := target_total / (checkpoint_count + 1)  # 分段目标  :contentReference[oaicite:14]{index=14}
    segment_acc := 0
    total_acc := 0
    last_final := 0  # 上一节最终难度，用于动态系数  :contentReference[oaicite:15]{index=15}

# Grid 占用表：记录 build space 每个逻辑单元是否已被占用
PlacementMap := 3D boolean array with bounds = build_space_size^3  # 中心(0,0,0) 右手系  :contentReference[oaicite:16]{index=16}

# 世界坐标换算（用于导出给UE5时）：
# 逻辑单元 -> 厘米：1 单元 = 16 cm ；Z方向世界偏移：(safety_zone_size - 1)/2  :contentReference[oaicite:17]{index=17} :contentReference[oaicite:18]{index=18}


# =========================
# 初始化流程
# =========================

function load_tracks(csv_path) -> list[TrackDef]:
    # 解析CSV得到TrackDef集合；根据标签分类（起点/终点/检查点/单/多出口）
    # 建立若干索引：by_tag, by_size, by_exit_direction, etc.
    return track_defs

function init_context(params: BuildParams):
    tracks := load_tracks(params.csv_path)
    classify tracks into:
        start_pool, end_pool, checkpoint_trigger_pool, single_exit_pool, multi_exit_pool
        # 规则来源：初始化与分类指引  :contentReference[oaicite:19]{index=19}
    PlacementMap.clear()
    difficulty := DifficultyState(
        segment_target = (params.target_difficulty_range.upper)/ (params.checkpoint_count + 1),
        segment_acc=0, total_acc=0, last_final=0
    )
    result := empty list[PlacedTrack]
    return (tracks, pools, PlacementMap, difficulty, result)


# =========================
# 起点放置
# =========================

function place_start(pools, PlacementMap, params) -> PlacedTrack:
    pos := random_pos_within_build_space(params.build_space_size)   #  :contentReference[oaicite:20]{index=20}
    start_def := random_choice(pools.start_pool)                    #  :contentReference[oaicite:21]{index=21}
    facing := default_facing()   # 起点不旋转，面向可设定默认
    spin := 0                    # 显式设为不旋转  :contentReference[oaicite:22]{index=22}
    if can_place(start_def, pos, facing, spin, PlacementMap, params):
        occupy_cells(start_def, pos, facing, spin, PlacementMap)
        start := PlacedTrack(def=start_def, pos=pos, facing=facing, spin=spin)
        return start
    else:
        retry with a new pos (bounded attempts), else raise failure


# =========================
# 迭代主循环
# =========================

function generate_maze(params: BuildParams):
    tracks, pools, PlacementMap, difficulty, result := init_context(params)

    current := place_start(pools, PlacementMap, params)
    append result <- current

    while True:
        # 1) 从当前模块的出口集合中，随机选择一个出口
        exit := random_choice(current.def.exits)  #  :contentReference[oaicite:23]{index=23}

        # 2) 根据出口定义确定下一段面向，并从允许自旋里随机选取自旋角
        next_facing := facing_from_exit(current.facing, exit.direction)  # 朝向由上段出口决定  :contentReference[oaicite:24]{index=24}
        spin := random_choice(exit.allowed_rotations)                     # 自旋不继承上级，取出口自带列表  :contentReference[oaicite:25]{index=25}

        # 3) 计算候选放置位置（基于当前模块位置 + 出口相对偏移）
        candidate_pos := current.pos + rotate(exit.relative_pos, current.facing, spin)

        # 4) 按需筛选候选轨道（尺寸、出口匹配、难度节奏、分支需要等）
        pool := choose_pool_for_progress(pools, difficulty, params)  # 可根据“是否到检查点/终点阈值”选择类别
        candidates := filter_by_facing_and_size(pool, next_facing, bounds=params.build_space_size)

        # 5) 对候选尝试放置（碰撞检测与越界检查）
        placed := None
        for def in shuffled(candidates):
            if can_place(def, candidate_pos, next_facing, spin, PlacementMap, params):
                occupy_cells(def, candidate_pos, next_facing, spin, PlacementMap)  # 重叠/越界检查通过  :contentReference[oaicite:26]{index=26}
                placed := PlacedTrack(def=def, pos=candidate_pos, facing=next_facing, spin=spin)
                break

        # 6) 若放置失败，回溯一步重试，否则推进
        if placed is None:
            backtrack(result, PlacementMap)   # 撤销上一步  :contentReference[oaicite:27]{index=27}
            current := last(result)
            if no further backtrack available: raise generation_failure
            continue
        else:
            append result <- placed
            current := placed

        # 7) 更新难度累计
        final_diff := placed.def.base_difficulty * (1 + difficulty.last_final * 0.1)   #  :contentReference[oaicite:28]{index=28}
        difficulty.segment_acc += final_diff
        difficulty.total_acc += final_diff
        difficulty.last_final = final_diff                                           #  :contentReference[oaicite:29]{index=29}

        # 8) 检查点触发逻辑
        if should_force_checkpoint(difficulty.segment_acc, difficulty.segment_target, params):  # 超上限必触发  :contentReference[oaicite:30]{index=30}
            place_checkpoint_trigger(current, pools.checkpoint_trigger_pool, PlacementMap, params)
            difficulty.segment_acc = 0
        else if should_prob_checkpoint(difficulty.segment_acc, difficulty.segment_target):      # 过下限50%概率  :contentReference[oaicite:31]{index=31}
            if coin_flip(0.5):
                place_checkpoint_trigger(current, pools.checkpoint_trigger_pool, PlacementMap, params)
                difficulty.segment_acc = 0

        # 9) 终点触发逻辑（总难度）
        if must_end(difficulty.total_acc, params.target_difficulty_range):                    # 超上限强制  :contentReference[oaicite:32]{index=32}
            if try_place_endpoint(current, pools.end_pool, PlacementMap, params):
                break  # 终止生成  :contentReference[oaicite:33]{index=33}
        else if should_prob_end(difficulty.total_acc, params.target_difficulty_range):        # 过下限75%概率  :contentReference[oaicite:34]{index=34}
            if coin_flip(0.75) and try_place_endpoint(current, pools.end_pool, PlacementMap, params):
                break

    return result


# =========================
# 放置与碰撞检测
# =========================

function can_place(def: TrackDef, pos, facing, spin, PlacementMap, params) -> bool:
    occupied_cells := footprint_cells(def.size, pos, facing, spin)
    for cell in occupied_cells:
        if not inside_build_space(cell, params.build_space_size): return False  # 越界
        if PlacementMap[cell] == True: return False                             # 重叠
    return True

function occupy_cells(def, pos, facing, spin, PlacementMap):
    for cell in footprint_cells(def.size, pos, facing, spin):
        PlacementMap[cell] = True

function place_checkpoint_trigger(current, trigger_pool, PlacementMap, params):
    # 选择 >=2 出口的模块；一个出口产出Checkpoint Actor，另一个继续主路径  :contentReference[oaicite:35]{index=35}
    trigger_def := pick_multi_exit_def(trigger_pool)
    # 以当前出口为基准尝试相邻放置（与常规放置一致）
    # 放置成功后：在其一个出口处记录“Checkpoint Actor”生成事件；主路径从另一出口继续

function try_place_endpoint(current, end_pool, PlacementMap, params) -> bool:
    # 类似常规放置，选择终点类模块，并确保其不再生成后续路径
    end_def := select_end_def(end_pool)
    # 以当前出口为基准尝试放置，成功则返回 True


# =========================
# 导出（供UE5使用）
# =========================

function export_result(result: list[PlacedTrack], params):
    # 1) 逻辑->世界坐标（厘米）转换：每轴 *16；Z 叠加 world_offset_z=(safety_zone_size-1)/2
    # 2) 输出脚本或 JSON/CSV 等，供蓝图/构建流程读取  :contentReference[oaicite:36]{index=36}
    # 3) 轨道局部坐标系：+X前 +Y右 +Z上；1x1x1单元范围按文档  :contentReference[oaicite:37]{index=37}
