from pathlib import Path
import sys

# Ensure the project root is on the Python path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from MazeGeneration import MazeGenerator


def test_world_offset_z():
    """world_offset_z should compute proper Z offset and world offset vector."""
    build_space = 17
    safety_zone = 31
    mg = MazeGenerator(
        csv_path=Path('rail_config.csv'),
        difficulty_range=(0, 1),
        build_space_size=build_space,
        safety_zone_size=safety_zone,
        checkpoint_count=0,
    )

    offset_z = mg.world_offset_z(mg.safety_zone_size)
    assert offset_z == 15

    world_offset = (0, 0, offset_z)
    assert world_offset == (0, 0, 15)
