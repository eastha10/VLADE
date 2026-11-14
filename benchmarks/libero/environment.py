# Derived from AVA-VLA @ 69ae5dfb90f4eb714259693bd2c222e0b3de6456.
# Apache-2.0 / inherited MIT: src/vlade/models/teachers/AVA_LICENSE.txt.
"""LIBERO/LIBERO+ environment construction through the installed benchmark package."""
from pathlib import Path


def create_environment(task, resolution: int = 256, seed: int = 0):
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    bddl = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl), camera_heights=resolution, camera_widths=resolution
    )
    env.seed(seed)
    return env
