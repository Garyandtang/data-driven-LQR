from enum import Enum


class Task(str, Enum):
    """Environment tasks enumeration class."""

    STABILIZATION = 'stabilization'  # Stabilization task.
    TRAJ_TRACKING = 'traj_tracking'  # Trajectory tracking task.
