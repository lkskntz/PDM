"""
High-level test for ObstacleHoverEnv + RRT3D.

- Builds the environment with obstacles
- Plans a global path using RRT
- Moves the drone along the path with a simple proportional controller
- Visualizes path in PyBullet
"""

import time
import numpy as np
import pybullet as p

from custom_env import ObstacleHoverEnv
from rrt_planner import RRT3D
from gym_pybullet_drones.utils.enums import ActionType

# ----------------------------
# 1️⃣ Create environment
# ----------------------------
ctrl_freq = 48   # control frequency in Hz
dt = 1.0 / ctrl_freq

env = ObstacleHoverEnv(
    phase=2,               # 1, 2, or 3
    gui=True,
    record=False,
    pyb_freq=240,
    ctrl_freq=ctrl_freq,
    act=ActionType.VEL  # velocity control for simple proportional controller
)

obs = env.reset()

# ----------------------------
# 2️⃣ Define collision function
# ----------------------------
def collision_fn(pos):
    """
    Check if drone at `pos` collides with obstacles.
    Uses AABB overlap queries in PyBullet.
    """
    drone_radius = 0.2
    aabb_min = [pos[0]-drone_radius, pos[1]-drone_radius, pos[2]-drone_radius]
    aabb_max = [pos[0]+drone_radius, pos[1]+drone_radius, pos[2]+drone_radius]
    overlapping = p.getOverlappingObjects(aabb_min, aabb_max)
    return overlapping is None  # True if no collision

# ----------------------------
# 3️⃣ Define RRT planner
# ----------------------------
start_pos = np.array([-6.0, -4.0, 1.0]) 
#start_pos = env._getDroneStateVector(0)[:3]
goal_pos = [1.0, 1.0, 1.0]  # inside safe bounds to avoid truncation
bounds = [[-1.5, 1.5], [-1.5, 1.5], [0, 2.0]]

planner = RRT3D(start=start_pos, goal=goal_pos, bounds=bounds, is_collision_free=collision_fn)
path = planner.plan(max_iter=2000, step_size=0.2)

if path is None:
    print("RRT failed to find a path!")
    env.close()
    exit()

# ----------------------------
# 4️⃣ Visualize path
# ----------------------------
for i in range(len(path)-1):
    p.addUserDebugLine(path[i], path[i+1], [1,0,0], 2)  # red line

# ----------------------------
# 5️⃣ Fly drone along path
# ----------------------------
Kp = 0.8  # proportional gain

for waypoint in path:
    while True:
        # Get current drone position
        pos = env._getDroneStateVector(0)[:3]

        # Compute velocity command
        vel_cmd = np.array(waypoint) - np.array(pos)
        vel_cmd = np.clip(vel_cmd * Kp, -1.0, 1.0)  # limit max velocity

        # Step environment
        obs, reward, terminated, truncated, info = env.step([vel_cmd.tolist()])

        # Check if reached waypoint or truncated/terminated
        if np.linalg.norm(pos - np.array(waypoint)) < 0.1 or terminated or truncated:
            break

        time.sleep(dt)

# ----------------------------
# 6️⃣ Keep GUI open and close
# ----------------------------
input("Path completed! Press Enter to exit...")
env.close()
