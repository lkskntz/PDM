import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import random

# =============================
# CONFIG
# =============================
use_rrt = False  # True = use RRT path, False = direct goal
dt = 0.2
N = 15
nx = 6
nu = 3
v_max = 3.0
a_max = 2.0
safety_dist = 0.6  # distance to obstacle (soft constraint)

# =============================
# ENVIRONMENT
# =============================
x_start = np.array([0.2, 0.2, 0.2])
x_goal  = np.array([8.5, 9.5, 2.2])

# Planes
z_ground = 0.0
z_ceiling = 3.0

bounds = np.array([
    [0, 10],
    [0, 10],
    [z_ground , z_ceiling]
])

# Box obstacles (pillars)
width = 0.5
x = np.array([1.0, 3.0, 6.0, 9.0])
y = np.array([1.0, 3.0, 6.0, 9.0])

pillars = []
for xi in x:
    for yi in y:
        pillars.append({
            "min": np.array([xi, yi, 0.0]),
            "max": np.array([xi+width, yi+width, z_ceiling]),
            "center": np.array([xi + width/2, yi + width/2, z_ceiling/2]),
            "radius": np.sqrt(2*(width/2)**2)  # voor soft penalty xy
        })

# =============================
# COLLISION CHECKING (RRT)
# =============================
def point_in_box(p, box):
    return np.all(p >= box["min"]) and np.all(p <= box["max"])

def collision_free(p):
    if p[2] <= z_ground or p[2] >= z_ceiling:
        return False
    for box in pillars:
        # Uitbreiding van de AABB met safety_dist
        min_box = box["min"] - safety_dist
        max_box = box["max"] + safety_dist
        if np.all(p >= min_box) and np.all(p <= max_box):
            return False
    return True

# =============================
# RRT GLOBAL PLANNER
# =============================
class Node:
    def __init__(self, pos, parent=None):
        self.pos = pos
        self.parent = parent

def rrt(start, goal, max_iter=3000, step_size=0.4):
    tree = [Node(start)]
    for _ in range(max_iter):
        rnd = goal if random.random() < 0.1 else np.array([
            random.uniform(*bounds[0]),
            random.uniform(*bounds[1]),
            random.uniform(z_ground+0.1, z_ceiling-0.1)
        ])
        nearest = min(tree, key=lambda n: np.linalg.norm(n.pos - rnd))
        direction = rnd - nearest.pos
        direction /= np.linalg.norm(direction)
        new_pos = nearest.pos + step_size*direction
        if not collision_free(new_pos): continue
        new_node = Node(new_pos, nearest)
        tree.append(new_node)
        if np.linalg.norm(new_pos - goal) < step_size:
            tree.append(Node(goal, new_node))
            break
    # Extract path
    path = []
    node = tree[-1]
    while node:
        path.append(node.pos)
        node = node.parent
    return path[::-1]

if use_rrt:
    rrt_path = rrt(x_start, x_goal)
else:
    rrt_path = [x_goal]  # MPC volgt direct doel

# =============================
# MPC SETUP
# =============================
opti = ca.Opti()
X = opti.variable(nx, N+1)
U = opti.variable(nu, N)
X0 = opti.parameter(nx)
Xref = opti.parameter(3)

cost = 0
for k in range(N):
    xk = X[:, k]
    uk = U[:, k]

    # Reference tracking
    cost += ca.dot(xk[0:3] - Xref, xk[0:3] - Xref)
    cost += 0.1*ca.dot(uk, uk)

    # Soft obstacle penalty
    for obs in pillars:
        obs_center = obs["center"]
        obs_radius = obs["radius"] + safety_dist
        dist = ca.sqrt((xk[0]-obs_center[0])**2 + (xk[1]-obs_center[1])**2)
        penalty = ca.fmax(0, obs_radius - dist)
        cost += 50 * penalty**2  # sterkte van de soft constraint

    # System dynamics
    x_next = ca.vertcat(
        xk[0] + dt*xk[3],
        xk[1] + dt*xk[4],
        xk[2] + dt*xk[5],
        xk[3] + dt*uk[0],
        xk[4] + dt*uk[1],
        xk[5] + dt*uk[2]
    )
    opti.subject_to(X[:,k+1] == x_next)

    # Input/state limits
    opti.subject_to(ca.dot(uk, uk) <= a_max**2)
    opti.subject_to(ca.dot(xk[3:6], xk[3:6]) <= v_max**2)
    opti.subject_to(xk[2] >= z_ground+0.1)
    opti.subject_to(xk[2] <= z_ceiling-0.1)

opti.subject_to(X[:,0] == X0)
opti.minimize(cost)
opti.solver("ipopt", {"print_time": False}, {"print_level": 0})

# =============================
# VISUALIZATION HELPERS
# =============================
def draw_plane(ax, z, color, alpha):
    Xp, Yp = np.meshgrid(
        np.linspace(bounds[0][0], bounds[0][1], 10),
        np.linspace(bounds[1][0], bounds[1][1], 10)
    )
    Zp = np.ones_like(Xp) * z
    ax.plot_surface(Xp, Yp, Zp, color=color, alpha=alpha, shade=False)

def draw_box(ax, box, color='gray', alpha=0.4):
    min_pt = box["min"]
    max_pt = box["max"]
    faces = [
        [[min_pt[0], min_pt[1], min_pt[2]],
         [max_pt[0], min_pt[1], min_pt[2]],
         [max_pt[0], max_pt[1], min_pt[2]],
         [min_pt[0], max_pt[1], min_pt[2]]],

        [[min_pt[0], min_pt[1], max_pt[2]],
         [max_pt[0], min_pt[1], max_pt[2]],
         [max_pt[0], max_pt[1], max_pt[2]],
         [min_pt[0], max_pt[1], max_pt[2]]],

        [[min_pt[0], min_pt[1], min_pt[2]],
         [min_pt[0], max_pt[1], min_pt[2]],
         [min_pt[0], max_pt[1], max_pt[2]],
         [min_pt[0], min_pt[1], max_pt[2]]],

        [[max_pt[0], min_pt[1], min_pt[2]],
         [max_pt[0], max_pt[1], min_pt[2]],
         [max_pt[0], max_pt[1], max_pt[2]],
         [max_pt[0], min_pt[1], max_pt[2]]],

        [[min_pt[0], min_pt[1], min_pt[2]],
         [max_pt[0], min_pt[1], min_pt[2]],
         [max_pt[0], min_pt[1], max_pt[2]],
         [min_pt[0], min_pt[1], max_pt[2]]],

        [[min_pt[0], max_pt[1], min_pt[2]],
         [max_pt[0], max_pt[1], min_pt[2]],
         [max_pt[0], max_pt[1], max_pt[2]],
         [min_pt[0], max_pt[1], max_pt[2]]],
    ]
    ax.add_collection3d(Poly3DCollection(faces, facecolors=color, alpha=alpha))

# =============================
# SIMULATION LOOP
# =============================
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x_state = np.array([*x_start, 0, 0, 0])
trajectory = [x_state[:3]]
waypoint_idx = 0

def draw():
    ax.clear()
    draw_plane(ax, z_ground, color='saddlebrown', alpha=0.4)
    draw_plane(ax, z_ceiling, color='lightgray', alpha=0.3)
    for box in pillars:
        draw_box(ax, box, color='darkgray', alpha=0.5)
    path = np.array(rrt_path)
    ax.plot(path[:,0], path[:,1], path[:,2], 'k--')
    traj = np.array(trajectory)
    ax.plot(traj[:,0], traj[:,1], traj[:,2], 'b')
    ax.scatter(*x_state[:3], c='blue', s=50)
    ax.scatter(*x_goal, c='green', s=80)
    ax.set_xlim(*bounds[0])
    ax.set_ylim(*bounds[1])
    ax.set_zlim(*bounds[2])
    ax.set_title("3D MPC with Soft Obstacles" + (" + RRT" if use_rrt else ""))
    plt.pause(0.05)

# ======== SIMULATION ========
for _ in range(1600):
    # Determine current waypoint
    if waypoint_idx < len(rrt_path) - 1:
        if np.linalg.norm(x_state[:3] - rrt_path[waypoint_idx]) < 0.3:
            waypoint_idx += 1

    opti.set_value(X0, x_state)
    opti.set_value(Xref, rrt_path[waypoint_idx])
    opti.set_initial(X, np.tile(x_state.reshape(-1,1), (1, N+1)))
    opti.set_initial(U, np.zeros((nu, N)))

    sol = opti.solve()
    u0 = sol.value(U[:,0])

    x_state = np.array([
        x_state[0] + dt*x_state[3],
        x_state[1] + dt*x_state[4],
        x_state[2] + dt*x_state[5],
        x_state[3] + dt*u0[0],
        x_state[4] + dt*u0[1],
        x_state[5] + dt*u0[2]
    ])
    trajectory.append(x_state[:3])
    draw()

plt.ioff()
plt.show()
