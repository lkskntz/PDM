import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import random
from matplotlib.animation import FFMpegWriter
import time

# =============================
# CONFIG
# =============================
use_rrt = True          # True = use RRT path, False = direct goal
use_moving_goal = False   # True = sinus-like moving goal (turn rrt off), False = static goal

rrt_step = 2 #smaller gives more detailed path, but more expensive (may need to increase max iter in rrt)
dt = 0.2
N = 15
nx = 6
nu = 3
v_max = 3.0
a_max = 2.0
safety_dist = 0.6       # distance to obstacle (soft constraint)

goal_tolerance = 0.25 # [m] distance to goal to consider it reached
hold_time = 2.0 # [s] keep simulation running for few seconds after goal reached
max_steps = 3000 # safety cap in case goal is not reached

record_video = True
slowed_down = True
slowdown_rate = 1.0 # Recording/simulation is slowed down x.x times (visual only)
fps = int(1/dt)

# =============================
# ENVIRONMENT
# =============================
x_start = np.array([0.2, 0.2, 0.2])
x_goal_static = np.array([8.5, 8.5, 2.2])

# Boundaries
z_ground = 0.0
z_ceiling = 3.0
bounds = np.array([
    [0, 10],      # x
    [0, 10],      # y
    [z_ground, z_ceiling]  # z
])

# =============================
# OBSTACLES
# =============================
pillars = []

# Parameters
width = 0.5
beam_thickness = 0.3
beam_height_low = 0.6
beam_height_high = 2.0
x = np.array([1.0, 3.0, 6.0, 9.0])
y = np.array([1.0, 3.0, 6.0, 9.0])

# Vertical pillars
for xi in x:
    for yi in y:
        min_pt = np.array([xi, yi, z_ground])
        max_pt = np.array([xi + width, yi + width, z_ceiling])
        center = 0.5 * (min_pt + max_pt)
        radius = np.linalg.norm(max_pt[:2] - center[:2])
        pillars.append({"type":"pillar","min":min_pt,"max":max_pt,"center":center,"radius":radius})

# Horizontal beams
beam_extension = 1.0
for yi in y:
    for zc in [beam_height_low, beam_height_high]:
        min_pt = np.array([x[0]-beam_extension, yi, zc])
        max_pt = np.array([x[-1]+width+beam_extension, yi+width, zc+beam_thickness])
        center = 0.5*(min_pt+max_pt)
        radius = np.linalg.norm(max_pt[:2]-center[:2])
        pillars.append({"type":"beam","min":min_pt,"max":max_pt,"center":center,"radius":radius})

# Large green block
plants = {
    "type": "block",
    "min": np.array([1.0, 1.0, 0.0]),
    "max": np.array([9.0, 9.0, 0.6]),
    "center": 0.5*(np.array([1.0,1.0,0.0]) + np.array([9.0,9.0,2.0])),
    "radius": np.linalg.norm(np.array([9.0,9.0,0.0]) - np.array([1.0,1.0,0.0]))/2
}
pillars.append(plants)

# =============================
# COLLISION CHECK
# =============================
def collision_free(p, rrt_mode=False):
    # check bounds
    if p[0] <= bounds[0,0] or p[0] >= bounds[0,1]:
        return False
    if p[1] <= bounds[1,0] or p[1] >= bounds[1,1]:
        return False
    if p[2] <= bounds[2,0] or p[2] >= bounds[2,1]:
        return False


    # check obstacles
    for box in pillars:
        min_box = box["min"] - safety_dist
        max_box = box["max"] + safety_dist
        if np.all(p >= min_box) and np.all(p <= max_box):
            return False
    return True

# =============================
# SINUS MOVING GOAL
# =============================
def moving_goal(t):
    use_rrt = False
    x = 8 + 1.5*np.sin(0.3*t)
    y = 8 + 1.5*np.cos(0.2*t)
    z = 2.2
    return np.array([x,y,z])

# =============================
# RRT GLOBAL PLANNER
# =============================
class Node:
    def __init__(self, pos, parent=None):
        self.pos = pos
        self.parent = parent

def rrt(start, goal, max_iter=6000, step_size= rrt_step):
    tree = [Node(start)]
    for _ in range(max_iter):
        rnd = goal if random.random()<0.1 else np.array([
            random.uniform(*bounds[0]),
            random.uniform(*bounds[1]),
            random.uniform(bounds[2,0]+0.1, bounds[2,1]-0.1)
        ])
        nearest = min(tree, key=lambda n: np.linalg.norm(n.pos - rnd))
        direction = rnd - nearest.pos
        direction /= np.linalg.norm(direction)
        new_pos = nearest.pos + step_size*direction
        if not collision_free(new_pos, rrt_mode=True): 
            continue
        new_node = Node(new_pos, nearest)
        tree.append(new_node)
        if np.linalg.norm(new_pos - goal) < step_size:
            tree.append(Node(goal,new_node))
            break
    # extract path
    path=[]
    node=tree[-1]
    while node:
        path.append(node.pos)
        node=node.parent
    return path[::-1]


if use_rrt:
    rrt_path = rrt(x_start, x_goal_static)
else:
    rrt_path = [x_goal_static]


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
    xk = X[:,k]
    uk = U[:,k]

    # Reference tracking
    cost += ca.dot(xk[0:3]-Xref, xk[0:3]-Xref)
    cost += 0.1*ca.dot(uk, uk)

    # 3D obstacle penalty
    for obs in pillars:
        dx = ca.fmax(obs["min"][0]-xk[0],0) + ca.fmax(xk[0]-obs["max"][0],0)
        dy = ca.fmax(obs["min"][1]-xk[1],0) + ca.fmax(xk[1]-obs["max"][1],0)
        dz = ca.fmax(obs["min"][2]-xk[2],0) + ca.fmax(xk[2]-obs["max"][2],0)
        dist = ca.sqrt(dx*dx + dy*dy + dz*dz + 1e-6)
        penalty = ca.fmax(0, safety_dist - dist)
        cost += 200*penalty**2

    # Dynamics
    x_next = ca.vertcat(
        xk[0]+dt*xk[3],
        xk[1]+dt*xk[4],
        xk[2]+dt*xk[5],
        xk[3]+dt*uk[0],
        xk[4]+dt*uk[1],
        xk[5]+dt*uk[2]
    )
    opti.subject_to(X[:,k+1]==x_next)

    # Input/state limits
    opti.subject_to(ca.dot(uk,uk) <= a_max**2)
    opti.subject_to(ca.dot(xk[3:6],xk[3:6]) <= v_max**2)

    # Bounds (walls + floor/ceiling)
    opti.subject_to(X[0,k] >= bounds[0,0]+0.1)
    opti.subject_to(X[0,k] <= bounds[0,1]-0.1)
    opti.subject_to(X[1,k] >= bounds[1,0]+0.1)
    opti.subject_to(X[1,k] <= bounds[1,1]-0.1)
    opti.subject_to(X[2,k] >= bounds[2,0]+0.1)
    opti.subject_to(X[2,k] <= bounds[2,1]-0.1)

opti.subject_to(X[:,0]==X0)
opti.minimize(cost)
opti.solver("ipopt",{"print_time":False},{"print_level":0})

# =============================
# VISUALIZATION
# =============================
def draw_plane(ax,z,color,alpha):
    Xp,Yp=np.meshgrid(np.linspace(bounds[0][0],bounds[0][1],10),
                      np.linspace(bounds[1][0],bounds[1][1],10))
    Zp=np.ones_like(Xp)*z
    ax.plot_surface(Xp,Yp,Zp,color=color,alpha=alpha,shade=False)

def draw_box(ax,box,color='gray',alpha=0.4):
    min_pt = box["min"]
    max_pt = box["max"]
    faces=[
        [[min_pt[0],min_pt[1],min_pt[2]],
         [max_pt[0],min_pt[1],min_pt[2]],
         [max_pt[0],max_pt[1],min_pt[2]],
         [min_pt[0],max_pt[1],min_pt[2]]],

        [[min_pt[0],min_pt[1],max_pt[2]],
         [max_pt[0],min_pt[1],max_pt[2]],
         [max_pt[0],max_pt[1],max_pt[2]],
         [min_pt[0],max_pt[1],max_pt[2]]],

        [[min_pt[0],min_pt[1],min_pt[2]],
         [min_pt[0],max_pt[1],min_pt[2]],
         [min_pt[0],max_pt[1],max_pt[2]],
         [min_pt[0],min_pt[1],max_pt[2]]],

        [[max_pt[0],min_pt[1],min_pt[2]],
         [max_pt[0],max_pt[1],min_pt[2]],
         [max_pt[0],max_pt[1],max_pt[2]],
         [max_pt[0],min_pt[1],max_pt[2]]],

        [[min_pt[0],min_pt[1],min_pt[2]],
         [max_pt[0],min_pt[1],min_pt[2]],
         [max_pt[0],min_pt[1],max_pt[2]],
         [min_pt[0],min_pt[1],max_pt[2]]],

        [[min_pt[0],max_pt[1],min_pt[2]],
         [max_pt[0],max_pt[1],min_pt[2]],
         [max_pt[0],max_pt[1],max_pt[2]],
         [min_pt[0],max_pt[1],max_pt[2]]]
    ]
    ax.add_collection3d(Poly3DCollection(faces,facecolors=color,alpha=alpha))

# =============================
# SIMULATION LOOP
# =============================
plt.ion()
fig=plt.figure()
ax=fig.add_subplot(111,projection='3d')

x_state = np.array([*x_start,0,0,0])
trajectory = [x_state[:3]]
waypoint_idx = 0
goal_pos = x_goal_static.copy()
t_sim = 0.0

def draw():
    ax.clear()
    draw_plane(ax,bounds[2,0],'saddlebrown',0.4)
    draw_plane(ax,bounds[2,1],'lightgray',0.3)
    for box in pillars:
        if box["type"]=="block":
            draw_box(ax,box,'green',0.4)
        else:
            draw_box(ax,box,'darkgray',0.5)
    path=np.array(rrt_path)
    ax.plot(path[:,0],path[:,1],path[:,2],'k--')
    traj=np.array(trajectory)
    ax.plot(traj[:,0],traj[:,1],traj[:,2],'b')
    ax.scatter(*x_state[:3],c='blue',s=50)
    ax.scatter(*goal_pos,c='red' if use_moving_goal else 'green',s=80)
    ax.set_xlim(*bounds[0])
    ax.set_ylim(*bounds[1])
    ax.set_zlim(*bounds[2])
    ax.set_title("3D MPC with Soft Obstacles" + (" + RRT" if use_rrt else ""))
    plt.pause(0.05)

# ===== VIDEO RECORDING =====
if use_rrt:
    writer = FFMpegWriter(
        fps=fps,
        metadata=dict(title="3D MPC+RRT+CrossBeams", artist="matplotlib"),
        bitrate=1000
    )
    video_filename = "mpc_rrt_crossbeams_simulation.mp4"
else:
    writer = FFMpegWriter(
        fps=fps,
        metadata=dict(title="3D MPC+CrossBeams", artist="matplotlib"),
        bitrate=1000
    )
    video_filename = "mpc_crossbeams_simulation.mp4"


# ======== SIMULATION ========
goal_reached = False
goal_reached_step = None
goal_reached_time = None

step = 0
max_hold_steps = int(hold_time / dt)

if record_video:
    with writer.saving(fig, video_filename, dpi=150):
        while step < max_steps:
            # Update goal
            if use_moving_goal:
                goal_pos = moving_goal(t_sim)
            else:
                goal_pos = x_goal_static.copy()

            # Stop condition
            dist_g = np.linalg.norm(x_state[:3]-goal_pos)
            print(np.round(dist_g, 4)) # prints the distance to the target, when it is smaller then 0.4 we consider the target to be 'hit'

            if not goal_reached and dist_g < goal_tolerance:
                goal_reached = True
                goal_reached_step = step
                goal_reached_time = goal_reached_step * dt
                print(f"Goal reached at step {goal_reached_step}!")
                print(f"Goal was reached in {goal_reached_time} seconds")

            # Determine current waypoint
            if use_rrt:
                if waypoint_idx < len(rrt_path)-1 and np.linalg.norm(x_state[:3]-rrt_path[waypoint_idx])<0.3:
                    waypoint_idx += 1
                target = rrt_path[waypoint_idx]
            else:
                target = goal_pos

            opti.set_value(X0,x_state)
            opti.set_value(Xref,target)
            opti.set_initial(X,np.tile(x_state.reshape(-1,1),(1,N+1)))
            opti.set_initial(U,np.zeros((nu,N)))

            sol = opti.solve()
            u0 = sol.value(U[:,0])

            # Integrate dynamics
            x_state = np.array([
                x_state[0]+dt*x_state[3],
                x_state[1]+dt*x_state[4],
                x_state[2]+dt*x_state[5],
                x_state[3]+dt*u0[0],
                x_state[4]+dt*u0[1],
                x_state[5]+dt*u0[2]
            ])
            trajectory.append(x_state[:3])
            t_sim += dt

            if not goal_reached and step == max_steps - 1:
                print("Goal not reached")
            
            if goal_reached_step is not None and step - goal_reached_step >= max_hold_steps:
                print("Hold time over, simulation is stopped")
                draw()
                writer.grab_frame()
                plt.ioff()
                plt.close(fig)
                break

            draw()
            writer.grab_frame()
            if slowed_down:
                time.sleep(dt * slowdown_rate)
            step += 1

else:
    while step < max_steps:
        # Update goal
        if use_moving_goal:
            goal_pos = moving_goal(t_sim)
        else:
            goal_pos = x_goal_static.copy()

        # Stop condition
        dist_g = np.linalg.norm(x_state[:3]-goal_pos)
        print(np.round(dist_g, 4)) # prints the distance to the target, when it is smaller then 0.4 we consider the target to be 'hit'

        if not goal_reached and dist_g < goal_tolerance:
            goal_reached = True
            goal_reached_step = step
            goal_reached_time = goal_reached_step * dt
            print(f"Goal reached at step {goal_reached_step}!")
            print(f"Goal was reached in {goal_reached_time} seconds")

        # Determine current waypoint
        if use_rrt:
            if waypoint_idx < len(rrt_path)-1 and np.linalg.norm(x_state[:3]-rrt_path[waypoint_idx])<0.3:
                waypoint_idx += 1
            target = rrt_path[waypoint_idx]
        else:
            target = goal_pos

        opti.set_value(X0,x_state)
        opti.set_value(Xref,target)
        opti.set_initial(X,np.tile(x_state.reshape(-1,1),(1,N+1)))
        opti.set_initial(U,np.zeros((nu,N)))

        sol = opti.solve()
        u0 = sol.value(U[:,0])

        # Integrate dynamics
        x_state = np.array([
            x_state[0]+dt*x_state[3],
            x_state[1]+dt*x_state[4],
            x_state[2]+dt*x_state[5],
            x_state[3]+dt*u0[0],
            x_state[4]+dt*u0[1],
            x_state[5]+dt*u0[2]
        ])
        trajectory.append(x_state[:3])
        t_sim += dt

        if not goal_reached and step == max_steps - 1:
            print("Goal not reached")
        
        if goal_reached_step is not None and step - goal_reached_step >= max_hold_steps:
            print("Hold time over, simulation is stopped")
            draw()
            writer.grab_frame()
            plt.ioff()
            plt.close(fig)
            break

        draw()
        writer.grab_frame()
        if slowed_down:
            time.sleep(dt * slowdown_rate)
        step += 1

plt.ioff()
plt.show()
