from gym_pybullet_drones.control.BaseControl import BaseControl
from scipy.spatial.transform import Rotation as R

import numpy as np
class RRTPlanner:
    def __init__(self, bounds, obstacles=None, step_size=0.3, max_iter=1000):
        self.bounds = bounds
        self.step = step_size
        self.max_iter = max_iter
        self.obstacles = obstacles if obstacles is not None else []

    def plan(self, start, goal):
        nodes = [start]
        parents = {0: None}
        for _ in range(self.max_iter):
            rnd = self._sample()
            idx = self._nearest(nodes, rnd)
            new = self._steer(nodes[idx], rnd)
            if self._collision_free(new) and self._edge_free(nodes[idx], new):
                nodes.append(new)
                parents[len(nodes)-1] = idx
                if np.linalg.norm(new - goal) < self.step:
                    nodes.append(goal)
                    parents[len(nodes)-1] = len(nodes)-2
                    return self._extract_path(nodes, parents)
        return None

    def _sample(self):
        return np.array([
            np.random.uniform(*self.bounds[0]),
            np.random.uniform(*self.bounds[1]),
            np.random.uniform(*self.bounds[2])
        ])

    def _nearest(self, nodes, p):
        return np.argmin([np.linalg.norm(n - p) for n in nodes])

    def _steer(self, a, b):
        d = b - a
        dist = np.linalg.norm(d)
        if dist < self.step:
            return b
        return a + self.step * d / dist

    def _collision_free(self, p):
        for obs in self.obstacles:
            if np.linalg.norm(p - obs.center) < obs.radius:
                return False
        return True

    def _edge_free(self, a, b, n=10):
        for i in range(n+1):
            p = a + i / n * (b - a)
            if not self._collision_free(p):
                return False
        return True

    def _extract_path(self, nodes, parents):
        path = []
        idx = len(nodes) - 1
        while idx is not None:
            path.append(nodes[idx])
            idx = parents[idx]
        return path[::-1]

def densify(path, ds=0.1):
    dense = []
    for i in range(len(path)-1):
        d = path[i+1] - path[i]
        L = np.linalg.norm(d)
        n = max(int(L/ds), 1)
        for j in range(n):
            dense.append(path[i] + j/n*d)
    dense.append(path[-1])
    return dense

import casadi as ca
def quad_dynamics(x, u, m=0.027, g=9.81):
    # States
    p = x[0:3]
    v = x[3:6]
    q = x[6:10]
    w = x[10:13]

    # Controls
    T = u[0]
    tau = u[1:4]

    # Rotation matrix
    qw = q[0]; qx = q[1]; qy = q[2]; qz = q[3]
    R_q = ca.vertcat(
        ca.horzcat(1 - 2*(qy**2 + qz**2), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)),
        ca.horzcat(2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2), 2*(qy*qz - qx*qw)),
        ca.horzcat(2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2))
    )

    # Translational acceleration
    a = (T / m) * R_q[:,2] - ca.vertcat(0,0,g)

    # Quaternion derivative
    wx = w[0]; wy = w[1]; wz = w[2]
    Omega = ca.vertcat(
        ca.horzcat(0, -wx, -wy, -wz),
        ca.horzcat(wx, 0, wz, -wy),
        ca.horzcat(wy, -wz, 0, wx),
        ca.horzcat(wz, wy, -wx, 0)
    )
    q_dot = 0.5 * Omega @ q

    # Angular acceleration
    I = ca.diag(ca.DM([1.4e-5, 1.4e-5, 2.17e-5]))
    Iinv = ca.inv(I)
    w_dot = Iinv @ (tau - ca.cross(w, I@w))

    # State derivative
    xdot = ca.vertcat(v, a, q_dot, w_dot)
    return xdot

def quad_mpc_cost(x, u, x_ref, u_ref, Wp=10, Wv=5, Wq=10, Ww=5, Wu=0.01):
    # Position + velocity error
    p_err = x[0:3] - x_ref[0:3]
    v_err = x[3:6] - x_ref[3:6]

    # Quaternion error (simplified for small angles)
    q_err = x[6:10] - x_ref[6:10]
    w_err = x[10:13] - x_ref[10:13]

    cost = ca.mtimes([Wp*p_err.T, p_err]) + ca.mtimes([Wv*v_err.T, v_err]) + \
           ca.mtimes([Wq*q_err.T, q_err]) + ca.mtimes([Ww*w_err.T, w_err]) + \
           ca.mtimes([Wu*u.T, u])
    return cost

class RRTMPCController(BaseControl):
    def __init__(self, drone_model, bounds, obstacles=None, horizon=50, dt=0.02):
        super().__init__(drone_model=drone_model)
        self.dt = dt
        self.N = horizon

        self.rrt = RRTPlanner(bounds, obstacles=obstacles)
        self.path = None
        self.wp_idx = 0
        self.MAX_RPM = 10000

        # URDF constants
        self.m = 0.027
        self.g = 9.81
        self.kf = 3.16e-10
        self.km = 7.94e-12
        self.arm = 0.0397
        self.MAX_RPM = 15000.0

        # MPC solver (Acados)
        self.ocp_solver = self._build_solver()

        # Warm start
        self.x_traj = [np.zeros(13) for _ in range(self.N+1)]
        self.u_traj = [np.zeros(4) for _ in range(self.N)]
        for k in range(self.N+1):
            self.x_traj[k][6:10] = np.array([1.0, 0.0, 0.0, 0.0])


    def _build_solver(self):
        from acados_template import AcadosOcp, AcadosModel, AcadosOcpSolver

        # --- States and controls ---
        x = ca.MX.sym("x", 13)  # [p,v,q,w]
        u = ca.MX.sym("u", 4)   # [T, tau_x, tau_y, tau_z]

        # Parameters: reference position, velocity, desired quaternion, initial state
        p_ref = ca.MX.sym("p_ref", 3)
        v_ref = ca.MX.sym("v_ref", 3)
        q_ref = ca.MX.sym("q_ref", 4)  # full desired quaternion
        x0_param = ca.MX.sym("x0", 13)
        params = ca.vertcat(p_ref, v_ref, q_ref, x0_param)

        # --- Model ---
        model = AcadosModel()
        model.name = "quad_bodyz_mpc"
        model.x = x
        model.u = u
        model.p = params
        model.f_expl_expr = quad_dynamics(x, u)

        # --- Quaternion error function ---
        q = x[6:10]
        q_err = ca.vertcat(q[0]-q_ref[0], q[1]-q_ref[1], q[2]-q_ref[2], q[3]-q_ref[3])  # simple linear error
        w_err = x[10:13]  # angular rates

        # Position & velocity errors
        p_err = x[0:3] - p_ref
        v_err = x[3:6] - v_ref

        # --- OCP setup ---
        ocp = AcadosOcp()
        ocp.model = model
        ocp.dims.N = self.N
        ocp.solver_options.tf = self.N * self.dt
        ocp.parameter_values = np.zeros(23)

        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.cost.cost_type_e = "NONLINEAR_LS"

        # Stack cost vector
        ocp.model.cost_y_expr = ca.vertcat(
            p_err,       # 3
            v_err,       # 3
            q_err,       # 4
            w_err,       # 3
            u - ca.vertcat(self.m*self.g, 0, 0, 0)  # control deviation
        )

        ocp.model.cost_y_expr_e = ca.vertcat(p_err, v_err, q_err)  # final cost

        # --- Weights ---
        ocp.cost.W = np.diag(
            [15,15,20] +    # position
            [6,6,8] +       # velocity
            [50,50,50,50] + # quaternion
            [5,5,5] +       # angular rates
            [0.05,0.02,0.02,0.01]  # control
        )
        ocp.cost.W_e = np.diag([20,20,25, 8,8,10, 50,50,50,50])

        # References
        ocp.cost.yref = np.zeros(17)
        ocp.cost.yref_e = np.zeros(10)

        # Input bounds
        ocp.constraints.idxbu = np.arange(4)
        tau_xy_max = 0.003
        tau_z_max  = 0.0015

        ocp.constraints.lbu = np.array([
            0.8 * self.m * self.g,
            -tau_xy_max,
            -tau_xy_max,
            -tau_z_max
        ])
        ocp.constraints.ubu = np.array([
            1.3 * self.m * self.g,
            tau_xy_max,
            tau_xy_max,
            tau_z_max
        ])

        ocp.solver_options.nlp_solver_max_iter = 100
        ocp.solver_options.nlp_solver_tol_stat = 1e-4
        ocp.solver_options.nlp_solver_tol_eq   = 1e-4
        ocp.solver_options.nlp_solver_tol_ineq = 1e-4
        ocp.solver_options.nlp_solver_tol_comp = 1e-4

        return AcadosOcpSolver(ocp, json_file="quad_bodyz_mpc.json")


    def computeRPM(self, thrust, tau):
        # Solve M*w^2 = [T, tau_x, tau_y, tau_z]
        kf, km, l = self.kf, self.km, self.arm
        M = np.array([
            [kf, kf, kf, kf],
            [0, l*kf, 0, -l*kf],
            [-l*kf, 0, l*kf, 0],
            [km, -km, km, -km]
        ])
        u = np.array([thrust, tau[0], tau[1], tau[2]])
        w2 = np.linalg.solve(M, u)
        w2 = np.clip(w2, 0, self.MAX_RPM**2)
        return np.sqrt(w2)


    def desired_quaternion_from_acc(self, a_des, yaw_des=0.0):
        zb = a_des / (np.linalg.norm(a_des) + 1e-6)

        xc = np.array([np.cos(yaw_des), np.sin(yaw_des), 0.0])
        yb = np.cross(zb, xc)
        yb /= np.linalg.norm(yb)

        xb = np.cross(yb, zb)
        R_des = np.column_stack((xb, yb, zb))

        q = R.from_matrix(R_des).as_quat()
        return np.array([q[3], q[0], q[1], q[2]])


    def computeControl(self, state, target_pos):
        state = state.flatten()
        p = state[0:3]
        v = state[3:6]
        q_pb = state[6:10]
        w = state[10:13]

        T_hover = self.m * self.g

        # ---------- RRT path planning ----------
        if self.path is None or self.wp_idx >= len(self.path):
            plan = self.rrt.plan(p, target_pos.flatten())
            if plan is None:
                plan = [p, target_pos.flatten()]
            self.path = densify(plan, ds=0.5)
            self.wp_idx = 0

        # ---------- Waypoint selection ----------
        target = self.path[self.wp_idx]
        err = target - p

        xy_err = np.linalg.norm(err[:2])
        z_err  = abs(err[2])

        if xy_err < 0.2 and z_err < 0.2:
            self.wp_idx = min(self.wp_idx + 1, len(self.path) - 1)
            target = self.path[self.wp_idx]

        # ---------- MPC reference construction ----------
        N = self.N
        path_len = len(self.path)

        kp_xy, kd_xy = 2.0, 2.0
        kp_z,  kd_z  = 3.0, 2.5

        for k in range(N):
            idx = min(self.wp_idx + k, path_len - 1)
            p_ref = self.path[idx]
            v_ref = np.zeros(3)

            # --- Desired acceleration ---
            a_xy = kp_xy * (p_ref[:2] - p[:2]) - kd_xy * v[:2]
            a_xy = np.clip(a_xy, -2.0, 2.0)

            a_z  = kp_z * (p_ref[2] - p[2]) - kd_z * v[2]

            a_des = np.array([a_xy[0], a_xy[1], self.g + a_z])

            # --- Desired orientation ---
            q_ref = self.desired_quaternion_from_acc(a_des, yaw_des=0.0)

            # --- Parameter vector (23) ---
            param = np.hstack([p_ref, v_ref, q_ref, p, v, q_pb, w])
            self.ocp_solver.set(k, "p", param)

            # --- Warm start ---
            x_warm = np.hstack([p_ref, v_ref, q_ref, np.zeros(3)])
            self.ocp_solver.set(k, "x", x_warm)
            self.ocp_solver.set(k, "u", np.array([T_hover, 0, 0, 0]))

        # ---------- Terminal step ----------
        idx = min(self.wp_idx + N, path_len - 1)
        p_ref = self.path[idx]
        v_ref = np.zeros(3)

        a_xy = kp_xy * (p_ref[:2] - p[:2]) - kd_xy * v[:2]
        a_z  = kp_z * (p_ref[2] - p[2]) - kd_z * v[2]
        a_des = np.array([a_xy[0], a_xy[1], self.g + a_z])

        q_ref = self.desired_quaternion_from_acc(a_des)

        param = np.hstack([p_ref, v_ref, q_ref, p, v, q_pb, w])
        self.ocp_solver.set(N, "p", param)
        self.ocp_solver.set(N, "x", np.hstack([p_ref, v_ref, q_ref, np.zeros(3)]))

        # ---------- Solve MPC ----------
        self.ocp_solver.solve()
        if self.ocp_solver.get_status() != 0:
            return np.ones((1,4)) * np.sqrt(T_hover / (4*self.kf))

        u0 = self.ocp_solver.get(0, "u")
        thrust = u0[0]
        tau = u0[1:4]

        rpm = self.computeRPM(thrust, tau)
        # ---------- Optional logging ----------
        Rmat = R.from_quat([q_pb[1], q_pb[2], q_pb[3], q_pb[0]]).as_matrix()
        yaw = R.from_quat([q_pb[1], q_pb[2], q_pb[3], q_pb[0]]).as_euler("xyz")[2]
        print("Waypoint:", target)
        print("Thrust:", thrust)
        print("RPM:", rpm)
        print("Body z:", Rmat[:,2])
        print("yaw (deg):", np.degrees(yaw))

        return rpm.reshape(1,4)
