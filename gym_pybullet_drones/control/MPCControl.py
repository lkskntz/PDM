import numpy as np
import casadi as ca
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel
from gym_pybullet_drones.control.BaseControl import BaseControl
from scipy.spatial.transform import Rotation as R


# =====================================================
# RRT PLANNER (unchanged)
# =====================================================
class RRTPlanner:
    def __init__(self, bounds, step_size=0.3, max_iter=1000):
        self.bounds = bounds
        self.step = step_size
        self.max_iter = max_iter

    def plan(self, start, goal):
        nodes = [start]
        parents = {0: None}

        for _ in range(self.max_iter):
            rnd = self._sample()
            idx = self._nearest(nodes, rnd)
            new = self._steer(nodes[idx], rnd)

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
            np.random.uniform(*self.bounds[2]),
        ])

    def _nearest(self, nodes, p):
        return np.argmin([np.linalg.norm(n - p) for n in nodes])

    def _steer(self, a, b):
        d = b - a
        dist = np.linalg.norm(d)
        if dist < self.step:
            return b
        return a + self.step * d / dist

    def _extract_path(self, nodes, parents):
        path = []
        idx = len(nodes) - 1
        while idx is not None:
            path.append(nodes[idx])
            idx = parents[idx]
        return path[::-1]


# =====================================================
# MPC CONTROLLER
# =====================================================
class MPCControl(BaseControl):

    def __init__(self, drone_model, g=9.81, m=0.027, horizon=50, dt=0.02):
        super().__init__(drone_model=drone_model)

        self.dt = dt
        self.N = horizon
        self.m = m
        self.g = g

        self.nx = 7
        self.nu = 3

        self.W_q = 10.0
        self.W_w = 5.0
        self.W_tau = 0.01
        self.MAX_RPM = 25000.0

        self.ocp_solver = self._build_solver()
        self.x_traj = [np.zeros(self.nx) for _ in range(self.N+1)]
        self.u_traj = [np.zeros(3) for _ in range(self.N)]

        # RRT
        self.rrt = RRTPlanner([(-5,5), (-5,5), (0.2,3)])
        self.path = None
        self.wp_idx = 0

    # ---------------------------------------------------
    def computeRPM(self, thrust, torques):
        kf = self.KF
        km = self.KM
        l  = 0.0397

        M = np.array([
            [ kf,        kf,        kf,        kf       ],
            [ 0,   l*kf,        0,  -l*kf       ],
            [-l*kf,      0,   l*kf,        0    ],
            [ km,   -km,      km,    -km       ]
        ])

        u = np.array([thrust, torques[0], torques[1], torques[2]])
        w2 = np.linalg.solve(M, u)

        w2 = np.clip(w2, 0.0, self.MAX_RPM**2)
        rpm = np.sqrt(w2)
        return rpm.reshape((4,))


    # ---------------------------------------------------
    def _rot_dynamics(self, x, u):
        q, w = x[:4], x[4:]
        tau = u

        I = ca.diag(ca.DM([1.4e-5, 1.4e-5, 2.17e-5]))
        Iinv = ca.inv(I)

        wx = w[0]; wy = w[1]; wz = w[2]
        Omega = ca.vertcat(
            ca.horzcat(0, -wx, -wy, -wz),
            ca.horzcat(wx, 0, wz, -wy),
            ca.horzcat(wy, -wz, 0, wx),
            ca.horzcat(wz, wy, -wx, 0)
        )

        q_dot = 0.5 * Omega @ q
        w_dot = Iinv @ (tau - ca.cross(w, I @ w))
        return ca.vertcat(q_dot, w_dot)

    # ---------------------------------------------------
    def _build_solver(self):
        ocp = AcadosOcp()

        x = ca.MX.sym("x", self.nx)
        u = ca.MX.sym("u", self.nu)
        q_ref = ca.MX.sym("q_ref", 4)

        model = AcadosModel()
        model.name = "quad_rot_mpc"
        model.x = x
        model.u = u
        model.p = q_ref
        model.f_expl_expr = self._rot_dynamics(x, u)
        ocp.parameter_values = np.zeros(4)

        q_err = x[:4] - q_ref

        model.cost_y_expr = ca.vertcat(
            self.W_q * q_err,
            self.W_w * x[4:7],
            self.W_tau * u
        )
        model.cost_y_expr_e = ca.vertcat(
            self.W_q * q_err,
            self.W_w * x[4:7]
        )

        ocp.model = model
        ocp.dims.N = self.N
        ocp.cost.cost_type = ocp.cost.cost_type_e = "NONLINEAR_LS"
        ocp.cost.W = np.eye(10)
        ocp.cost.W_e = np.eye(7)
        ocp.cost.yref = np.zeros(10)
        ocp.cost.yref_e = np.zeros(7)
        ocp.constraints.lbu = [-0.02, -0.02, -0.01]
        ocp.constraints.ubu = [ 0.02,  0.02,  0.01]
        ocp.constraints.idxbu = np.arange(3)
        ocp.constraints.idxbx_0 = np.arange(self.nx)
        ocp.constraints.lbx_0 = np.zeros(self.nx)
        ocp.constraints.ubx_0 = np.zeros(self.nx)

        ocp.solver_options.tf = self.N * self.dt
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"

        return AcadosOcpSolver(ocp, json_file="rot_mpc.json")

    # ---------------------------------------------------
    def computeControl(self, state, target_pos):

        state = state.flatten()
        p, q_pb = state[0:3], state[3:7]
        v, w = state[7:10], state[10:13]

        q = np.array([q_pb[3], q_pb[0], q_pb[1], q_pb[2]])
        q = q / np.linalg.norm(q)

        # ---------- RRT ----------
        if self.path is None or self.wp_idx >= len(self.path):
            self.path = self.rrt.plan(p, target_pos)
            self.wp_idx = 0

        target = target_pos if self.path is None else self.path[self.wp_idx]
        if np.linalg.norm(target - p) < 0.3:
            self.wp_idx += 1

        # ---------- POSITION PD ----------
        kp, kd = 4.0, 3.0
        a_des = kp * (target - p) - kd * v
        F_des = self.m * (a_des + np.array([0,0,self.g]))

        # ---------- THRUST FIX ----------
        T = np.linalg.norm(F_des)
        T_min = 0.05 * self.m * self.g
        T_max = 2.25 * self.m * self.g
        T = np.clip(T, T_min, T_max)

        # ---------- TILT LIMIT ----------
        z_des = F_des / np.linalg.norm(F_des)
        max_tilt = np.deg2rad(30)
        z_des[0] = np.clip(z_des[0], -np.sin(max_tilt), np.sin(max_tilt))
        z_des[1] = np.clip(z_des[1], -np.sin(max_tilt), np.sin(max_tilt))
        z_des /= np.linalg.norm(z_des)

        x_c = np.array([1,0,0])
        y_des = np.cross(z_des, x_c)
        y_des /= np.linalg.norm(y_des)
        x_des = np.cross(y_des, z_des)

        R_des = np.column_stack((x_des, y_des, z_des))
        qd = R.from_matrix(R_des).as_quat()
        q_ref = np.array([qd[3], qd[0], qd[1], qd[2]])

        # ---------- MPC ----------
        x0 = np.hstack([q, w])
        self.ocp_solver.set(0, "lbx", x0)
        self.ocp_solver.set(0, "ubx", x0)

        for k in range(self.N):
            self.ocp_solver.set(k, "p", q_ref)

        self.ocp_solver.solve()
        self.x_traj[0][:4] = self.x_traj[0][:4] / np.linalg.norm(self.x_traj[0][:4])
        tau = self.ocp_solver.get(0, "u")

        return self.computeRPM(T, tau).reshape((1,4))
