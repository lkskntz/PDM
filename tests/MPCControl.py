import numpy as np
import casadi as ca
from gym_pybullet_drones.control.BaseControl import BaseControl
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel

class MPCControl(BaseControl):
    def __init__(self, drone_model, g=9.81, m=0.03, horizon=50, dt=0.02):
        super().__init__(drone_model=drone_model)
        self.g = g
        self.m = m
        self.horizon = horizon
        self.dt = dt

        self.nx = 12
        self.nu = 4

        # Adjusted weights: allow tilting for horizontal motion
        self.Q = np.diag([30, 30, 45,
                          10.0, 10.0, 5.0,
                          8.0, 8.0, 12.0,
                          .80, .80, 10.0])
        self.R = np.diag([0.2, 1.0, 1.0, 0.5])  # increased torque weights

        self.ocp_solver = self.build_acados_nmpc()

    def computeRPM(self, thrust, torques):
        k_f = self.KF
        k_m = self.KM
        M = np.array([
            [k_f, k_f, k_f, k_f],
            [0, k_f, 0, -k_f],
            [-k_f, 0, k_f, 0],
            [k_m, -k_m, k_m, -k_m]
        ])
        w2 = np.linalg.pinv(M) @ np.hstack([thrust, torques])
        rpm = np.sqrt(np.maximum(w2, 0)) * 60 / (2 * np.pi)
        return rpm

    def quad_dynamics_casadi(self, x, u):
        px, py, pz = x[0], x[1], x[2]
        phi, theta, psi = x[3], x[4], x[5]
        vx, vy, vz = x[6], x[7], x[8]
        wx, wy, wz = x[9], x[10], x[11]
        T, tx, ty, tz = u[0], u[1], u[2], u[3]

        Ixx, Iyy, Izz = 1.4e-5, 1.4e-5, 2.17e-5
        J = ca.diag(ca.DM([Ixx, Iyy, Izz]))
        Jinv = ca.inv(J)

        cphi, sphi = ca.cos(phi), ca.sin(phi)
        cth, sth = ca.cos(theta), ca.sin(theta)
        cpsi, spsi = ca.cos(psi), ca.sin(psi)
        R_bw = ca.vertcat(
            ca.horzcat(cpsi*cth, cpsi*sth*sphi - spsi*cphi, cpsi*sth*cphi + spsi*sphi),
            ca.horzcat(spsi*cth, spsi*sth*sphi + cpsi*cphi, spsi*sth*cphi - cpsi*sphi),
            ca.horzcat(-sth, cth*sphi, cth*cphi)
        )

        p_dot = ca.vertcat(vx, vy, vz)
        E = ca.vertcat(
            ca.horzcat(1, sphi*ca.tan(theta), cphi*ca.tan(theta)),
            ca.horzcat(0, cphi, -sphi),
            ca.horzcat(0, sphi/cth, cphi/cth)
        )
        att_dot = E @ ca.vertcat(wx, wy, wz)

        v_dot = (R_bw @ ca.vertcat(0, 0, T)) / self.m - ca.vertcat(0, 0, self.g)
        D = ca.diag(ca.DM([3e-4, 3e-4, 5e-4]))  # damping
        omega = ca.vertcat(wx, wy, wz)

        omega_dot = Jinv @ (
            ca.vertcat(tx, ty, tz)
            - ca.cross(omega, J @ omega)
            - D @ omega)

        return ca.vertcat(p_dot, att_dot, v_dot, omega_dot)

    def build_acados_nmpc(self):
        ocp = AcadosOcp()
        x = ca.MX.sym("x", self.nx)
        u = ca.MX.sym("u", self.nu)
        f_expl = ca.Function("f_expl", [x, u], [self.quad_dynamics_casadi(x, u)])

        # ---------------- Model ----------------
        ocp.model = AcadosModel()
        ocp.model.name = 'quadrotor'
        ocp.model.x = x
        ocp.model.u = u
        ocp.model.f_expl_expr = f_expl(x, u)
        ocp.model.p = []

        # ---------------- Horizon ----------------
        ocp.dims.N = self.horizon

        # ---------------- Cost setup ----------------

        nx, nu = self.nx, self.nu
        ny = nx + nu

        # nonlinear cost expressions
        ocp.model.cost_y_expr = ca.vertcat(x, u)
        ocp.model.cost_y_expr_e = x

        ocp.cost.cost_type = "NONLINEAR_LS"
        ocp.cost.cost_type_e = "NONLINEAR_LS"

        ocp.cost.W = np.block([
            [self.Q, np.zeros((nx, nu))],
            [np.zeros((nu, nx)), self.R]
        ])
        ocp.cost.W_e = self.Q

        ocp.cost.yref = np.zeros(ny)
        ocp.cost.yref_e = np.zeros(nx)


        # ---------------- Input constraints ----------------
        T_hover = self.m * self.g
        ocp.constraints.lbu = [0.1*T_hover, -0.1, -0.1, -0.05]
        ocp.constraints.ubu = [1.5*T_hover, 0.1, 0.1, 0.05]
        ocp.constraints.idxbu = np.array([0,1,2,3])

        # ---------------- Initial state constraints ----------------
        # Placeholders; actual state will be set in computeControl()
        ocp.constraints.idxbx_0 = np.arange(self.nx)
        ocp.constraints.lbx_0 = np.zeros(self.nx)
        ocp.constraints.ubx_0 = np.zeros(self.nx)

        max_tilt = 0.5  # rad ≈ 28°
        ocp.constraints.lbx = np.array([-max_tilt, -max_tilt])
        ocp.constraints.ubx = np.array([ max_tilt,  max_tilt])
        ocp.constraints.idxbx = np.array([3, 4])

        # ---------------- Solver options ----------------
        ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
        ocp.solver_options.integrator_type = "ERK"
        ocp.solver_options.tf = self.horizon * self.dt

        # ---------------- Create solver ----------------
        solver = AcadosOcpSolver(ocp, json_file="acados_ocp.json")
        return solver

    def computeControl(
        self,
        state,
        target_pos,
        target_rpy=np.zeros(3),
        target_vel=np.zeros(3),
        target_rpy_rates=np.zeros(3),
    ):
        state = np.asarray(state).flatten()
        assert state.size >= 12

        pos = state[0:3]
        rpy = state[3:6]
        vel = state[6:9]
        omega = state[9:12]

        x0 = np.hstack([pos, rpy, vel, omega])

        x_ref = np.hstack([
            target_pos,
            target_rpy,
            target_vel,
            target_rpy_rates
        ])

        u_hover = np.array([self.m * self.g, 0.0, 0.0, 0.0])
        yref_stage = np.hstack([x_ref, u_hover])

        # --------------------------------------------------
        # Initial state constraint
        # --------------------------------------------------
        self.ocp_solver.set(0, "lbx", x0)
        self.ocp_solver.set(0, "ubx", x0)

        # --------------------------------------------------
        # Warm start (shift previous solution)
        # --------------------------------------------------
        if not hasattr(self, "x_traj"):
            # First call: initialize guesses
            self.x_traj = [x0.copy() for _ in range(self.horizon + 1)]
            self.u_traj = [u_hover.copy() for _ in range(self.horizon)]
        else:
            # Shift trajectories
            self.x_traj[:-1] = self.x_traj[1:]
            self.x_traj[-1] = self.x_traj[-2].copy()

            self.u_traj[:-1] = self.u_traj[1:]
            self.u_traj[-1] = self.u_traj[-2].copy()

        # --------------------------------------------------
        # Set warm-start guesses and references
        # --------------------------------------------------
        for k in range(self.horizon):
            self.ocp_solver.set(k, "x", self.x_traj[k])
            self.ocp_solver.set(k, "u", self.u_traj[k])
            self.ocp_solver.set(k, "yref", yref_stage)

        # Terminal reference (state only)
        self.ocp_solver.set(self.horizon, "yref", x_ref)

        # --------------------------------------------------
        # Solve NMPC
        # --------------------------------------------------
        status = self.ocp_solver.solve()

        if status != 0:
            # Solver failed → safe fallback
            thrust = self.m * self.g
            torques = np.zeros(3)
            rpm = self.computeRPM(thrust, torques)
            return rpm.reshape((1, 4))

        # --------------------------------------------------
        # Extract optimal control and trajectory
        # --------------------------------------------------
        u_opt = self.ocp_solver.get(0, "u")
        thrust, torques = u_opt[0], u_opt[1:]

        # Save full solution for next warm-start
        for k in range(self.horizon):
            self.x_traj[k] = self.ocp_solver.get(k, "x")
            self.u_traj[k] = self.ocp_solver.get(k, "u")
        self.x_traj[self.horizon] = self.ocp_solver.get(self.horizon, "x")

        rpm = self.computeRPM(thrust, torques)
        return rpm.reshape((1, 4))
