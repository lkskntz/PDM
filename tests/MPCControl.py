import numpy as np
import cvxpy as cp
from scipy.spatial.transform import Rotation
from gym_pybullet_drones.control.BaseControl import BaseControl
from gym_pybullet_drones.utils.enums import DroneModel

class MPCControl(BaseControl):
    """
    Linear Model Predictive Control (MPC) Class using CVXPY.
    Linearized around hover state.
    """

    def __init__(self, 
                 drone_model: DroneModel, 
                 g: float = 9.8,
                 horizon: int = 10,
                 dt: float = 0.05):
        """
        Initialization.
        """
        super().__init__(drone_model=drone_model, g=g)
        self.H = horizon
        self.dt = dt
        
        # --- ROBUST PARAMETER INITIALIZATION ---
        # BaseControl in some versions does not set self.M or self.J.
        # We manually set them for the standard Crazyflie 2.x model if missing.
        
        # Mass
        if not hasattr(self, 'M'):
            self.M = 0.027
            
        # Inertia (diagonal params)
        if not hasattr(self, 'J'):
            self.J = np.diag([1.4e-5, 1.4e-5, 2.17e-5])
            
        # Arm length (center to motor)
        if not hasattr(self, 'L'):
            self.L = 0.0397

        # Thrust coefficient
        if not hasattr(self, 'KF'):
            self.KF = 3.16e-10
            
        # Torque coefficient
        if not hasattr(self, 'KM'):
            self.KM = 7.94e-12

        # --- Dynamics Model (Linearized around Hover) ---
        # State: x = [px, py, pz, phi, theta, psi, vx, vy, vz, p, q, r] (12x1)
        # Input: u = [Thrust, Tau_x, Tau_y, Tau_z] (4x1)
        
        m = self.M
        Ixx = self.J[0,0]
        Iyy = self.J[1,1]
        Izz = self.J[2,2]

        # Continuous A matrix (approximate)
        A_c = np.zeros((12, 12))
        
        # Position derivative -> Velocity
        A_c[0:3, 6:9] = np.eye(3)
        
        # Velocity derivative -> Angles (g linearization)
        A_c[6, 4] = g       # x_acc approx g * theta
        A_c[7, 3] = -g      # y_acc approx -g * phi
        
        # Angle derivative -> Angular rates
        A_c[3:6, 9:12] = np.eye(3)
        
        # Continuous B matrix
        B_c = np.zeros((12, 4))
        B_c[8, 0] = 1/m         # z_acc driven by Thrust (u1)
        B_c[9, 1] = 1/Ixx       # p_dot driven by Torque X
        B_c[10, 2] = 1/Iyy      # q_dot driven by Torque Y
        B_c[11, 3] = 1/Izz      # r_dot driven by Torque Z

        # Discretize (Euler first order)
        self.A = np.eye(12) + A_c * self.dt
        self.B = B_c * self.dt

        # --- MPC Problem Setup (CVXPY) ---
        self.nx = 12
        self.nu = 4
        
        # Variables
        self.x_var = cp.Variable((self.nx, self.H + 1))
        self.u_var = cp.Variable((self.nu, self.H))
        
        # Parameters (to be updated every step)
        self.x_init = cp.Parameter(self.nx)
        self.x_ref = cp.Parameter(self.nx)
        
        # Weights (Tune these to change aggressive/smooth behavior)
        Q_diag = np.array([500, 500, 500,   # Position (x,y,z)
                           10, 10, 10,      # Attitude (r,p,y)
                           10, 10, 10,      # Velocity
                           1, 1, 1])        # Rates
        R_diag = np.array([0.1, 0.1, 0.1, 0.1]) # Input effort
        
        cost = 0
        constraints = []
        
        constraints.append(self.x_var[:, 0] == self.x_init)
        
        u_hover = np.array([m * g, 0, 0, 0]) # Nominal input
        
        for t in range(self.H):
            # Cost
            cost += cp.quad_form(self.x_var[:, t] - self.x_ref, np.diag(Q_diag))
            cost += cp.quad_form(self.u_var[:, t] - u_hover, np.diag(R_diag))
            
            # Dynamics
            constraints.append(self.x_var[:, t+1] == self.A @ self.x_var[:, t] + self.B @ self.u_var[:, t])
            
            # Input Constraints (Thrust >= 0, approx max thrust)
            constraints.append(self.u_var[0, t] <= 4 * m * g * 2.0) 
            constraints.append(self.u_var[0, t] >= 0)

        # Terminal Cost
        cost += cp.quad_form(self.x_var[:, self.H] - self.x_ref, np.diag(Q_diag))

        self.prob = cp.Problem(cp.Minimize(cost), constraints)

    def computeControl(self,
                       cur_pos,
                       cur_quat,
                       cur_vel,
                       cur_ang_vel,
                       target_pos,
                       target_vel=np.zeros(3),
                       target_rpy=np.zeros(3),
                       target_rpy_rates=np.zeros(3)):
        """
        Computes the control action using the MPC solver.
        """
        # 1. State Estimation / Prep
        rpy = Rotation.from_quat(cur_quat).as_euler('xyz', degrees=False)
        
        # Current state vector x = [pos, rpy, vel, ang_vel]
        x_curr = np.hstack([cur_pos, rpy, cur_vel, cur_ang_vel])
        
        # Reference state vector
        x_target = np.hstack([target_pos, target_rpy, target_vel, target_rpy_rates])
        
        # 2. Update CVXPY Parameters
        self.x_init.value = x_curr
        self.x_ref.value = x_target
        
        # 3. Solve
        try:
            self.prob.solve(solver=cp.OSQP, warm_start=True)
            if self.prob.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                 print(f"MPC Warning: Solver status: {self.prob.status}")
        except Exception as e:
            print(f"MPC Error: {e}")
            
        # 4. Extract Control Inputs
        if self.u_var.value is None:
            # Fallback to hover if solver fails
            u_opt = np.array([self.GRAVITY * self.M, 0, 0, 0])
        else:
            u_opt = self.u_var[:, 0].value

        thrust = u_opt[0]
        torques = u_opt[1:]
        
        # 5. Mixer (Standard X Configuration for CF2X)
        t = thrust / 4.0
        r = torques[0] / (4.0 * self.L * self.KF)
        p = torques[1] / (4.0 * self.L * self.KF)
        y = torques[2] / (4.0 * self.KM)

        w0_sq = t - r - p - y
        w1_sq = t - r + p + y
        w2_sq = t + r + p - y
        w3_sq = t + r - p + y
        
        w_sq = np.array([w0_sq, w1_sq, w2_sq, w3_sq])
        w_sq = np.maximum(w_sq, 0)
        rpm = np.sqrt(w_sq / self.KF)
        
        return rpm, cur_pos, rpy