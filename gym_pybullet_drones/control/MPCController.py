import numpy as np
from gym_pybullet_drones.control.BaseController import BaseController

class MPCController(BaseController):
    def __init__(self, drone_model, g=9.81, m=0.03, horizon=10, dt=0.02):
        super().__init__(drone_model=drone_model)
        self.g = g
        self.m = m
        self.horizon = horizon
        self.dt = dt
        self.psi = 0
        """
        In gym_pybullet_drones: u = [T, tau_x, tau_y, tau_z] -> u has 4 Dimensions
        x = [p_x, p_y, p_z, phi, theta, psi, v_x, v_y, v_z, w_x, w_y, w_z] -> 12 Dimensions -> A : 12x12, B : 12x4, Q : 12x12, R : 4x?
        x -->> phi -> w_x
        y -->> theta -> w_y
        z -->> psi -> w_z
        """
        self.A = np.array([[0,0,0,0,0,0,1,0,0,0,0,0],
                           [0,0,0,0,0,0,0,1,0,0,0,0],
                           [0,0,0,0,0,0,0,0,1,0,0,0],
                           [0,0,0,0,0,0,0,0,0,1,0,0],
                           [0,0,0,0,0,0,0,0,0,0,1,0],
                           [0,0,0,0,0,0,0,0,0,0,0,1],
                           [0,0,0,g*np.sin(self.psi),g*np.cos(self.psi),0,0,0,0,0,0,0],
                           [0,0,0,-1*g*np.cos(self.psi),g*np.sin(self.psi),0,0,0,0,0,0,0],
                           [0,0,0,0,0,0,0,0,-g,0,0,0],
                           [0,0,0,0,0,0,0,0,0,0,0,0],
                           [0,0,0,0,0,0,0,0,0,0,0,0],
                           [0,0,0,0,0,0,0,0,0,0,0,0]])

        self.B = np.array([[0,0,0,0],
                           [0,0,0,0],
                           [0,0,0,0],
                           [0,0,0,0],
                           [0,0,0,0],
                           [0,0,0,0],
                           [0,0,0,0],
                           [0,0,0,0],
                           [1/m,0,0,0],
                           [0,1/1.4e-5,0,0],
                           [0,0,1/1.4e-5,0],
                           [0,0,0,1/2.17e-5]])

        self.Q = np.eye(12)

        self.R = 0.1 * np.eye(4)


        def computeControl(self, control_timestep, state, target):
            pos = state[0:3]
            vel = state[10:13]
            attitude = state[7:10]
            omega = state[13:16]
            x = np.hstack([pos, eul, vel, omega])

            x_ref = np.hstack([target[0:3],                     # target position
                               np.array([0, 0, target[3]]),     # keep roll=pitch=0, set target yaw
                               np.zeros(3),                     # target velocities 0
                               np.zeros(3)])                    # target angular rates 0

            T_hover = self.m * self.g * 0.25                    # Thrust per rotor (equidistand rotors)
            u_hover = np.array([T_hover, 0, 0, 0])              # u for static Hovering

            nx = self.A.shape[0]
            nu = self.B.shaper[1]
            x_var = []
            u_var = []
            for k in range(self.horizon + 1):
                x_var += [cp.Variable(nx)]
            for k in range(self.horizon):
                u_var += [cp.Variable(nu)]

            constraints = [x_var[0] == x]   # initial values
            cost = 0
            """
            MPC

            MPC

            MPC
            """

            u = #TBD

            rpm = self._thrustToRPM(u)
            return rpm
