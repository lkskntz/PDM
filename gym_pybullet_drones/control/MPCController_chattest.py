# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import cvxpy as cp
from gym_pybullet_drones.control.BaseController import BaseController

class MPCController(BaseController):
    def __init__(self, drone_model, g=9.81, m=0.03, horizon=10, dt=0.02):
        super().__init__(drone_model=drone_model)

        self.g = g
        self.m = m
        self.horizon = horizon
        self.dt = dt
        self.psi = 0.0

        # -------- Continuous-time linear model --------
        self.A = np.array([
            [0,0,0,0,0,0,1,0,0,0,0,0],
            [0,0,0,0,0,0,0,1,0,0,0,0],
            [0,0,0,0,0,0,0,0,1,0,0,0],
            [0,0,0,0,0,0,0,0,0,1,0,0],
            [0,0,0,0,0,0,0,0,0,0,1,0],
            [0,0,0,0,0,0,0,0,0,0,0,1],
            [0,0,0, self.g*np.sin(self.psi), self.g*np.cos(self.psi),0,0,0,0,0,0,0],
            [0,0,0,-self.g*np.cos(self.psi), self.g*np.sin(self.psi),0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0]
        ])

        self.B = np.array([
            [0,0,0,0],
            [0,0,0,0],
            [0,0,0,0],
            [0,0,0,0],
            [0,0,0,0],
            [0,0,0,0],
            [0,0,0,0],
            [0,0,0,0],
            [1/self.m,0,0,0],
            [0,1/1.4e-5,0,0],
            [0,0,1/1.4e-5,0],
            [0,0,0,1/2.17e-5]
        ])

        # -------- Discretization (Euler) --------
        self.Ad = np.eye(12) + self.A * self.dt
        self.Bd = self.B * self.dt

        # -------- MPC weights --------
        self.Q = np.diag([10,10,20,   5,5,5,   1,1,1,   0.1,0.1,0.1])
        self.R = 0.01 * np.eye(4)

    # ------------------------------------------------
    def computeControl(self, control_timestep, state, target):
        """
        state  = [pos(3), quat(4), rpy(3), vel(3), omega(3)]
        target = [x, y, z, yaw]
        """

        pos = state[0:3]
        eul = state[7:10]
        vel = state[10:13]
        omega = state[13:16]

        x0 = np.hstack([pos, eul, vel, omega])

        x_ref = np.hstack([
            target[0:3],           # position
            [0, 0, target[3]],     # roll, pitch, yaw
            np.zeros(3),           # linear velocity
            np.zeros(3)            # angular velocity
        ])

        # Hover input
        u_hover = np.array([self.m * self.g, 0, 0, 0])

        nx = 12
        nu = 4

        x = cp.Variable((nx, self.horizon + 1))
        u = cp.Variable((nu, self.horizon))

        cost = 0
        constraints = [x[:, 0] == x0]

        for k in range(self.horizon):
            cost += cp.quad_form(x[:, k] - x_ref, self.Q)
            cost += cp.quad_form(u[:, k] - u_hover, self.R)

            constraints += [
                x[:, k+1] == self.Ad @ x[:, k] + self.Bd @ u[:, k],
                u[0, k] >= 0.0,          # thrust ≥ 0
                u[0, k] <= 2.0 * self.m * self.g
            ]

        cost += cp.quad_form(x[:, self.horizon] - x_ref, self.Q)

        prob = cp.Problem(cp.Minimize(cost), constraints)
        prob.solve(solver=cp.OSQP, warm_start=True)

        if prob.status != cp.OPTIMAL:
            return self._thrustToRPM(u_hover)

        u_cmd = u[:, 0].value
        rpm = self._thrustToRPM(u_cmd)

        return rpm
