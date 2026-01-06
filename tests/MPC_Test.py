import sys
import os
# Add the parent directory to the search path to find local gym_pybullet_drones modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
from gym_pybullet_drones.utils.utils import sync
from gym_pybullet_drones.utils.enums import DroneModel, Physics

# Import custom env and new controller
from gym_pybullet_drones.custom_env import ObstacleHoverEnv
from MPCControl import MPCControl

def test_mpc():
    # --- Config ---
    drone_model = DroneModel.CF2X
    
    # --- Environment ---
    # Instantiate your custom environment
    env = ObstacleHoverEnv(drone_model=drone_model, gui=True)
    
    # !!! CRITICAL !!!
    # ObstacleHoverEnv is an RL env that normally expects normalized actions [-1, 1].
    # We are sending raw RPMs from MPC. We must override the preprocessing to allow this.
    env._preprocessAction = lambda x: x 
    
    # --- Controller ---
    ctrl_freq = 50
    ctrl_dt = 1/ctrl_freq
    mpc = MPCControl(drone_model=drone_model, dt=ctrl_dt, horizon=15)

    # --- Target Handling (The Fix) ---
    # We ensure target_point is shape (3,)
    if hasattr(env, 'TARGET_POS') and env.TARGET_POS is not None:
        if env.TARGET_POS.ndim == 1:
            target_point = env.TARGET_POS  # It is already [x, y, z]
        else:
            target_point = env.TARGET_POS[0] # It is [[x, y, z], ...]
    else:
        target_point = np.array([0.0, 0.0, 1.0])

    # Safety check
    if np.isscalar(target_point) or len(target_point) != 3:
        print(f"[WARNING] Invalid target shape: {target_point}. Defaulting to [0,0,1]")
        target_point = np.array([0.0, 0.0, 1.0])

    print(f"[INFO] MPC Target Point: {target_point}")

    # --- Simulation Loop ---
    start = time.time()
    
    # BaseAviary usually exposes .PYB_FREQ (default 240Hz)
    pyb_freq = getattr(env, 'PYB_FREQ', 240) 
    steps = int(10 * pyb_freq) # Run for 10 seconds
    ctrl_step_ratio = int(pyb_freq / ctrl_freq)

    obs, info = env.reset()
    
    # Initialize action dictionary
    action = {'0': np.array([0, 0, 0, 0])}

    for i in range(steps):
        
        # 1. Compute Control at specific frequency
        if i % ctrl_step_ratio == 0:
            
            # Access exact state from the environment directly for the MPC
            p_curr = env.pos[0]
            q_curr = env.quat[0]
            v_curr = env.vel[0]
            w_curr = env.ang_v[0]

            rpm, _, _ = mpc.computeControl(cur_pos=p_curr,
                                           cur_quat=q_curr,
                                           cur_vel=v_curr,
                                           cur_ang_vel=w_curr,
                                           target_pos=target_point)
            
            action['0'] = rpm

        # 2. Step the simulation
        # Using raw RPMs because we patched _preprocessAction
        ret = env.step(action)
        
        # 3. Sync visualization
        sync(i, start, 1/pyb_freq)

    env.close()

if __name__ == "__main__":
    test_mpc()