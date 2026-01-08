"""
Run gym-pybullet-drones with a custom MPC controller following a circular trajectory
"""

import time
import numpy as np

from gym_pybullet_drones.envs.custom_env import ObstacleCtrlEnv
from gym_pybullet_drones.utils.enums import DroneModel, Physics
#from gym_pybullet_drones.control.MPCControl import MPCControl as Control
from gym_pybullet_drones.control.RRTMPC import RRTMPCController as Control

def main():

    #### Environment parameters ##############################################
    DRONE_MODEL = DroneModel.CF2X
    NUM_DRONES = 1
    PHYSICS = Physics.PYB
    SIM_FREQ = 240        # Hz (physics)
    CTRL_FREQ = 48        # Hz (controller)
    ctrl_dt = 1.0 / CTRL_FREQ
    GUI = True
    DURATION_SEC = 40

    #### Create environment ##################################################
    init_pos = np.array([[2.0, 0.0, 0.0]])  # starting at (0,0,1)
    env = ObstacleCtrlEnv(phase=3,
        drone_model=DRONE_MODEL,
        num_drones=NUM_DRONES,
        physics=PHYSICS,
        gui=GUI,
        initial_xyzs=init_pos,
        initial_rpys=np.zeros((1,3)),
    )

    bounds = [(-7.5, 7.5),  # x
              (-5.0, 5.0),  # y
              (0.0, 5.0)]    # z


    #### Create MPC controller ###############################################
    mpc = Control(
        drone_model=DRONE_MODEL,
        bounds=bounds
    )
    print("\nL:", getattr(mpc, "L", None))
    print("KF:", getattr(mpc, "KF", None))
    print("KM:", getattr(mpc, "KM", None))

    #### Reset environment ###################################################
    obs = env.reset()

    # --- Trajectory parameters ---
    print("[INFO] Starting MPC control loop")

    target_pos = np.array([[4.0,0.0,3.0]])

    # Compute motor RPMs using MPC
    #rpm = mpc.computeControl(
     #   state=obs[0],
      #  target_pos=target_pos
    #)

    # Step simulation
    #obs, reward, terminated, truncated, info = env.step(rpm)
    t = 0
    while t < DURATION_SEC:
        # Compute control
        rpm = mpc.computeControl(state=obs[0], target_pos=target_pos)

        # Step simulation
        obs, reward, terminated, truncated, info = env.step(rpm)

        # Slow down if GUI
        if GUI:
            time.sleep(ctrl_dt)

        # Check early termination
        if terminated or truncated:
            print("[INFO] Episode ended early, resetting")
            obs = env.reset()

        t += ctrl_dt
    # Slow down to real time if GUI is enabled
    if GUI:
        time.sleep(ctrl_dt)

    if terminated or truncated:
        print("[INFO] Episode ended early, resetting")
        obs = env.reset()

    #### Close environment ###################################################
    env.close()
    print("[INFO] Simulation finished")


if __name__ == "__main__":
    main()
