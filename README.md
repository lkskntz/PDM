# PDM

Planning and Decision Making project

## Matplotlib simulations

To be able to run the MPC and RRT simulations of the matplotlib-version. It is assumed that Anaconda is available, as we will use a custom Anaconda environment to solve for all the dependencies.

### Linux/MacOS

1. In the root directory of the project files, that is the same directory as this README, there is a setup shell-script. This script provides the setup of the Anaconda environment. Simply run: `bash ./mpl_setup_env.sh`
2. After the environment setup has complete, you can activate the environment using: `conda activate mpc_drone_env`
3. With the environment activated, the simulations can be run using:

```bash
python ./Matplotlib/MPC_mpl_fix.py
python ./Matplotlib/MPC_mpl_mvt.py
python ./Matplotlib/MPC_mpl_crossbeams.py
```

4. When you are done with any simulation and you wish to remove the environment, run the following two commands:

```bash
conda deactivate
conda remove -n mpc_drone_env --all
```

### Windows

1. In the root directory of the project files, that is the same directory as this README, there is a setup batch-file. This script provides the setup of the Anaconda environment. Simply double-click `mpl_setup_env.bat` or open Anaconda Prompt, navigate to the directory of this README and the `mpl_setup_env.bat` and run: `mpl_setup_env.bat`
2. After the environment setup has complete, you can activate the environment using: `conda activate mpc_drone_env`
3. With the environment activated, the simulations can be run using:

```batch
python ./Matplotlib/MPC_mpl_fix.py
python ./Matplotlib/MPC_mpl_mvt.py
python ./Matplotlib/MPC_mpl_crossbeams.py
```

4. When you are done with any simulation and you wish to remove the environment, run the following two commands:

```batch
conda deactivate
conda remove -n mpc_drone_env --all
```
