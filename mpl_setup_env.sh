#!/bin/bash

# Define environment name
ENV_NAME="mpc_drone_env"

echo "Checking for Conda..."
if ! command -v conda &> /dev/null
then
    echo "Conda could not be found. Please install Anaconda or Miniconda first."
    exit
fi

echo "Creating Conda environment: $ENV_NAME"
# Create environment and install packages in one command
conda create -n $ENV_NAME -c conda-forge python=3.10 casadi numpy matplotlib ffmpeg -y

echo "------------------------------------------------"
echo "Setup Complete!"
echo "To activate the environment, use:"
echo "conda activate $ENV_NAME"
echo "Then the following matplotlib simulations can be run: "
echo "python ./Matplotlib/MPC_mpl_fix.py"
echo "python ./Matplotlib/MPC_mpl_mvt.py"
echo "python ./Matplotlib/MPC_mpl_crossbeams.py"
echo "------------------------------------------------"