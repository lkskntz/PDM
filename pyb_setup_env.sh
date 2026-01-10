#!/bin/bash
#Script installs gym-pybullet-drones and acados solver (Casadi an python env TBA)
#gym-pybullet-drones install guide: https://github.com/utiasDSL/gym-pybullet-drones
#Acados install guide: https://docs.acados.org/installation/index.html
#CasADi install guide: https://web.casadi.org/get/

# Setup conda environment
ENV_NAME="mpc_pybullet"
PROJECT_ROOT=$(pwd) # get path of current directory

echo "Checking for Conda..." # Makes sure that Anaconda or Miniconda is available on system
if ! command -v conda &> /dev/null
then
    echo "Conda could not be found. Please install Anaconda or Miniconda first."
    exit
fi

echo "Creating Conda environment: $ENV_NAME ..."
# Create environment and install packages in one command
conda create -n "$ENV_NAME" -c conda-forge python=3.10 casadi numpy scipy pybullet tqdm gymnasium matplotlib -y

eval "$(conda shell.bash hook)" # init conda so that conda activate can be used
conda activate "$ENV_NAME"


# Install Acados
echo "Installing Acados..."
if [ ! -d "$PROJECT_ROOT/acados" ]; then
    git clone https://github.com/acados/acados.git "$PROJECT_ROOT/acados"
fi
cd "$PROJECT_ROOT/acados"
git submodule update --recursive --init
mkdir -p build
cd build
cmake -DBUILD_SHARED_LIBS=ON -DACADOS_WITH_PYTHON=ON ..
make -j4
make install
cd $PROJECT_ROOT


# Install Acados python interface and gym_pybullet_drones
echo "Installing acados_template python interface..."
pip install -e "$PROJECT_ROOT/acados/interfaces/acados_template"
pip install -e "$PROJECT_ROOT/gym_pybullet_drones"
#pushd gym_pybullet_drones > /dev/null
#pip install -e .
#popd > /dev/null


# Configuration environment
echo "Configuring environment variables..."
conda env config vars set ACADOS_SOURCE_DIR="$PROJECT_ROOT/acados" -n "$ENV_NAME"
conda env config vars set LD_LIBRARY_PATH="$PROJECT_ROOT/acados/lib" -n "$ENV_NAME"
conda deactivate

echo "------------------------------------------------"
echo "Setup Complete!"
echo "To activate the environment, use:"
echo "conda activate $ENV_NAME"
echo "------------------------------------------------"