#!/bin/bash
#Script installs gym-pybullet-drones and acados solver (Casadi an python env TBA)
#gym-pybullet-drones install guide: https://github.com/utiasDSL/gym-pybullet-drones
#Acados install guide: https://docs.acados.org/installation/index.html
#CasADi install guide: https://web.casadi.org/get/

#Pybullet
git clone https://github.com/utiasDSL/gym-pybullet-drones.git

#Acados
git clone https://github.com/acados/acados.git
cd acados
git submodule update --recursive --init
mkdir -p build
cd build
cmake -DBUILD_SHARED_LIBS=ON ..
make install -j4

cd ..
#Correct Exported variable necessary to run the scripts. ACADOS_SOURCE_DIR and LD_LIBRARY_PATH both required.
#export ACADOS_SOURCE_DIR = $(pwd)
#export LD_LIBRARY_PATH=$ACADOS_SOURCE_DIR/lib

#Conda / Python
conda install casadi
