@echo off
SET ENV_NAME=mpc_drone_env

echo Checking for Conda...
call conda --version
if %errorlevel% neq 0 (
    echo Conda is not installed or not in your PATH. 
    echo Please run this from the Anaconda Prompt.
    pause
    exit /b
)

echo Creating Conda environment: %ENV_NAME%
:: Create environment and install packages
call conda create -n %ENV_NAME% -c conda-forge python=3.10 casadi numpy matplotlib ffmpeg -y

echo ------------------------------------------------
echo Setup Complete!
echo To activate the environment, use:
echo conda activate %ENV_NAME%
echo Then the following matplotlib simulations can be run:
echo python Matplotlib\MPC_mpl_fix.py
echo python Matplotlib\MPC_mpl_mvt.py
echo python Matplotlib\MPC_mpl_crossbeams.py
echo ------------------------------------------------
pause