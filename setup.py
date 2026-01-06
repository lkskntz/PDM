from setuptools import setup, find_packages

setup(
    name='gym_pybullet_drones',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pybullet',
        'gym',
        'scipy',
        'cvxpy',
        'osqp',
        'matplotlib'
    ]
)