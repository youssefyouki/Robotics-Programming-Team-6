import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'milestone_03'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='youssefyouki',
    maintainer_email='youssef.hazem9123@gmail.com',
    description='Milestone 03: velocity/acceleration kinematics nodes and simulator validation',
    license='MIT',
    entry_points={
        'console_scripts': [
            'sim_node                  = milestone_03.sim_node:main',
            'velocity_kinematics_node  = milestone_03.velocity_kinematics_node:main',
            'inverse_kinematics_node   = milestone_03.inverse_kinematics_node:main',
            'trajectory_node           = milestone_03.trajectory_node:main',
            'validation_node           = milestone_03.validation_node:main',
        ],
    },
)
