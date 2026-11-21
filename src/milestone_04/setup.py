import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'milestone_04'

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
    description='Milestone 04: trajectory planning, motor-level control and colour sorting',
    license='MIT',
    entry_points={
        'console_scripts': [
            'cell_sim_node   = milestone_04.cell_sim_node:main',
            'sorting_node    = milestone_04.sorting_node:main',
            'controller_node = milestone_04.controller_node:main',
            'evaluation_node = milestone_04.evaluation_node:main',
            'control_panel   = milestone_04.control_panel_node:main',
            'view_node       = milestone_04.view_node:main',
        ],
    },
)
