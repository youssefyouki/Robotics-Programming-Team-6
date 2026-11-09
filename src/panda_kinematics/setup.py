import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'panda_kinematics'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='youssef',
    maintainer_email='youssef.hazem75@yahoo.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'kinematics = panda_kinematics.kinematics:main',
            'trajectory_node = panda_kinematics.trajectory_node:main',
            'trajectory_experiment = panda_kinematics.trajectory_experiment:main',
            'tracking_logger = panda_kinematics.tracking_logger:main',
        ],
    },
)
