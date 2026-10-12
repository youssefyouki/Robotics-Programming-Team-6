from setuptools import find_packages, setup

package_name = 'milestone_02'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='youssefyouki',
    maintainer_email='youssef.hazem9123@gmail.com',
    description='Milestone 02: FK, IK, and Joint Slider GUI for Koch v1.1',
    license='MIT',
    entry_points={
        'console_scripts': [
            'fk_node            = milestone_02.fk_node:main',
            'ik_node            = milestone_02.ik_node:main',
            'joint_slider_gui   = milestone_02.joint_slider_gui:main',
        ],
    },
)
