import os
from glob import glob
from setuptools import setup

package_name = 'ur3_llm_control'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Nguyen Van An',
    maintainer_email='student@todo.com',
    description='LLM-based task execution for UR3 in MoveIt 2',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'skill_executor = ur3_llm_control.skill_executor:main',
            'spawn_scene = ur3_llm_control.spawn_scene:main',
        ],
    },
)