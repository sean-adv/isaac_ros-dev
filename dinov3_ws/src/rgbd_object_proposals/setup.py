from glob import glob
from setuptools import find_packages, setup

setup(
    name='rgbd_object_proposals', version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/rgbd_object_proposals']),
        ('share/rgbd_object_proposals', ['package.xml', 'README.md']),
        ('share/rgbd_object_proposals/config', glob('config/*.yaml')),
        ('share/rgbd_object_proposals/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools', 'numpy'],
    maintainer='Workspace maintainers', maintainer_email='admin@example.com',
    description='Aligned RGB-D geometric object proposals for crop-based recognition',
    license='Apache-2.0',
    entry_points={'console_scripts': ['rgbd_object_proposal_node = rgbd_object_proposals.node:main']},
)
