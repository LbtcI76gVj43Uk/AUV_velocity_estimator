import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'velocity_estimator'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))) # Pass arguments to launch
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='henri',
    maintainer_email='henri.kirschke@stud.th-owl.de',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'camera_node = velocity_estimator.camera_publisher:main',
            'estimator_node = velocity_estimator.estimator_pipeline:main',
            'logger_node = velocity_estimator.logger:main',
        ],
    },
)
