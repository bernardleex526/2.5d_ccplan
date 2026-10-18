from setuptools import setup

package_name = 'hill_patrol_testing'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'PyYAML'],
    zip_safe=True,
    maintainer='bernardx526',
    maintainer_email='bernardx526@163.com',
    description='Threshold reading and metric reporting for hill patrol tests.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={'console_scripts': [
        'global_elevation_builder = hill_patrol_testing.global_elevation_builder:main',
    ]},
)
