from setuptools import setup

package_name = 'traversability_maps'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='bernardx526',
    maintainer_email='bernardx526@163.com',
    description='Traversability GridMaps for Nav2 with stair zones made traversable.',
    license='Apache-2.0',
    entry_points={'console_scripts': ['traversability_maps = traversability_maps.node:main']},
)
