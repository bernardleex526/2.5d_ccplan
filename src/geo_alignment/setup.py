from setuptools import setup

package_name = 'geo_alignment'

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
    description='Map <-> WGS84 alignment (plan T24).',
    license='Apache-2.0',
    entry_points={'console_scripts': ['fit_alignment = geo_alignment.align:main']},
)
