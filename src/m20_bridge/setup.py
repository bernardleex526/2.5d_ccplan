from setuptools import setup

package_name = 'm20_bridge'

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
    description='M20 hill patrol protocol encoding and decoding.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': ['mock_m20_server = m20_bridge.mock_m20_server:main'],
    },
)
