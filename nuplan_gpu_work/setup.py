import os

import setuptools

# Change directory to allow installation from anywhere
script_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_folder)

# Installs
setuptools.setup(
    name='nuplan_gpu_work',
    version='1.0.0',
    author='Pegah',
    author_email='khayatan.pegah@valeo.com',
    description='placing an obstacle in the simulation in front of the ego',
    url='',
    python_requires='>=3.9',
    packages=setuptools.find_packages(script_folder),
    package_data={'': ['*.json']},
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3.9',
        'Operating System :: OS Independent',
        'License :: Free for non-commercial use',
    ],
    license='apache-2.0',
)
