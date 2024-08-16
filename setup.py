from distutils.core import setup
from setuptools import find_packages
from setuptools.command.install import install
import os

setup(
    name='svh_layer',
    version='1.0.0',
    description='svh kinematics layer',
    author='Wei Wei',
    author_email='wei.wei2018@ia.ac.cn',
    url='wei.wei2018@ia.ac.cn',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'torch',
        'trimesh',
        'roma',
    ]
)