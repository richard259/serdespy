""" setup.py - Distutils setup file for serdespy package
    Richard Barrie
    2021
"""

from setuptools import setup, find_packages
import serdespy

setup(
    name="serdespy",
    version=serdespy.__version__,
    packages=find_packages(),
    include_package_data=True,
    license="BSD",
    description="Python Library for SerDes system simulation",
    long_description=open("README.md").read(),
    author="Richard Barrie",
    author_email="richard.barrie@mail.utoronto.ca",
    install_requires=[
        "numpy",
        "scikit-rf",
        "scipy",
    ],
    keywords=["serdes", "communication", "simulator"],
    classifiers=[
        "Environment :: Console",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Telecommunications Industry",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering",
    ],
)