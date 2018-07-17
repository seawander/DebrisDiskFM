#!/usr/bin/env python
from setuptools import setup


setup(
    name="DebrisDiskFM",
    version="1.0",
    author="Bin Ren",
    author_email="bin.ren@jhu.edu",
    url="https://github.com/seawander/DebrisDiskFM",
    py_modules=["DebrisDiskFM"],
    description="Forward modeling for debris disks in scattered light",
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
    install_requires=['numpy', 'scipy', 'astropy']
)
