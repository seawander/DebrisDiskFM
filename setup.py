#!/usr/bin/env python
from setuptools import setup, find_packages
try:
    from distutils.config import ConfigParser
except ImportError:
    from configparser import ConfigParser

# Read configuration variables in from setup.cfg
conf = ConfigParser()
conf.read(['setup.cfg'])

# Get some config values
metadata = dict(conf.items('metadata'))
PACKAGENAME = metadata.get('package_name', 'packagename')
DESCRIPTION = metadata.get('description', '')
AUTHOR = metadata['author']
AUTHOR_EMAIL = metadata['author_email']
URL = metadata['url']
LICENSE = metadata['license']
VERSION = metadata['version']






setup(
    name=PACKAGENAME,
    version=VERSION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    license=LICENSE,
    url=URL,
    py_modules=["debrisdiskfm"],
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
    ],
    install_requires=['numpy', 'scipy', 'astropy'],
    packages=find_packages(),
    package_data={},
)
