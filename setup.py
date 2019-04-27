#!/usr/bin/env python
# -*- coding: utf-8 -*-

import io
import os
import sys
from shutil import rmtree

from setuptools import find_packages, setup, Command

# Package meta-data.
NAME = 'drosophVAE'
DESCRIPTION = 'Using VAE to build a latent space of the behavioural data of drosophila melanogaster flies'
URL = 'https://github.com/samuelsmal/drosophVAE'
EMAIL = 'samuel.edlervonbaussnern@epfl.ch'
AUTHOR = 'Samuel von Baußnern'
REQUIRED = []

here = os.path.abspath(os.path.dirname(__file__))

setup(
    name=NAME,
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    packages=find_packages(exclude=('tests',)),
    install_requires=REQUIRED,
    include_package_data=True,
    license='MIT',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy']
)
