#!/usr/bin/env python3
#-*- coding: utf-8 -*-

from setuptools import setup, find_packages
import os

# Get the version from __init__.py
with open('regression/__init__.py') as fid:
    for line in fid:
        if line.startswith('__version__'):
            version = line.strip().split()[-1][1:-1]
            break

# Read README
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='regression',
    version=version,
    description='Python package to compute multiple linear regression',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Alexandre LALLE',
    author_email='alexandrelalle825@gmail.com',
    url='https://github.com/Alexandre-Lalle/linear_regression',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.19.0',
        'matplotlib>=3.3.0',
        'seaborn>=0.11.0',
        'matplotlib>=3.3.0',
    ],
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.0',
            'flake8>=3.9.0',
            'sphinx>=3.0',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering',
    ],
    python_requires='>=3.8',
    include_package_data=True
)
