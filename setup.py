#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name='scPhere',
    description="Deep generative model embedding single-cell RNA-Seq profiles on hyperspheres or hyperbolic spaces",
    version='0.1.0',
    author='Jiarui Ding and Aviv Regev',
    author_email='jding@broadinstitue.org',
    keywords="scPhere",
    license='BSD 3-clause',
    url="https://github.com/klarman-cell-observatory/scPhere",
    install_requires=['numpy >= 1.16.4',
                      'tensorflow == 1.14.0',
                      'scipy >= 1.3.0',
                      'pandas >= 0.21.0',
                      'matplotlib >= 3.1.0',
                      'tensorflow_probability == 0.7.0',
                      ],
    packages=find_packages(),
    python_requires='>=3.6',
)
