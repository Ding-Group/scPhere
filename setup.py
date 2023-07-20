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
    install_requires=['numpy >= 1.21.5',
                      'pytorch == 1.12.1',
                      'scipy >= 1.7.3',
                      'pandas >= 1.3.5',
                      'matplotlib >= 3.5.3',
                      'seaborn >= 0.12.2',
                      ],
    packages=find_packages(),
    python_requires='>=3.7.14',
)
