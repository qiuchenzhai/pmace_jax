from setuptools import setup, find_packages, Extension
import numpy as np
import os

NAME = "pmace_jax"
VERSION = "0.1"
DESCR = "An implementation of PMACE framework using JAX"
REQUIRES = ['numpy']
LICENSE = "BSD-3-Clause"

AUTHOR = 'pmace_jax development team'
EMAIL = "qzhai@purdue.edu"
PACKAGE_DIR = "pmace_jax"

setup(install_requires=REQUIRES,
      zip_safe=False,
      name=NAME,
      version=VERSION,
      description=DESCR,
      author=AUTHOR,
      author_email=EMAIL,
      license=LICENSE,
      packages=find_packages(include=['pmace_jax']),
      )

