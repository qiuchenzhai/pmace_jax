from setuptools import setup, find_packages, Extension

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
#       packages=find_packages(include=['pmace_jax']),
      packages=find_packages(),
#       install_requires=['numpy==1.22.*', 'matplotlib>=3.5', 'scipy==1.8.0', 'pandas==1.4.2',
#                         'tifffile==2022.5.4', 'PyYAML==6.0', 'imagecodecs==2022.2.22', 'h5py==3.7.0'],
     )