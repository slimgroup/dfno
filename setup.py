from setuptools import setup, find_packages

requirements = ['distdl', 'mat73', 'matplotlib', 'mpi4py', 'numpy', 'scipy', 'torch==1.9']

setup(
    name = 'dfno',
    version = '0.1',
    author = 'Thomas Grady',
    author_email = 'tgrady6@gatech.edu',
    license = 'MIT',
    install_requires=requirements,
    packages = find_packages(),
    description = 'Distributed Fourier Neural Operator based on DistDL')