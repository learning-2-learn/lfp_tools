from setuptools import setup, find_packages
from glob import glob

with open("README.md", "r") as fh:
    long_description = fh.read()
    
setup(
    name="lfp_tools",
    version="0.0.1",
    author="John Ferré",
    author_email="jbferre@uw.edu",
    description="Useful tools that pertain to wcst lfp data",
    long_description=long_description,
    url="https://github.com/learning-2-learn/lfp_tools",
    packages=find_packages(),
    package_data={'lfp_tools': glob('lfp_tools/data/*')},
    include_package_data=True,
    classifiers=[
        "Programming language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating system :: OS Independent",
        "Intended audience :: Science/Research",
        "Topic :: Scientific/Engineering"
    ],
    python_requires='>=3.8',
    install_requires=["numpy>=1.8.1", "matplotlib", "pandas", "aiobotocore>=1.0.1", "h5py", "s3fs", "tqdm", "dask", "dask_gateway"]
)