from setuptools import setup, find_packages
from glob import glob

with open("README.md", "r") as fh:
    long_description = fh.read()
    
setup(
    name="tools_wcst_lfp",
    version="0.0.1",
    author="John Ferré",
    author_email="jbferre@uw.edu",
    description="Useful tools that pertain to wcst lfp data",
    long_description=long_description,
    url="jdfksdjfl",
    packages=find_packages(),
    package_data={'tools_wcst_lfp': glob('data/*')},
    classifiers=[
        "Programming language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating system :: OS Independent",
        "Intended audience :: Science/Research",
        "Topic :: Scientific/Engineering"
    ],
    python_requires='>=3.8',
    install_requires=["numpy", "pandas", "collections", "json", "h5py", "time", "os", "s3fs", "configparser"]
)