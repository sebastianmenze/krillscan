import pathlib
from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "readme.md").read_text()

# This call to setup() does all the work
setup(
    name="krillscan",
    version="0.2.9",
    description="A python module for automatic analysis of backscatter data from vessels of opportunity",
    long_description=README,
    long_description_content_type="text/markdown",
    url="",
    author="Sebastian Menze",
    author_email="sebastian.menze@gmail.com",
    classifiers=[
        "Programming Language :: Python :: 3.9",
    ],
    packages=find_packages(),
    include_package_data=True,
    install_requires=["lxml","numpy","matplotlib","pandas",
    "future","pyproj","scikit_image","scipy","toml","geopy",
    "tables","xarray","netcdf4"])
