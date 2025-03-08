"""Setup script to manage the dependencies of the package."""

from setuptools import setup, find_packages

setup(
    name="mri_longitudinal_analysis",
    version="0.3",
    packages=find_packages(),
    install_requires=[
        'torch>=2.1.2',
        'SimpleITK>=2.3.1',
        # Add other dependencies from requirements.txt here
    ],
    dependency_links=[
        'git+https://github.com/MIC-DKFZ/HD-BET.git#egg=HD_BET'
    ]
)
