from setuptools import setup, find_packages
import os, re, ast

setup(
    name='ir_analysis',
    version='1.0',
    description='Tools for analysing MAST-U IR data',
    url='git@git.ccfe.ac.uk/tfarley/ir_analysis',
    author='tfarley',
    author_email='tom.farley@ukaea.uk',
    # license=ccfepyutils.__license__,
    packages=['ir_analysis'],
    # packages=find_packages(exclude=['docs', 'external', 'misc', 'tests', 'third_party']),
    # package_data={},
    # include_package_data=True,
    install_requires=[
        "numpy >= 1.12.0",
        "scipy",
        "xarray",
        "pandas",
        "matplotlib",
        "opencv-python",
        ],
    python_requires='>=3',
    setup_requires=['pytest-runner'],
    # test_suite='tests.test_suite_fast',  # 'tests.test_suite_slow'
    tests_require=['pytest-cov'],
    zip_safe=False,
long_description=open('README.md').read()
)