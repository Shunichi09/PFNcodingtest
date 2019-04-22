from setuptools import find_packages
from setuptools import setup

install_requires = ['numpy', 'matplotlib']
tests_require = ['pytest>=3.2.0', 'mock']
setup_requires = ["pytest-runner"]

setup(
    name='pfn_coding_test',
    version='0.0.1',
    description='PFN internship coding test',
    author='Shunichi Sekiguchi',
    author_email='quick1st97@keio.jp',
    install_requires=install_requires,
    license='MIT License',
    packages=find_packages(exclude=('tests')),
    setup_requires=setup_requires,
    test_suite='tests',
    tests_require=tests_require
)