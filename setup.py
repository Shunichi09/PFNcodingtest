from setuptools import find_packages
from setuptools import setup

install_requires = ['numpy', 'matplotlib']

setup(
    name='pfn_coding_test',
    version='0.0.1',
    description='PFN internship coding test',
    author='Shunichi Sekiguchi',
    author_email='quick1st97@keio.jp',
    install_requires=install_requires,
    license='MIT License',
    packages=find_packages(exclude=('tests')),
    test_suite='tests'
)