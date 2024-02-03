# FROM LOCAL FOLDER
# python setup.py bdist_wheel 
# pip install .

# FROM GITHUB
# pip install [--upgrade] git+https://github.com/armando-palacio/opticomlib.git


from setuptools import setup

DISTNAME = "opticomlib"
DESCRIPTION = "Python package for optical communication systems."
LONG_DESCRIPTION = open("README.md", encoding="utf8").read()
MAINTAINER = "Armando P. Romeu"
MAINTAINER_EMAIL = "armandopr3009@gmail.com"
URL = "https://github.com/armando-palacio/opticomlib.git"
LICENSE = "MIT"
VERSION = "0.4.2"

setup(
    name=DISTNAME,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    description=DESCRIPTION,
    license=LICENSE,
    url=URL,
    version=VERSION,
    packages=['opticomlib'],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Telecommunications Industry",
        "Programming Language :: Python :: 3",
    ],
    install_requires=[
        'scipy>=1.12.0',
        'numpy>=1.26.3',
        'matplotlib>=3.8.2',
        'scikit-learn>=1.4.0',
        'tqdm>=4.66.1',
        'pympler>=1.0.1',
        'sphinx>=7.2.6',
        'renku>=2.9.1',
        'sphinx-rtd-theme>=1.3.0',
    ],
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    python_requires='<=3.10',
)