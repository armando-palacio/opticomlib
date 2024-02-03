# FROM LOCAL FOLDER
# python setup.py bdist_wheel 
# pip install .

# FROM GITHUB
# pip install [--upgrade] git+https://github.com/armando-palacio/opticomlib.git


from setuptools import setup

DISTNAME = "OptiComLib"
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
        'scipy',
        'numpy',
        'matplotlib',
        'scikit-learn',
        'tqdm',
        'pympler',
        'sphinx-rtd-theme>=1.2.2',
    ],
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    python_requires='<=3.10',
)