# FROM LOCAL FOLDER
# python setup.py bdist_wheel 
# pip install .

# FROM GITHUB
# pip install [--upgrade] git+https://github.com/armando-palacio/opticomlib.git


from setuptools import setup
import os

try:
    os.environ['REQUIREMENTS'] = open('requirements.txt').read()
    os.environ['VERSION'] = open('VERSION.txt').read()
except:
    pass

DISTNAME = "opticomlib"
DESCRIPTION = "Python package for optical communication systems."
LONG_DESCRIPTION = open('README.md', encoding='utf-8').read()
MAINTAINER = "Armando P. Romeu"
MAINTAINER_EMAIL = "armandopr3009@gmail.com"
URL = "https://github.com/armando-palacio/opticomlib.git"
LICENSE = "MIT"
VERSION = os.getenv('VERSION').strip()
REQUIREMENTS = os.getenv('REQUIREMENTS').split()

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
    install_requires=REQUIREMENTS,
    long_description=LONG_DESCRIPTION[LONG_DESCRIPTION.find("#"):],
    long_description_content_type="text/markdown",
    python_requires='<=3.10',
)