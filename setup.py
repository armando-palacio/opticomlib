# FROM LOCAL FOLDER
# python setup.py bdist_wheel 
# pip install .

# FROM GITHUB
# pip install [--upgrade] git+https://github.com/armando-palacio/opticomlib.git


from setuptools import setup

def get_version():
    """
    Find the value assigned to __version__ in opticomlib/__init__.py.

    This function assumes that there is a line of the form

        __version__ = "version-string"

    in that file.  It returns the string version-string, or None if such a
    line is not found.
    """
    with open("opticomlib/__init__.py", "r") as f:
        for line in f:
            s = [w.strip() for w in line.split("=", 1)]
            if len(s) == 2 and s[0] == "__version__":
                return s[1][1:-1]
            
def get_long_description():
    """
    Return the README.md file content skipping the first two lines.
    """
    with open('README.md', encoding='utf-8') as f:
        content = f.read()
        return content[content.find("\n\n") + 2:]

def get_requirements():
    """
    Return the content of the requirements.txt file as a list of strings.
    """
    with open('requirements.txt', encoding='utf-8') as f:
        return f.read().split()

DISTNAME = "opticomlib"
DESCRIPTION = "Python package for optical communication systems."
LONG_DESCRIPTION = get_long_description()
MAINTAINER = "Armando P. Romeu"
MAINTAINER_EMAIL = "armandopr3009@gmail.com"
URL = "https://github.com/armando-palacio/opticomlib.git"
LICENSE = "MIT"
VERSION = get_version()
REQUIREMENTS = get_requirements()

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
    python_requires='>3.8',
    include_package_data=True,
)