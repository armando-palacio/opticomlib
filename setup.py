# FROM LOCAL FOLDER
# python setup.py bdist_wheel 
# pip install [--upgrade] .\dist\opticomlib-x.x-py3-none-any.whl

# FROM GITHUB
# pip install [--upgrade] git+https://github.com/armando-palacio/opticomlib.git


from setuptools import setup

setup(
    name='opticomlib',
    version='0.3',
    description='Python package for optical communication systems.',
    url='https://github.com/armando-palacio/opticomlib.git',
    author='Armando Palacio',
    author_email='armandopr3009@gmail.com',
    license='MIT',
    packages=['opticomlib'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
    zip_safe=False,
    install_requires=[
        'scipy',
        'numpy',
        'matplotlib',
        'sklearn',
        'tqdm',
        'pympler'
    ],
)