# Contributor Guide

Thank you for your interest in contributing to this project! This guide will help you get started with the contribution process.

## Table of Contents
- [Getting Started](#getting-started)
- [Contributing Code](#contributing-code)
- [Reporting Issues](#reporting-issues)

## Getting Started

To contribute to this project, you will need to follow these steps:

1. Fork the repository.
2. Clone the forked repository to your local machine.
3. Create a new branch for your changes.
4. Make your changes and commit them.
5. Push your changes to your forked repository.
6. Open a pull request to the main repository.

If you are a collaborator, you can clone the main repository directly and create a new branch for your changes. 

## Contributing Code

#### When contributing code, please follow these guidelines:

- Write clean and readable code.
- Include appropriate comments and documentation.
- Follow the coding style and conventions used in the project.
- Make sure your changes do not break any existing functionality.
- If you are adding a new feature, make sure to include appropriate test and examples.
- If you are fixing a bug, make sure to include a test that reproduces the bug and is fixed by your changes.

#### Before opening a pull request, make sure that:

1. Add the new feature in the appropriate module in library. For new modulations create a new module with the name of modulations and their specific functions. 
2. If new functions are added, they must have their respective docstrings in [Google Style Format](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html). In module `opticomlib.devices` there are some examples too.
3. Include at least one example of usage of the new feature.
4. If new moddules are added they must have their title at the top of file in the following format:
    ```python
    """	
    ===========================================================
    Title of the module (:mod:`opticomlib.__new_module_name__`)
    ===========================================================

    .. autosummary::
    :toctree: generated/

        __new_function_name__               -- Description of the function.
        __new_function_name__               -- Description of the function.
        __new_function_name__               -- Description of the function.
    """
    ```
    and the following code must be added to `docs/source/modules.rst` file in order to be included in the documentation:
    ```rst
    .. automodule:: opticomlib.__new_module_name__
        :members:
        :noindex:
    ```
5. Images used in docstrings must be in the `docs/source/_images` folder and the following code must be added in the docstring to be included in the documentation:
    ```rst
    .. image:: /_images/_filename_.png
        :align: center 
    ``` 

## Reporting Issues

If you encounter any issues or have suggestions for improvements, please open an issue on the GitHub repository. When reporting issues, please provide as much detail as possible, including steps to reproduce the issue and any relevant error messages.
