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

### When contributing code, please follow these guidelines

- Write clean and readable code.
- Include appropriate comments and documentation.
- Follow the coding style and conventions used in the project.
- Make sure your changes do not break any existing functionality.
- If you are adding a new feature, make sure to include appropriate test and examples.
- If you are fixing a bug, make sure to include a test that reproduces the bug and is fixed by your changes.

### Before opening a pull request, make sure that

1. Add the new feature in the appropriate module in library. For new modulations create a new module with the name of modulations and their specific functions.
2. If new functions are added, they must have their respective docstrings in [Numpy Style Format](https://numpydoc.readthedocs.io/en/latest/format.html). Example is provided below.

   ```python
    def function_name(arg1, arg2, *args, **kwargs):
        """ A short description of the function.

        A longer description of the function. Here you can describe the 
        purpose of the function, not the implementation details. 
        Implementation details must be described in Notes section.
        Make sure you follow the order of the sections below.

        Parameters
        ----------
        arg1 : :obj:`type`
            Description of the argument. Include allways :obj:`` for the 
            type of the argument, in order to correctly generate the documentation then.
        arg2 : :obj:`type`
            Description of the argument.

        \*args : :obj:`iterable`
            Include a backslash before the asterisk in order to correctly generate the documentation.
        \*\*kwargs : :obj:`dict`
            You can also include a list of keyword arguments like this.

            - ``p`` use double backticks to include the name of the argument.
            - ``q`` use double backticks to include the name of the argument.
        
        Returns
        -------
        out1 : :obj:`type`
            Description of the return value.
        out2 : :obj:`type`
            Description of the return value.
        
        :obj:`type`
            You can write only the type of the return.

        Raises
        ------
        ValueError
            Include a list of exceptions that the function can raise.

        Notes
        -----
        Here you can include implementation details of the function. You can 
        also include references like [1]_. You can include inline math
        expressions like :math:`x^2 + y^2 = z^2` or block math expressions
        like:
        
        .. math::
            x^2 + y^2 = z^2

        Can incluye an image like this:

        .. image:: /_images/_filename_.png
            :align: center
            :caption: Description of the image.

        For which you must first include the image in the directory `docs/source/_images`.

        References
        ----------
        .. [1] Reference to a paper, book, website, etc.

        Examples
        --------
        >>> function_name(1, 2)
        3

        You can add a plot like this:

        .. plot::

            import matplotlib.pyplot as plt
            import numpy as np

            x = np.linspace(0, 2*np.pi, 100)
            y = function_name(x, 3*x)

            plt.plot(x, y)
            plt.show()
        """
        return arg1 + arg2
    ```

3. Include at least one example of usage of the new feature.
4. If new moddules are added they must include the autosummary directive with all implemented functions and a short description of each one in docstring on the top of the module in order to be included in the documentation. Example is provided below.

    ```python
    """
    .. autosummary::
    :toctree: generated/

        __new_function_name__               -- Description of the function.
        __new_function_name__               -- Description of the function.
        __new_function_name__               -- Description of the function.
    """
    ```

    then the following code must be included within `docs/source/modules.rst` file

    ```rst
    Title of the module
    -------------------

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
