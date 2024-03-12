# Contributing to Opticomlib

Thank you for your interest in contributing to this project! This guide will help you get started with the contribution process.

- [Contributing to Opticomlib](#contributing-to-opticomlib)
  - [Getting Started](#getting-started)
  - [Contributing Code](#contributing-code)
    - [When contributing code, please follow these guidelines](#when-contributing-code-please-follow-these-guidelines)
    - [Before opening a pull request, make sure that](#before-opening-a-pull-request-make-sure-that)
  - [Branches Architecture](#branches-architecture)
  - [Reporting Issues](#reporting-issues)

## Getting Started

To contribute to this project, you will need to follow these steps:

1. Fork the repository.
2. Clone the forked repository to your local machine (If you are a collaborator first step is not required).

    ```bash
    git clone https://github.com/armando-palacio/opticomlib.git
    cd opticomlib
    ```

3. Create a new branch for your changes.

    ```bash
    git checkout -b new-branch-name
    ```

4. Make your changes and commit them.

    ```bash
    git add .
    git commit -m "A short description of the changes made"
    ```

5. Push your changes to your forked (project) repository.

    ```bash
    git push origin new-branch-name
    ```

6. Open a pull request to the main repository.

    - Go to the main repository and click on the "New pull request" button.
    - Select your forked repository and the branch with your changes.
    - Add a title and description to your pull request.
    - Click on the "Create pull request" button.

## Contributing Code

### When contributing code, please follow these guidelines

- Write clean and readable code.
- Include appropriate comments and documentation.
- Follow the coding style and conventions used in the project.
- Make sure your changes do not break any existing functionality.
- If you are adding a new feature, make sure to include appropriate test and examples.
- If you are fixing a bug, make sure to include a test that reproduces the bug and is fixed by your changes.

### Before opening a pull request, make sure that

1. Add the new feature in the appropriate module in library. Be sure to include the new feature in autosummary directive (module doctring at the top of the file) in order to be included in the documentation.

2. For new modulation create a new module with the name of modulation and their specific functions. Be sure to include module docstring at the top of the file with following format:

    ```python
    """
    .. rubric:: Functions
    .. autosummary::

        func_1
        func_2
        .
        .
        .

    .. rubric:: Classes
    .. autosummary::

        class_1
        class_2
        .
        .
        .
    """
    ```

    If there are no classes, then the classes section can be omitted. If there are no functions, then the functions section can be omitted.

    Then, you have to create a new file `module_name.rst` within `docs/source/` with the name of new module and the follow code inside:

    ```rst
    Title of the module
    -------------------
    Use this module to (do that it does).

    .. code-block:: python

        >>> import opticomlib.module_name as abbreviation

    Some another description of the module if it is necessary.

    .. automodule:: opticomlib.module_name
        :members:
        :member-order: bysource
    ```

    Finally, open `docs/source/index.rst` and include the new module in the toctree directive. This step is required to include the new feature in the sphinx documentation.

3. If new functions are added, they must have their respective docstrings in [Numpy Style Format](https://numpydoc.readthedocs.io/en/latest/format.html). Example is provided below.

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

4. If new class is added it should have its respective docstring in the top of the class and for each attribute and method of it. It must be in [Numpy Style Format](https://numpydoc.readthedocs.io/en/latest/format.html). Example is provided below.

    ```python
    class ClassName:
        """ A short description of the class.

        A longer description of the class. In contrast to the functions
        you can include the implementation details of the class here.

        .. image:: _images/_filename_.png  # can include some images
            :align: center
            :width: 100%

        Also can include math equations. For example:

        .. math::
            x^2 + y^2 = z^2
        
        .. rubric:: Attributes
        .. autosummary::

            ~ClassName.attr_1
            ~ClassName.attr_2
            .
            .
            .

        .. rubric:: Methods
        .. autosummary::

            method_name_1
            method_name_2
            .
            .
            .
        """
        def __init__(self, arg1, arg2):
            """ A short description of the method.

            A longer description of init method.

            Parameters
            ----------
            arg1 : :obj:`type`
                Description of the argument. Include allways :obj:`` for the 
                type of the argument, in order to correctly generate the documentation then.
            arg2 : :obj:`type`
                Description of the argument.
            """
            self.arg1 = arg1
            """A description of attribute arg1"""
            self.arg2 = arg2
            """A description of attribute arg2"""
        
        def method_name(self, arg1, arg2):
            """ docstring like a function """       
    ```

5. Include at least one example of usage of the new feature.

6. Images used in docstrings must be in the `docs/source/_images` folder and the following code must be added in the docstring to be included in the documentation:

    ```rst
    .. image:: /_images/_filename_.png
        :align: center
        :width: 80%
    ```

## Branches Architecture

Let's take a closer look at using branches in a Python project on GitHub. Here's a detailed guide to structuring and working with branches in this project:

1. Main branch (`main`):

    This branch should only contain stable and tested code that is ready to be deployed into production.
    It is recommended that this branch be protected, to avoid direct changes and the need for revisions via pull requests.
    Development branches:

2. Development Branch (`dev`):

    This branch is used to integrate and test new features before merging them into the main branch. Feature branches may be created from this branch for the development of new features.

3. Feature branches (`feature/*`):

    Separate branches are created to develop specific features or solve specific problems.
    Each feature branch should be descriptive, representing the functionality being developed.
    When implementation is complete and the functionality has been tested, the feature branch is merged with the development branch.

4. Release branches (`release/tage_name`):

    Release branches are created when a new version of the software is being prepared for production deployment.
    These branches are used for final testing, last-minute bug fixes, and documentation preparation prior to deployment.
    Once testing is complete, the release branch is merged with both the main and development branches.

5. Hotfix branches (`hotfix/*`):

    Hotfix branches are created to address critical issues that require an immediate fix in production.
    These branches are derived directly from the main branch.
    Once the problem is fixed, the hotfix branch is merged with both the main branch and the development branch.

This branching strategy promotes an orderly and controlled workflow, facilitates collaboration between team members, and ensures code stability in production. Proper use of branches helps maintain a clear and organized change history in the repository.

## Reporting Issues

If you encounter any issues or have suggestions for improvements, please open an issue on the GitHub repository. When reporting issues, please provide as much detail as possible, including steps to reproduce the issue and any relevant error messages.
