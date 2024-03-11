from .typing import *

from .utils import *

import os

with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'VERSION.txt')) as version_file:
    __version__ = version_file.read().strip()

del os