from .typing import *

from .utils import *

import os 

if os.path.isfile('../VERSION.txt'):
    __version__ = open('../VERSION.txt').read()
else:
    __version__ = None
del os
