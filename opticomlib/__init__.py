from .typing import (
    gv, 
    electrical_signal, 
    optical_signal, 
    binary_sequence,
    eye,
)

from .utils import (
    dec2bin,
    str2array,
    get_time,
    tic,
    toc,
    db,
    dbm,
    idb,
    idbm,
    gaus,
    Q,
)

import os 

if os.path.isfile('../VERSION.txt'):
    __version__ = open('../VERSION.txt').read()
else:
    __version__ = None
del os
