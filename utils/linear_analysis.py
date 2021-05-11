#import R
import os
import re

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['R_HOME'] = 'C:\Program Files\R\R-4.0.3'

import pymer4
from pymer4.models import Lmer, Lm

from utils.utils import import_all

if __name__ == " main":
    import_all()
    print('worked')

    np.random.random(1)

import_all()
print('worked')

np.random.random(1)