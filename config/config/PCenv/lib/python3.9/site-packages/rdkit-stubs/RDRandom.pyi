"""
 making random numbers consistent so we get good regressions

"""
from __future__ import annotations
import random as _random
from random.Random import randrange
from random.Random import seed
from random.Random import shuffle
import sys as sys
__all__ = ['random', 'randrange', 'seed', 'shuffle', 'sys']
def random(self):
    """
    random() -> x in the interval [0, 1).
    """
