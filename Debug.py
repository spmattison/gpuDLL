# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 17:48:47 2016

@author: OHNS
"""

import numpy as np
import pythonInterfaceTest as gpu
import math
import time
import matplotlib.pyplot as plt

gpu.MonsterMash(10)
vlad = np.zeros(10, dtype = "float32")
gpu.GraveyardSmash(vlad, 10)
print(vlad[2])