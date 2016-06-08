# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 15:34:48 2016

@author: OHNS
"""

import numpy as np
from ctypes import *
import matplotlib.pyplot as plt
import re

#numpy pointer to float array
fPointer = ndpointer(c_float, flags = "C_CONTIGUOUS")

#pointer to numpy unsigned short array
uSPointer = ndpointer(c_ushort, flags = "C_CONTIGUOUS") 

#load the DLL libary
gpuLib = cdll.LoadLibrary("..\dll\gpuOCT.dll")


#define function for setting up gpu
#called to setup parameters of the GPU, only call once prior to scan start 
setup = gpuLib.setupGPU

#set argument types for dll function
setup.argtypes = [fPointer, fPointer,c_int,c_int,c_int,c_int,c_int]