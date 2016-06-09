# -*- coding: utf-8 -*-
"""
Created on Wed Jun  8 17:04:57 2016

@author: OHNS
"""

import numpy as np
import pythonInterfaceTest as gpu
import math
import time
import matplotlib.pyplot as plt
print ("TEST")
fftSize = 4096
rawSize = 2048
cropLength = 2048
numAlines = 16383
gpu.setDigitizer(0)
startWn = 1/1.35
endWn = 1/1.25
k = np.zeros(rawSize, dtype = "float32")
z = np.zeros(20)
for i in range(2048):
    k[i] = startWn - (startWn - endWn * (i / 2048))

for i in range(20):
    z[i] = 200 + 50 * i
S = np.zeros(2048*20)
S = np.reshape(S, (20, 2048))
for i in range(20):
    for j in range(2048):
        S[i][j] = math.cos(2*2*math.pi*k[j]*z[i])
Smean = np.zeros(2048)
Smean = np.mean(S, 0)
plt.plot(Smean)

uShort = np.dtype('u2')
disp = np.ones(rawSize, dtype = "float32") * 65536.

dataV = np.ones(rawSize, dtype = bool)
window = np.ones(rawSize, dtype = "float32")

dataIn = np.ones(numAlines * rawSize, dtype  = uShort)
dataOut = np.zeros(numAlines*cropLength, dtype = "float32")

error = gpu.setup(disp,window,rawSize,fftSize,numAlines,0,cropLength,False)

print("Setup Error: ", error)
t1 = time.time()
#error = gpu.pingError()
#print("Pinged Error: ", error)
image = gpu.gpuOCTMagReshape(dataIn, dataOut,cropLength,numAlines)
#error = gpu.processAllData(dataIn, dataOut)
#print("Process Error: ", error)
#dataOut = np.reshape(dataOut,(numAlines, cropLength))
t2 = time.time()
print("Time: ", t2-t1)
error = gpu.clearGPU()
error = gpu.endGPU()
print("End Error: ", error)
print(image[0][0], image[(numAlines-1)][500], image[0][2])
