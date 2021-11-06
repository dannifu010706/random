#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 15:47:04 2020

@author: alexmcentarffer
"""

import numpy as np
import matplotlib.pyplot as plt
import os

path_to_files = "/Users/alexmcentarffer/Data/fTAR_angle_dependent"

file_paths = os.listdir(path_to_files)

for file_path in file_paths:
    with open("/Users/alexmcentarffer/data/fTAR_angle_dependent/" + str(file_path), "r") as file:
        lines = file.readlines()
        lines = np.asarray(lines,dtype='float64')
        x = np.arange(0,len(lines),1)
        plt.plot(x,lines)
        plt.xlim(0,150)
        plt.savefig(str(file_path) + '.png')
        plt.show()
        