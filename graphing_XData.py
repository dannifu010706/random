#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 14:15:39 2020

@author: alexmcentarffer
"""

import numpy as np
import matplotlib.pyplot as plt
import os

path_to_files = "/Users/alexmcentarffer/Data/X_angle_dependent"

file_paths = os.listdir(path_to_files)

for file_path in file_paths:
    with open("/Users/alexmcentarffer/Data/X_angle_dependent/" + str(file_path), "r") as file:
        lines = file.readlines()
        lines = np.asarray(lines)
        arr = []
        for j in range(25):
            arr.append([float(x) for x in lines[j].split(' ')])
        arr = np.array(arr)
        plt.plot(arr[:,0],arr[:,1],'bo')
        plt.savefig(str(file_path) + '.png')
        plt.show()
        
        