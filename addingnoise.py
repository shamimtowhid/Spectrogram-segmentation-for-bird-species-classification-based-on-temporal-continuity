#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 23:14:41 2017

@author: shamim
"""

import numpy as np
import pandas as pd


gaussian_noise = pd.DataFrame(np.random.normal(0.0,0.3,(1424,20)))
gaussian_noise = 0.1 * gaussian_noise
gaussian_noise = gaussian_noise + 1

df = pd.read_csv("test_copy_without_labels.csv",header=None)
df = df * gaussian_noise
df.to_csv("after_adding_noise.csv",header=False,index=False)
print("done")

#print(gaussian_noise)
#print(df.head())

