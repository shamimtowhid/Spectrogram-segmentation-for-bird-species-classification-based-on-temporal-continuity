# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd

df=pd.read_csv('test.csv',header=None)



df1=pd.concat([df]*4, ignore_index=True)

df1.to_csv('test_copy.csv',header=False,index=False) 