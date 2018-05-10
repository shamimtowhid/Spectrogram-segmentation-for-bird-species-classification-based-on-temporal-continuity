#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 30 22:44:07 2017

@author: shamim
"""
import pandas as pd
from shutil import copyfile

for i in range(1,155):
    filename=str(i)+'.csv'
    df1=pd.read_csv('finalfeatures/'+filename,header=None)
    j=df1[0].count()
#    print(j)
    if(j<=8):
        copyfile('./finalfeatures/'+filename,'./shorten_final_features/'+filename)
    
    
    #copyfile('./wav/'+df.loc[j][0],'./newwav/'+df.loc[j][1])