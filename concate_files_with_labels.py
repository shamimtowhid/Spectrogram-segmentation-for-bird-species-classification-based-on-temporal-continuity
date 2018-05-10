import pandas as pd
import os
src_files = os.listdir('./shorten_final_features')
df1=pd.read_csv('2.csv',header=None)

for file_name in src_files:
    full_file_name = os.path.join('./shorten_final_features/', file_name)
    if (os.path.isfile(full_file_name)):
        df2=pd.read_csv(full_file_name,header=None)
        df1=df1.append(df2)

df1.to_csv('test.csv',header=False,index=False)