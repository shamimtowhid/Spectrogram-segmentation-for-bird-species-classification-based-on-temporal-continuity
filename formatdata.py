import pandas as pd


df1=pd.read_csv('main.csv',header=None)
print(df1.loc[52][0])
df1=df1.fillna(value=0.0)
print(df1.loc[52][0])

df1.to_csv('main2.csv',header=False,index=False)