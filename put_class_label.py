import pandas as pd

df = pd.read_csv('train.csv',header=None)

for i in range(2,156):
	for j in range(3,23):
		if (df.loc[i][j]==str(1)):
			print('file name: {} |||| class number: {}'.format(df.loc[i][1],j-2))
			filename=df.loc[i][1].replace('wav','csv')
			df2=pd.read_csv('features/'+filename,header=None)
			df2[20]=str(j-2)
			df2.to_csv('finalfeatures/'+filename,header=False,index=False)
