import pandas as pd
from shutil import copyfile

df = pd.read_csv('filenames.csv')

for j in range(0,167):
	print('now copying '+df.loc[j][0])
	copyfile('./wav/'+df.loc[j][0],'./newwav/'+df.loc[j][1])
