import os
import shutil
src_files = os.listdir('./features')

j=1
for file_name in src_files:
    full_file_name = os.path.join('./features/', file_name)
    if (os.path.isfile(full_file_name)):
        shutil.copyfile(full_file_name, './finalfeatures/'+str(j)+'.csv')
        j=j+1