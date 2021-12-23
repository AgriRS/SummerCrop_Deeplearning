import os
import numpy as np
from numpy import *

#修改参数部分
#所有txt结果的文件夹路径
txt_path = r'D:\SummerSentinelData\Results\ConfusionMatrix\DeepLearning\Spetral'
class_Num = 10 #分类类别数量，除去背景值的,这需要根据自己分类数量进行修改
#生成结果的txt保存位置
txt_Path = r"D:\SummerSentinelData\Results\ConfusionMatrix\DeepLearning\Spetral\Spetral_Merge_result.txt"


if os.path.exists(txt_Path):
    os.remove(txt_Path)

confuse_Matrix = np.zeros((class_Num, class_Num), dtype = longlong)
sub_confuse_Matrix = np.zeros((class_Num, class_Num), dtype = longlong)
list_data = {}
file_num = 1
files = os.listdir(txt_path)

for file in files:
    sub_list_data = []
    print(file)
    f_open = open(os.path.join(txt_path, file))
    for i in range(class_Num):
        dataList = f_open.readline()
        sub_list_data.append(dataList.split(','))

    list_data[file_num] = sub_list_data
    file_num = file_num + 1

# print(list_data[1][9][9])
print('------------------Sum of the Confusion results--------------------')
# print(len(list_data))
sum_list_data = []
# for i in range(2, txt_File_num):
for _row in range(0, class_Num):
    for _col in range(0, class_Num):
        for l_n in range(1, len(list_data) + 1):
            if l_n == 1:
                confuse_Matrix[_row, _col] = longlong(list_data[1][_row][_col])
            else:
                sub_confuse_Matrix[_row, _col] = longlong(list_data[l_n][_row][_col])
                confuse_Matrix[_row, _col] = confuse_Matrix[_row, _col] + sub_confuse_Matrix[_row, _col]

print(confuse_Matrix)

np.savetxt(txt_Path, confuse_Matrix, fmt="%d", delimiter=',')

print(confuse_Matrix.sum())
print('-----------------------------Success-------------------------------')