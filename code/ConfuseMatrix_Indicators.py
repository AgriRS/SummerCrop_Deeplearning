
# !----------  提取所有类别标签在不同波段上的特征值，以进行混淆矩阵生成 ---------------！#
import gdal
import os
import numpy as np
from numpy import *

# !----------  重要前提，请保证 两个  tif  影像的区域  一致 ---------------！#
# !----------  当区域  不一致  时，用背景值 填充一致 ---------------！#

def readTif(fileName):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName + "无法读取影像数据！")
    return dataset

#修改参数部分
#分类结果的tif影像路径
Image_Path = r'D:\classify.tif'

#地面真值需要对比的tif文件路径
LabelPath = r'D:\label.tif'
#  2019_area_2_Samples  2019_area_8_Samples  2020_area_2_Samples   2020_area_8_Samples
#  2021_area_2_Samples  2021_area_3_Samples  2021_area_5_Samples   2021_area_8_Samples

#生成结果的txt文件保存路径
txt_Path = r"D:\1111.txt"

class_Num = 10 #分类类别数量，除去背景值的,这需要根据自己分类数量进行修改

#以下代码不需要再进行参数修改
if os.path.exists(txt_Path):
    os.remove(txt_Path)

dataset = readTif(Image_Path)
Tif_width = dataset.RasterXSize
Tif_height = dataset.RasterYSize

Image_data = dataset.ReadAsArray(0, 0, Tif_width, Tif_height)

datasets = readTif(LabelPath)
Tifwidth = datasets.RasterXSize
Tifheight = datasets.RasterYSize

Label_data = datasets.ReadAsArray(0, 0, Tifwidth, Tifheight)

confuse_Matrix = np.zeros((class_Num, class_Num), dtype = longlong)
confuse_Matrix_label = 0
confuse_Matrix_class = 0
str_label_class = ''

for i in range(Label_data.shape[0]):
    for j in range(Label_data.shape[1]):
        # print(Label_data[i][j])
        if Label_data[i][j] > 0 and Label_data[i][j] <= class_Num: #10类别，超出类别的不算
            var_value = str(Label_data[i][j]) + ',' + str(Image_data[i][j]) + "\n"  # 获取值
            if Label_data[i][j] > class_Num or Image_data[i][j] > class_Num:
                print(var_value)

            # str_label_class = str(Label_data[i][j] - 1) + '_' + str(Image_data[i][j] - 1)

            confuse_Matrix_label = Label_data[i][j] - 1
            confuse_Matrix_class = Image_data[i][j] - 1

            confuse_Matrix[confuse_Matrix_label, confuse_Matrix_class] = confuse_Matrix[confuse_Matrix_label, confuse_Matrix_class] + 1
# test data as an example
# confuse_Matrix = np.array([[400801,1120,47,140,1,661,664,1217,10,442],
#                 [4143,130889,152,257,0,128,11,267,0,13],
#                 [2493,2127,43868,82,33,945,3,157,5,1],
#                 [3580,16,0,55153,0,1,685,151,94,0],
#                 [32,0,0,1,7690,1189,442,5845,1,267],
#                 [958,101,9,248,23,19410,651,3193,1,138],
#                 [376,0,0,12,0,297,45702,1244,59,17],
#                 [10349,1593,404,1080,368,864,628,162370,754,1575],
#                 [16,0,2,77,0,5,223,289,29740,15],
#                 [54,3,4,0,0,133,30,616,1,11344]
#                 ])

print(confuse_Matrix)

print('----------------------------Confusion Matrix Success-------------------------')

np.savetxt(txt_Path, confuse_Matrix, fmt="%d", delimiter=',')

print('----------------------------Write to txt file Success------------------------')

dialog_value = np.diag(confuse_Matrix).sum() #for OA accuracy

overall_accuracy = dialog_value / confuse_Matrix.sum()
overall_accuracy = ('%.6f' % overall_accuracy)

print('-----------------------------------OA Success--------------------------------')
print('Overall accuracy (OA): ' + overall_accuracy)

Precision_value_sum = 0 #for F1_score
Precision_value = ''

for i in range(class_Num):
    if(confuse_Matrix[i].sum() != 0):
        p_item_value = confuse_Matrix[i][i] / confuse_Matrix[i].sum()
        Precision_value_sum = Precision_value_sum + p_item_value
        p_item_value = ('%.6f' % p_item_value) #根据需要保留小数位数，这里为6位
    else:
        p_item_value = 'NAN'
    Precision_value = Precision_value + str(p_item_value) + ','

print('----------------------------------UA Success-------------------------------')
print('Productor’s accuracy (UA): ' + Precision_value)

Recall_value = ''
Recall_value_sum = 0 #for F1_score

for i in range(class_Num):
    if(confuse_Matrix[:, i].sum() != 0):
        r_item_value = confuse_Matrix[i][i] / confuse_Matrix[:, i].sum()
        Recall_value_sum = Recall_value_sum + r_item_value
        r_item_value = ('%.6f' % r_item_value) #根据需要保留小数位数，这里为6位
    else:
        r_item_value = 'NAN'
    Recall_value = Recall_value + str(r_item_value) + ','
print('---------------------------------PA Success-------------------------------')
print('User’s accuracy (PA): ' + Recall_value)

c_Num = 0
F1_score = 0.0
F1_score_class_Num = 0
f1_score_classes = ''
str_f1 = ''

for i in range(class_Num):
    if confuse_Matrix[i].sum() != 0:
        p_v = confuse_Matrix[i][i] / confuse_Matrix[i].sum()
    else:
        p_v = 0
    if confuse_Matrix[:, i].sum() != 0:
        r_v = confuse_Matrix[i][i] / confuse_Matrix[:, i].sum()
    else:
        r_v = 0
    if confuse_Matrix[:, i].sum() == 0 and confuse_Matrix[i].sum() == 0:
        c_Num = c_Num + 1
        str_f1 = 'NAN'
    else:
        if p_v + r_v != 0:
            f1_v = 2 * p_v * r_v / (p_v + r_v)
        else:
            f1_v = 0
        # f1_v = ('%.6f' % f1_v)
        F1_score = F1_score + f1_v
        str_f1 = str(f1_v)
        f1_score_classes = f1_score_classes + str_f1 + ','

F1_score = F1_score / (class_Num - c_Num)
F1_score_class_Num = class_Num - c_Num
F1_score = ('%.6f' % F1_score)
print('-------------------------------F1_score Success---------------------------')
print('F1_score: ' + F1_score)
print('F1_score for each class: ' + f1_score_classes)
print('The num for F1_score : ' + str(F1_score_class_Num))

F1_score_old = 0
Precision_value_avg = Precision_value_sum / (class_Num - c_Num)
Recall_value_avg = Recall_value_sum / (class_Num - c_Num)
F1_score_old = 2 * Precision_value_avg * Recall_value_avg / (Precision_value_avg + Recall_value_avg)
F1_score_old = ('%.6f' % F1_score_old)
print('-------------------------------F1_score_old Success---------------------------')
print('F1_score_old: ' + F1_score_old)

c_N = 0
Kappa_pe = np.float(0)
for i in range(class_Num):
    row_sum = confuse_Matrix[i].sum()
    column_sum = confuse_Matrix[:, i].sum()
    # Kappa_pe = Kappa_pe + (row_sum / confuse_Matrix.sum()) * (column_sum / confuse_Matrix.sum())
    Kappa_pe = Kappa_pe + (row_sum * column_sum) / (confuse_Matrix.sum() * confuse_Matrix.sum())

Kappa_value = (np.float(overall_accuracy) - Kappa_pe) / (1 - Kappa_pe)
Kappa_value = ('%.6f' % Kappa_value)
print('--------------------------Kappa coefficient Success----------------------')
print('Kappa: ' + Kappa_value)

IoU_value = 0
IoU_value_all = ''
for i in range(class_Num):
    TP_FN_FP = confuse_Matrix[i].sum() + confuse_Matrix[:, i].sum() - confuse_Matrix[i][i]
    if TP_FN_FP != 0:
        IoU_value = confuse_Matrix[i][i] / TP_FN_FP
        IoU_value = ('%.6f' % IoU_value)
    else:
        IoU_value = 'NAN'
    IoU_value_all = IoU_value_all + str(IoU_value) + ','

print('----------------------------------IoU Success--------------------------')
print('IoU : ' + IoU_value_all)


#将其他指标数据写入txt文件
str_Values_append_txt = 'Overall accuracy (OA): ' + str(overall_accuracy) + '\n' \
                        + 'Productor’s accuracy (PA): ' + str(Precision_value) + '\n' \
                        + 'User’s accuracy (UA): ' + str(Recall_value) + '\n' \
                        + 'F1_score: ' + str(F1_score) + '________num for f1_score:' + str(F1_score_class_Num) + '\n' \
                        + 'F1_score for each class: ' + f1_score_classes + '\n' \
                        + 'F1_score_old: ' + F1_score_old + '\n' \
                        + 'Kappa: ' + str(Kappa_value) + '\n' \
                        + 'IoU : ' + IoU_value_all

with open(txt_Path, mode = 'a') as file:
    file.write(str_Values_append_txt)

print('---------------------------------ALL Success-------------------------------')
