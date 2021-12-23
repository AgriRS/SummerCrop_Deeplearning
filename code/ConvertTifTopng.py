from osgeo import gdal
import os
import glob

# !----------  标签裁切后为tif影像，模型要求为PNG格式，进行批量格式转换 ---------------！#

LabelTifpath = r'D:\SummerSentinelData\StretchData\15Bands\TrainingData20211014\Label'    #Label Tif 所在文件夹
labelPNGPath = r'D:\SummerSentinelData\StretchData\15Bands\TrainingData20211014\LabelPNG'    #输出Label png 所在文件夹
filenames = os.listdir(LabelTifpath)
tif_count = 0
for fn in filenames:
    print(fn)
    file_path = os.path.join(LabelTifpath, fn)
    print(file_path)
    ds = gdal.Open(file_path)
    # print(ds)
    driver = gdal.GetDriverByName('PNG')
    pngPath = os.path.join(labelPNGPath, fn.replace('tif', 'png')) #.split('.')[0]+".png"
    print(pngPath)
    dst_ds = driver.CreateCopy(pngPath, ds)
    dst_ds = None
    src_ds = None
    tif_count = tif_count +1
print('一共处理了 ', tif_count, ' 个影像！')
for xmlfile in glob.glob(os.path.join(labelPNGPath,'*.xml')):
    os.remove(xmlfile)