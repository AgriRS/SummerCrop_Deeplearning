from osgeo import gdal
import os
import glob
import math

# !----------  将test形成的PNG且经过PNGConvertToTif.py处理后的结果进行拼接形成完整结果 ---------------！#

#获取影像范围的的左上角和右下角坐标
def GetExtent(in_fn):
    ds = gdal.Open(in_fn)
    geotrans = list(ds.GetGeoTransform())
    xsize = ds.RasterXSize
    ysize = ds.RasterYSize
    min_x = geotrans[0]
    max_y = geotrans[3]
    max_x = geotrans[0]+xsize*geotrans[1]
    min_y = geotrans[3]+ysize*geotrans[5]
    ds=None
    return min_x, max_y, max_x, min_y
#test结果Mosaic生成文件路径
path = r"D:\SummerSentinelData\20211120\All_prediction_Images"
os.chdir(path)
mosaic_file_name = '2021_5.tif'


#如果存在同名影像则先删除
if os.path.exists(mosaic_file_name):
    os.remove(mosaic_file_name)

#获取所有结果的tif文件  16Bands_012model  20Bands_011model
in_files = glob.glob(r"D:\SummerSentinelData\20211120\All_prediction_Images\2021_5\*.tif") #GF6_1
in_fn = in_files[0]
#获取待镶嵌栅格的最大最小的坐标值
min_x, max_y, max_x, min_y = GetExtent(in_fn)
for in_fn in in_files[1:]:
    minx, maxy, maxx, miny = GetExtent(in_fn)
    min_x = min(min_x, minx)
    min_y = min(min_y, miny)
    max_x = max(max_x, maxx)
    max_y = max(max_y, maxy)
#计算镶嵌后影像的行列号
in_ds = gdal.Open(in_files[0])
geotrans = list(in_ds.GetGeoTransform())
width = geotrans[1]
height = geotrans[5]

columns = math.ceil((max_x-min_x)/width)
rows = math.ceil((max_y-min_y)/(-height))
in_band = in_ds.GetRasterBand(1)

driver = gdal.GetDriverByName('GTiff')
out_ds = driver.Create(mosaic_file_name, columns, rows, 1, in_band.DataType)
out_ds.SetProjection(in_ds.GetProjection())
geotrans[0] = min_x
geotrans[3] = max_y
out_ds.SetGeoTransform(geotrans)
out_band = out_ds.GetRasterBand(1)
#定义仿射逆变换
inv_geotrans = gdal.InvGeoTransform(geotrans)
#开始逐渐写入
for in_fn in in_files:
    in_ds = gdal.Open(in_fn)
    in_gt = in_ds.GetGeoTransform()
    #仿射逆变换
    offset = gdal.ApplyGeoTransform(inv_geotrans, in_gt[0], in_gt[3])
    x, y = map(int, offset)
    print(x, y)
    trans = gdal.Transformer(in_ds,out_ds, []) #in_ds是源栅格，out_ds是目标栅格
    success, xyz = trans.TransformPoint(False, 0, 0) #计算in_ds中左上角像元对应out_ds中的行列号
    x, y, z = map(int, xyz)
    print(x, y, z)
    data = in_ds.GetRasterBand(1).ReadAsArray()
    out_band.WriteArray(data, x, y) #x，y是开始写入时左上角像元行列号
del in_ds, out_band, out_ds
