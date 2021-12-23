from osgeo import gdal

# !----------  将图像和标签以固定size进行裁切，同时保持裁切后的标签空间投影和位置的正确 ---------------！#

class ClipImageByLocation:
    # 读图像文件
    def read_img(self, filename):
        dataset = gdal.Open(filename)  # 打开文件
        im_width = dataset.RasterXSize  # 栅格矩阵的列数
        im_height = dataset.RasterYSize  # 栅格矩阵的行数

        im_geotrans = dataset.GetGeoTransform()  # 仿射矩阵
        im_proj = dataset.GetProjection()  # 地图投影信息
        im_data = dataset.ReadAsArray(0, 0, im_width, im_height)  # 将数据写成数组，对应栅格矩阵

        del dataset
        return im_proj, im_geotrans, im_data

    # 写文件，以写成tif为例
    def write_img(self, filename, im_proj, origin_x, origin_y, pixel_width, pixel_height, im_data):
        # gdal数据类型包括
        # gdal.GDT_Byte,
        # gdal .GDT_UInt16, gdal.GDT_Int16, gdal.GDT_UInt32, gdal.GDT_Int32,
        # gdal.GDT_Float32, gdal.GDT_Float64
        # print(filename)

        # 判断栅格数据的数据类型
        if 'int8' in im_data.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in im_data.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32

        # 判读数组维数
        if len(im_data.shape) == 3:
            im_bands, im_height, im_width = im_data.shape
        else:
            im_bands, (im_height, im_width) = 1, im_data.shape
            # 创建文件
        driver = gdal.GetDriverByName("GTiff")  # 数据类型必须有，因为要计算需要多大内存空间
        dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)

        dataset.SetGeoTransform((origin_x, pixel_width, 0, origin_y, 0, pixel_height))  # 写入仿射变换参数
        dataset.SetProjection(im_proj)  # 写入投影
        print('File name: ', filename, ' and Bands Numbers:', im_bands)

        if im_bands == 1:
            dataset.GetRasterBand(1).WriteArray(im_data)  # 写入数组数据
        else:
            for i in range(im_bands):
                dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
                # print('-----------------------------------------')
        del dataset

# 计算某行列下像元经纬度
# def calcLonLat(dataset, x, y):
#     minx, xres, xskew, maxy, yskew, yres = dataset.GetGeoTransform()
#     lon = minx + xres * x
#     lat = maxy +xres * y
#     return lon, lat

def ClipMydatasetTif(file_name, RepetitionRate, outputPath, size):
    #file_name = r"D:\pythorch\data\imagelabel\label.tif"
    Clip_num = 8
    RepetitionRate = int(1 - RepetitionRate)
    dataset = gdal.Open(file_name)
    minx, xres, xskew, maxy, yskew, yres = dataset.GetGeoTransform()
    proj, geotrans, data = ClipImageByLocation().read_img(file_name)  # 读数据
    # print(proj)
    # print(geotrans)
    # print(data.shape)
    # print(len(data.shape))
    if (len(data.shape) > 2):
        width = data.shape[1]
        height = data.shape[2]
    else:
        width, height = data.shape
    for j in range(int((height - size * RepetitionRate) / (size * (1 - RepetitionRate)))):
        for i in range(int((width - size * RepetitionRate) / (size * (1 - RepetitionRate)))):
    # for j in range(height // size):  # 切割成256*256小图
    #     for i in range(width // size):
            if (j == height // size):
                # cur_image = data[i * size:(i + 1) * size, j * size:(j + 1) * size]
                if (len(data.shape) == 2):
                    cur_image = data[
                                i * size * (1 - RepetitionRate): (i + 1) * size * (1 - RepetitionRate),
                                j * size * (1 - RepetitionRate): (j + 1) * size * (1 - RepetitionRate)]
                #  如果图像是多波段
                else:
                    cur_image = data[:,
                                i * size * (1 - RepetitionRate): (i + 1) * size * (1 - RepetitionRate),
                                j * size * (1 - RepetitionRate): (j + 1) * size * (1 - RepetitionRate)]
                lon = minx + xres * size * j
                lat = maxy + yres * (i * size)
                ClipImageByLocation().write_img(outputPath.format(i + Clip_num, j), proj,
                                                lon, lat, xres, yres, cur_image)  ##写数据
            else:
                # cur_image = data[i*size:(i + 1) * size, j * size:(j + 1) * size]
                if (len(data.shape) == 2):
                    cur_image = data[
                                i * size * (1 - RepetitionRate): (i + 1) * size * (1 - RepetitionRate),
                                j * size * (1 - RepetitionRate): (j + 1) * size * (1 - RepetitionRate)]
                #  如果图像是多波段
                else:
                    cur_image = data[:,
                                i * size * (1 - RepetitionRate): (i + 1) * size * (1 - RepetitionRate),
                                j * size * (1 - RepetitionRate): (j + 1) * size * (1 - RepetitionRate)]

                lon = minx + xres * size * j
                lat = maxy + yres * (i * size)
            ClipImageByLocation().write_img(outputPath.format(i + Clip_num, j), proj,
                                                lon, lat, xres, yres, cur_image)  ##写数据

if __name__ == "__main__":
    ClipMydatasetTif(r"D:\SummerSentinelData\StretchData\PreSpetralBands\SpetralVIsTextureBands\20200904_T50SKF_area8_VIs_Texture.tif", 0.1,
                     r'D:\SummerSentinelData\StretchData\20211018_area8\20Bands\{}_{}.tif', 256)

    ClipMydatasetTif(r"D:\SummerSentinelData\StretchData\MaskData\20200901_T50SLE_area_2_Mask.tif", 0.1,
                     r'D:\SummerSentinelData\StretchData\PreData\Label\{}_{}.tif', 256)

    print('------------------------Success-------------------')

