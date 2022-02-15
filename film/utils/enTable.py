# *_*coding: utf-8 *_*
# 对图像进行网格化处理

import os
import numpy as np
from osgeo import gdal


# 定义读取和保存图像的类
class GRID:

    def load_image(self, filename):
        image = gdal.Open(filename)

        img_width = image.RasterXSize
        img_height = image.RasterYSize

        img_geotrans = image.GetGeoTransform()
        img_proj = image.GetProjection()
        img_data = image.ReadAsArray(0, 0, img_width, img_height)

        del image

        return img_proj, img_geotrans, img_data

    def write_image(self, filename, img_proj, img_geotrans, img_data):
        # 判断栅格数据类型
        if 'int8' in img_data.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in img_data.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32

        # 判断数组维度
        if len(img_data.shape) == 3:
            img_bands, img_height, img_width = img_data.shape
        else:
            img_bands, (img_height, img_width) = 1, img_data.shape

        # 创建文件
        driver = gdal.GetDriverByName('GTiff')
        image = driver.Create(filename, img_width, img_height, img_bands, datatype)

        image.SetGeoTransform(img_geotrans)
        image.SetProjection(img_proj)

        if img_bands == 1:
            image.GetRasterBand(1).WriteArray(img_data)
        else:
            for i in range(img_bands):
                image.GetRasterBand(i + 1).WriteArray(img_data[i])

        del image  # 删除变量,保留数据


if __name__ == '__main__':
    import time
    import argparse

    parser = argparse.ArgumentParser(description='load remote sensing image and split to patch')
    parser.add_argument('--image_path',
                        default='D:/Large-scale Classification_5classes/Large-scale Classification_5classes/image_RGB_2/',
                        help='remote sensing image path')
    parser.add_argument('--patch_size',
                        default=56,
                        help='patch size')
    parser.add_argument('--patch_save',
                        default='D:/Large-scale Classification_5classes/Large-scale Classification_5classes/image_tabled_56_2/image2_',
                        help='save path of patch image')
    args = parser.parse_args()
    print('待处理图像路径为:{}'.format(args.image_path))
    print('分块大小为:{}'.format(args.patch_size))
    print('分块图像保存路径:{}'.format(args.patch_save))

    image_path = args.image_path
    image_list = os.listdir(image_path)
    # image_list.sort(key=lambda x: int(x[:-4])) # 对文件夹中的图像进行排序
    image_num = len(image_list)

    t_start = time.time()
    for k in range(image_num):
        time_start = time.time()
        img_name = image_path + image_list[k]
        proj, geotrans, data = GRID().load_image(img_name)

        # 图像分块
        patch_size = args.patch_size
        patch_save = args.patch_save
        channel, width, height = data.shape

        num = 0
        for i in range(width // patch_size):
            for j in range(height // patch_size):
                num += 1
                sub_image = data[:, i * patch_size:(i + 1) * patch_size, j * patch_size:(j + 1) * patch_size]
                str=i.__str__()+'_'+j.__str__()
                GRID().write_image(patch_save + '{}.tif'.format(str), proj, geotrans, sub_image)

        print(num)
        time_end = time.time()
        print('第{}张图像分块完毕, 耗时:{}秒'.format(k + 1, round((time_end - time_start), 4)))

    t_end = time.time()
    print('所有图像处理完毕,耗时:{}秒'.format(round((t_end - t_start), 4)))