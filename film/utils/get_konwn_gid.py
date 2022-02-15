import os
import json

import torch
from PIL import Image
from prettytable import PrettyTable
from torch import nn
from torchvision import transforms

import numpy as np
from osgeo import gdal


from film import resnet

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

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    # 指向需要遍历预测的图像文件夹
    imgs_root = r'D:\FILM_Network\gid\batch_predict\label_5_tabled_1'
    assert os.path.exists(imgs_root), f"file: '{imgs_root}' dose not exist."
    # 读取指定文件夹下所有jpg图像路径
    img_path_list = [os.path.join(imgs_root, i) for i in os.listdir(imgs_root) if i.endswith(".tif")]

    image_num=len(img_path_list)
    all_three_channel = []
    image_dict = {}
    table = PrettyTable()
    table.field_names = ["", "axis", "class"]

    # 获得像素点信息
    for k in range(image_num):
        img_name = img_path_list[k]
        proj, geotrans, data = GRID().load_image(img_name)
        image=Image.open(img_path_list[k])
        image_color=image.getcolors()
        print(image_color)

        # 获得图片坐标
        img_path_axis = img_path_list[k].split('image1')
        img_path_axis = img_path_axis[1].split('.')
        img_path_axis = img_path_axis[0].split('_')

        axis = '(' + img_path_axis[1].__str__() + ',' + img_path_axis[2].__str__() + ')'

        max_color_pixcel=0
        image_color_len=len(image_color)
        # 判断像素点
        for i in range(image_color_len):
            if image_color[i][0]>image_color[max_color_pixcel][0]:
                max_color_pixcel=i

        # 最多像素的颜色的RGB值
        max_rgb=image_color[max_color_pixcel][1]
        print(max_rgb)
        classes_gid_5 = ['bulit_up', 'forest', 'farmland', 'undefined', 'water']
        if max_rgb==(0,0,0):
            image_dict[axis]='undefined'
        elif max_rgb==(0,255,0):
            image_dict[axis]='farmland'
        elif max_rgb==(0,0,255):
            image_dict[axis]='water'
        elif max_rgb==(255,0,0):
            image_dict[axis]='bulit_up'
        else:
            image_dict[axis]='forest'

        table.add_row([img_path_list[k], axis, image_dict[axis]])

        # 获得五个颜色的RGB值
        if max_rgb not in all_three_channel:
            all_three_channel.append(max_rgb)

    jsObj = json.dumps(image_dict, indent=4)  # indent参数是换行和缩进
    fileObject = open('gid1_konwn_dict.json', 'w')
    fileObject.write(jsObj)
    fileObject.close()

    print(table)



if __name__ == '__main__':
    main()
