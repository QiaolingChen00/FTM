from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from skimage import io,data,color
import json

segments=np.load('segments2.npy')
image_center_id=np.load('image_center_id2.npy')
center_id=np.load('center_id2.npy')
#获得超像素矩阵的长宽
rows,cols=segments.shape
# 获得同样大小的0矩阵
labels=np.zeros([rows,cols])
gray=np.zeros([rows,cols])

# 获得类别字典，可以用224*224尺度下的网格坐标，得到分类
image_2_dict=r'D:\FILM_Network\film\utils\image2_join_dict_ft3.json'
image2_dict = json.load(open(image_2_dict))
#对相同分类设置同样的值
classes_gid_5 = {'bulit_up':0,
                     'forest':1,
                     'farmland':2,
                     'meadow':3,
                     'water':4,
                 'undefined':5}
len = len(image_center_id)
rgb_anno = np.zeros([rows, cols, 3])
# 循环遍历100个中心点
for i in range(len):
    # 格式化中心点，使其变为与类别字典key相同的结构
    if image_center_id[i][0]==30:
        image_center_id[i][0]=29
    if image_center_id[i][1]==32:
        image_center_id[i][1]=31
    id_str='('+str(image_center_id[i][0])+','+str(image_center_id[i][1])+')'
    # 利用id_str的key获得中心点的class，从而获得分类值0.1，0.2，0.3.....
    img_class=image2_dict[id_str]
    classes_img_5=classes_gid_5[img_class]

    segments_id=segments[center_id[i][0]][center_id[i][1]]
    # built-up: (255,0,0), farmland:(0,255,0), forest:(0,255,255), meadow:(255,255,0), water:(0,0,255)

    if classes_img_5 == 0:
        rgb_anno[..., 0][segments[:] == segments_id] = 255
    elif classes_img_5 == 1:
        rgb_anno[..., 1][segments[:] == segments_id] = 255
        rgb_anno[..., 2][segments[:] == segments_id] = 255
    elif classes_img_5 == 2:
        rgb_anno[..., 1][segments[:] == segments_id] = 255
    elif classes_img_5 == 3:
        rgb_anno[..., 0][segments[:] == segments_id] = 255
        rgb_anno[..., 1][segments[:] == segments_id] = 255
    elif classes_img_5 == 4:
        rgb_anno[..., 2][segments[:] == segments_id] = 255
    else:
        pass

# color=[255,255,255]
# location=(rgb_anno==color)
# # location = location[:, :, 0] & location[:, :, 1] & location[:, :, 2]
# rgb_anno[location]=[0,0,0]
for i in range(6800):
    for j in range(7200):
        if rgb_anno[i][j][0]==255 and rgb_anno[i][j][1]==255 and rgb_anno[i][j][2]==255:
            rgb_anno[i][j][0] = 0
            rgb_anno[i][j][1] = 0
            rgb_anno[i][j][2] = 0

    # gray[segments[:]==segments_id]=classes_img_5
    # for k in range(6800):
    #     for j in range(7200):
    #         if segments[k][j]==segments_id:
    #             gray[k][j]=classes_img_5



# rows_center=center_id[:,0]
# # cols_center=center_id[:,1]


# dst=color.label2rgb(gray)
plt.imshow(rgb_anno, cmap='viridis')
plt.axis('off')
f = plt.gcf()  # 获取当前图像
f.savefig(r'D:\image_2_division_1152108_ft.tif')
f.clear()
plt.ion()
plt.pause(100)
plt.close()


