from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from skimage import io,data,color
import json



img = plt.imread("D:\Large-scale Classification_5classes\Large-scale Classification_5classes\image_RGB_2\GF2_PMS2__L1A0000607681-MSS2.tif")
image = img_as_float(img)

#

#SLIC
# 获得超像素矩阵segments以及种子点center
segments,center = slic(image, n_segments = 300, sigma = 5)
#获得超像素矩阵的长宽
rows,cols=segments.shape
# 获得同样大小的0矩阵
labels=np.zeros([rows,cols])
gray=np.zeros([rows,cols])
#获得种子点的横纵像素坐标
center_id=center[:, 1:3]
# 获得种子点在224*224尺度下的网格坐标image_center_id，并取整
image_center_id=center_id/224
image_center_id=np.floor(image_center_id)
image_center_id = image_center_id.astype(np.uint8)

# 获得类别字典，可以用224*224尺度下的网格坐标，得到分类
image_2_dict=r'D:\FILM_Network\film\utils\image2_join_dict_film3.json'
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

np.save('segments2.npy',segments)
np.save('image_center_id2.npy',image_center_id)
np.save('center_id2.npy',center_id)