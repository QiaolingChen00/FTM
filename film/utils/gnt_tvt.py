import os
import pandas as pd 
import numpy as np
import pickle as pk

datapath = r'/home/luowei/Data/nwpu'
trainpath = os.path.join(datapath, 'train')
valpath = os.path.join(datapath,'val')
testpath = os.path.join(datapath, 'test')
progpath = os.path.dirname(__file__)
os.chdir(progpath)

train_images = []
train_classes = os.listdir(trainpath)
for subf in train_classes:
    image_names = os.listdir(os.path.join(trainpath, subf))
    image_names = [os.path.join(subf, img_name) for img_name in image_names]
    train_images.extend(image_names)
train_flags = [0] * len(train_images)


test_images = []
test_classes = os.listdir(testpath)
for subf in test_classes:
    image_names = os.listdir(os.path.join(testpath, subf))
    image_names = [os.path.join(subf, img_name) for img_name in image_names]
    test_images.extend(image_names)
test_flags = [1] * len(test_images)

val_classes = os.listdir(valpath)
val_images = []
val_flags = []
if val_classes:
    for subf in val_classes:
        image_names = os.listdir(os.path.join(valpath, subf))
        image_names = [os.path.join(subf, img_name) for img_name in image_names]
        val_images.extend(image_names)
    val_flags = [2] * len(val_images)

imdb = pd.DataFrame()
imdb['imgnames'] = train_images + test_images + val_images
imdb['tvt'] = train_flags + test_flags + val_flags

with open(os.path.join(datapath, 'imdb.pk'), 'wb') as f:
    pk.dump(imdb, f)
