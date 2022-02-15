import os,shutil
import numpy as np
from torchvision import transforms


data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
        "val": transforms.Compose([transforms.Resize(256),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor(),
                                   transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}


def generate_random_train_set(train_img_dir: str, num_train_per_class: int):
    
    dirfolder, datafolder = os.path.split(train_img_dir)
    if os.path.exists(os.path.join(dirfolder, 'fstrain')):
        shutil.rmtree(os.path.join(dirfolder, 'fstrain'))
        
    os.mkdir(os.path.join(dirfolder, 'fstrain'))
    folders = os.listdir(train_img_dir)
    for subfolder in folders:
        image_files = os.listdir(os.path.join(train_img_dir, subfolder))
        num_iamges = len(image_files)
        # assert num_iamges == 400, "Each class should have 400 images"
        index = np.random.permutation(num_iamges)
        train_index = index[:num_train_per_class]
    

        if not os.path.exists(os.path.join(dirfolder, 'fstrain', subfolder)):
            os.mkdir(os.path.join(dirfolder, 'fstrain', subfolder))


        for ind in train_index:
            shutil.copy(os.path.join(train_img_dir, subfolder, image_files[ind]), \
                os.path.join(dirfolder, 'fstrain', subfolder))
        