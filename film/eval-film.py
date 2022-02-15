'''
This code finetunes the model pretrained on NWPU on the GID rgb-224 dataset.

The GID rgb-224 dataset has 15 classes. 

It finetunes the pretrained resent on the gid training set, and evaluates its performance on the validata set, then saves the best finetuned model


The following hyper-params should be cared.
num_train_per_class: the number of training samples from each class

'''
import os
import json
import pdb
from numpy.core.fromnumeric import mean
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torchvision.models import resnet
from tqdm import tqdm
import torchvision.models as models
import resnet
import pickle as pk
from utils import generate_random_train_set,data_transform

# you shuold change the following paths to your own paths
modelzoo = os.path.expanduser('~/Code/pymodels')  # place for pytorch official models
datapath = os.path.expanduser('~/Data/gid')  # the path for your data
progpath = os.path.dirname(os.path.realpath(__file__))
modelpath = os.path.join(progpath, 'models')    # place to store your trained model
rsltpath = os.path.join(progpath, 'results')    # place to store your training records
import sys
sys.path.append(progpath)

def main():
    # training environment
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    ############## training hyper-parameters
    valid_batch_size = 256
    nw = min([os.cpu_count(), valid_batch_size if valid_batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

 

    ########################## preparing data
    
    # preparing validation data
    test_dataset = datasets.ImageFolder(root=os.path.join(datapath, "test"),
                                            transform=data_transform["val"])
    validate_loader = torch.utils.data.DataLoader(test_dataset,
                                                batch_size=valid_batch_size, shuffle=False,
                                                num_workers=nw)
    num_test_samples = len(test_dataset)
    num_classes = len(test_dataset.classes)

    print("using {} images for testing.".format(num_test_samples))

    acc = {}                                                                
    for num_train_per_class in [3,5,10,15,20,30,50]:
        acc_list = []
        for trial in range(5):
            ########################## preparing models  
            model = resnet.resnet34(pretrained=False, film_layer=True)
            in_channel = model.fc.in_features
            model.fc = nn.Linear(in_channel, num_classes)
            pretrained_weights = torch.load(os.path.join(modelpath, "film{}-resnet-trial{}.pth".format(num_train_per_class, trial)), map_location=device)
            model.load_state_dict(pretrained_weights, strict=True)
            model.to(device)

        
            #################  testing
            model.eval()
            running_test_acc = 0.0  # accumulate accurate number / epoch
            with torch.no_grad():
                val_bar = tqdm(validate_loader)
                for val_data in val_bar:
                    val_images, val_labels = val_data
                    outputs = model(val_images.to(device))
                    predict_y = torch.max(outputs, dim=1)[1]
                    running_test_acc += torch.eq(predict_y.cpu(), val_labels.cpu()).sum().item()
            
            test_acc = running_test_acc / num_test_samples
            acc_list.append(test_acc)

            # print("test acc:{:.3f}\n".format(test_acc))

        acc_avg = np.mean(acc_list)
        acc_std = np.std(acc_list)
        acc.setdefault('{}'.format(num_train_per_class), (acc_list, acc_avg, acc_std))
        print("train: {} per class, Avg acc:{:.4f}, var:{:.4f}\n".format(num_train_per_class, acc_avg, acc_std))

    rslt_file_path = os.path.join(rsltpath, 'eval-film-resnet.pk')
    with open(rslt_file_path, 'wb') as f:
        pk.dump(acc, f)
        

    print('Finished Testing')


if __name__ == '__main__':
    main()
