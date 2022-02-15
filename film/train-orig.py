'''
This code trains the original resnet on the NWPU dataset. 
It first trains the resent on the NWPU training set, and evaluates its performance on the validata set.
The params corrsponding to the best performance on the validation set are recorded.

It then trains the resnet on the training + validation set by using the cross validated params on the validation set.

Thus, this file should be implemented twice with "trainval=False" in the 1st trial and "trainval=True" in the 2nd.

The following hyper-params should be cared.
trainval: False: training model on the training set and validating it on the validation set, respectively.
          True: only training on the train+val set.
'''
import os
import json
import pdb

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torchvision.models import resnet
from tqdm import tqdm
import torchvision.models as models
import resnet
import pickle as pk
from utils import data_transform

# you shuold change the following paths to your own paths
modelzoo = os.path.expanduser('~/Code/pymodels')  # place for pytorch official models
datapath = os.path.expanduser('~/Data/nwpu')  # the path for your data
progpath = os.path.dirname(os.path.realpath(__file__))
modelpath = os.path.join(progpath, 'models')    # place to store your trained model
rsltpath = os.path.join(progpath, 'results')    # place to store your training records
import sys
sys.path.append(progpath)

def main():
    # training environment
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    ######################## training hyper-parameters
    batch_size = 128
    epochs = 30
    step =10
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    trainval = False    # False: train and validation; True: only training on train + validation set
    lr = 0.0001
    saved_model_path = os.path.join(modelpath, 'orig-resnet.pth')
    if trainval:
        train_set_path = "images"
    else:
        train_set_path = "train"

    ########################## preparing data
    # preparing training data
    train_dataset = datasets.ImageFolder(root=os.path.join(datapath, train_set_path),
                                         transform=data_transform["train"])
    num_train_samples = len(train_dataset)
    num_classes = len(train_dataset.classes)
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)


    # preparing validation data
    if not trainval:
        validate_dataset = datasets.ImageFolder(root=os.path.join(datapath, "val"),
                                                transform=data_transform["val"])
        num_val_samples = len(validate_dataset)
        validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                    batch_size=batch_size, shuffle=False,
                                                    num_workers=nw)

        print("using {} images for training, {} images for validation.".format(num_train_samples, num_val_samples))
    else:
        print("Training on train+val by using {} images.".format(num_train_samples))
                                                                        

    
    ########################## preparing models  
    model = resnet.resnet34(pretrained=False)
    if not trainval:  # False, finetuing on source training data using pretained weights on ImageNet   
        pretrained_weights = torch.load(os.path.join(modelzoo, "resnet34-b627a593.pth"), map_location=device)
        model.load_state_dict(pretrained_weights)
    else:
        pretrained_weights = torch.load(os.path.join(modelpath, "orig-resnet.pth"), map_location=device)
        model.load_state_dict(pretrained_weights)
    
    in_channel = model.fc.in_features
    model.fc = nn.Linear(in_channel, num_classes)
    model.to(device)

    
    ##################### other training criteria
    # define loss function
    loss_function = nn.CrossEntropyLoss()

    # construct an optimizer and scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.Adam(params, lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step)


    

    ####################### training loops
    best_acc = 0.0   
    rcd_train_loss = []
    rcd_train_acc = []
    rcd_val_acc = []

    for epoch in range(epochs):
        
        ####################### train
        model.train()
        running_loss = 0.0
        running_train_acc = 0.0
        
        for images, labels in train_loader:
            optimizer.zero_grad()
            logits = model(images.to(device))
            loss = loss_function(logits, labels.to(device))
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item()
            running_train_acc += torch.sum(torch.eq(torch.argmax(logits.cpu(), dim=1), labels.cpu()))

        
        rcd_train_loss.append(running_loss)
        cur_train_acc = running_train_acc/num_train_samples
        rcd_train_acc.append(cur_train_acc)

        ################# validation
        if not trainval:
            model.eval()
            running_val_acc = 0.0  # accumulate accurate number / epoch
            with torch.no_grad():
                val_bar = tqdm(validate_loader)
                for val_data in val_bar:
                    val_images, val_labels = val_data
                    outputs = model(val_images.to(device))
                    predict_y = torch.max(outputs, dim=1)[1]
                    running_val_acc += torch.eq(predict_y.cpu(), val_labels.cpu()).sum().item()

                    val_bar.desc = "valid epoch[{}/{}]".format(epoch, epochs)
            
            cur_val_acc = running_val_acc / num_val_samples
            rcd_val_acc.append(cur_val_acc)

            # storing the best model and params
            if cur_val_acc > best_acc:
                best_acc = cur_val_acc
                opt_epoch = epoch
                torch.save(model.state_dict(), saved_model_path)

        
        scheduler.step()
        
        cur_lr = scheduler.get_last_lr()
        if not trainval:
            print("Epoch-{}, train loss:{}, train acc:{:.3f}, val acc:{:.3f}, lr:{}\n".format(\
                epoch, running_loss, cur_train_acc, cur_val_acc, cur_lr))
        else:
            print("Epoch-{}, train loss:{}, train acc:{:.3f}, lr:{}\n".format(\
                epoch, running_loss, cur_train_acc, cur_lr))



    rslt_file_path = os.path.join(rsltpath, 'records-orig-resnet.pk')
    with open(rslt_file_path, 'wb') as f:
        pk.dump({'train_loss': rcd_train_loss, 'train_acc':rcd_train_acc,\
            'val_acc':rcd_val_acc, 'best_val_acc': best_acc, 'optimal_epoch':opt_epoch, 'batch_size':batch_size}, f)
        
    
    if trainval:
        torch.save(model.state_dict(), saved_model_path)

    print('Finished Training')


if __name__ == '__main__':
    main()
