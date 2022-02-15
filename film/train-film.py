'''
This code trians the filmed model pretrained on NWPU on the GID rgb-224 dataset.

The GID rgb-224 dataset has 15 classes. 

It trains the filmed-resent on the gid training set, and evaluates its performance on the validata set, 
then saves the best finetuned model

It should be noted that we randomly select "num_train_per_class" samples from the training set to compose 
a few-shot training set for finetuning in each trial. Thus, we model the case where only few labeled samples
are available for supervised learning.

We run the code under the same "num_train_per_class" for 5 trials to report the average performance.
 

The following hyper-params should be cared.
    num_train_per_class: the number of training samples from each class

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

    # training hyper-parameters
    batch_size = 64
    eval_batch_size = 256
    epochs = 50
    step = 15
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    num_train_per_class = 10
    lr = 0.003


    # preparing validation data
    validate_dataset = datasets.ImageFolder(root=os.path.join(datapath, "val"),
                                            transform=data_transform["val"])
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                batch_size=eval_batch_size, shuffle=False,
                                                num_workers=nw)
    num_val_samples = len(validate_dataset)



    for num_train_per_class in [3, 5, 10, 15, 20, 30, 50]:
        for trial in range(5):

            saved_model_path = os.path.join(modelpath, 'film{}-resnet-trial{}.pth'.format(num_train_per_class, trial))

            ##################### preparing training data
            generate_random_train_set(os.path.join(datapath, 'train'), num_train_per_class=num_train_per_class)
            train_dataset = datasets.ImageFolder(root=os.path.join(datapath, "fstrain"),
                                                transform=data_transform["train"])
            train_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=batch_size, shuffle=True,
                                                    num_workers=nw)
            num_train_samples = len(train_dataset)
            num_classes = len(train_dataset.classes)

            print("using {} images for training, {} images for validation.".format(num_train_samples, num_val_samples))

            #################### preparing models  
            model = resnet.resnet34(pretrained=False, model_dir=modelzoo, film_layer=True)
            pretrained_weights = torch.load(os.path.join(modelpath, "orig-resnet.pth"), map_location=device)
            pretrained_weights.pop('fc.weight')
            pretrained_weights.pop('fc.bias')
            model.load_state_dict(pretrained_weights, strict=False)

            for k,v in model.named_parameters():
                if k.__contains__('film'):
                    v.requires_grad=True
                else:
                    v.requires_grad=False
            
            in_channel = model.fc.in_features
            model.fc = nn.Linear(in_channel, num_classes)
            model.to(device)

            ################# training criteria
            # define loss function
            loss_function = nn.CrossEntropyLoss()

            # construct an optimizer and scheduler
            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = optim.Adam(params, lr=lr)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step)


            ########################### training loop
            best_acc = 0.0   
            rcd_train_loss = []
            rcd_train_acc = []
            rcd_val_acc = []
            best_epoch = None
            train_steps = len(train_loader)
            for epoch in range(epochs):
                ####################### train
                model.train()
                running_loss = 0.0
                running_train_acc = 0.0
                # train_bar = tqdm(train_loader)
                # for step, data in enumerate(train_bar):
                #     images, labels = data
                for images, labels in train_loader:
                    # pdb.set_trace()
                    optimizer.zero_grad()
                    logits = model(images.to(device))
                    loss = loss_function(logits, labels.to(device))
                    loss.backward()
                    optimizer.step()

                    # print statistics
                    running_loss += loss.item()
                    running_train_acc += torch.sum(torch.eq(torch.argmax(logits.cpu(), dim=1), labels.cpu()))

                
                rcd_train_loss.append(running_loss)
                cur_train_acc = running_train_acc/num_train_samples
                rcd_train_acc.append(cur_train_acc)

                ################# validation
                
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
                if cur_val_acc > best_acc:
                    best_acc = cur_val_acc
                    opt_epoch = epoch
                    torch.save(model.state_dict(), saved_model_path)

                
                scheduler.step()
                
                cur_lr = scheduler.get_last_lr()
                
                print("#{}-Trial-{}-Epoch-{}, train loss:{}, train acc:{:.3f}, val acc:{:.3f}, lr:{}\n".format(\
                    num_train_per_class, trial, epoch, running_loss, cur_train_acc, cur_val_acc, cur_lr))
            
            ####### record params
            rslt_file_path = os.path.join(rsltpath, 'records-film{}-resnet-trial{}.pk'.format(num_train_per_class, trial))
            with open(rslt_file_path, 'wb') as f:
                pk.dump({'train_loss': rcd_train_loss, 'train_acc':rcd_train_acc,\
                    'val_acc':rcd_val_acc, 'best_val_acc': best_acc, 'optimal_epoch':opt_epoch,\
                        'step':step, 'num_trains':num_train_per_class, 'batch_size':batch_size, 'total_epochs':epochs}, f)


    print('Finished Training')


if __name__ == '__main__':
    main()
