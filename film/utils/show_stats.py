import os
import numpy as np
import pickle as pk
import pprint

progpath = os.path.dirname(__file__)
os.chdir(progpath)
rsltpath = r'../results'
modelpath = r'../models'

def show_model_training_stats(path_to_model_stats: str) -> None:
    with open(path_to_model_stats, 'rb') as f:
        stats = pk.load(f)
    # pprint.pprint(stats)

    for key, val in stats.items():
        if key not in ['train_acc', 'train_loss', 'val_acc']:
            print(key, '--->', val)

def show_model_testing_stats(path_to_result_stats) -> None:
    with open(path_to_result_stats, 'rb') as f:
        stats = pk.load(f)
    for key, val in stats.items():
        print(key, '--->', val)

if __name__=='__main__':
    # show_model_training_stats(os.path.join(rsltpath, 'records-film3-resnet-trial0.pk'))

    show_model_testing_stats(os.path.join(rsltpath, 'eval-film-resnet.pk'))
    # show_model_testing_stats(os.path.join(rsltpath, 'eval-ft-resnet.pk'))
