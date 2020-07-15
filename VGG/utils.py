import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import pickle

def set_seed(seed):
    # for reproducibility. 
    # note that pytorch is not completely reproducible 
    # https://pytorch.org/docs/stable/notes/randomness.html  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  #mulii worker로 dataload를 할 때
    np.random.seed(seed)
    torch.initial_seed()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return None


def imshow(img, mean=0.5, std=0.5):
    img = img * std + mean     # unnormalize
    npimg = img.numpy()
    fig = plt.figure(figsize = (50,50))
    ax = fig.add_subplot(111)
    
    ax.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
        
        
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def weight_reset(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        m.weight.data.normal_(0.0, 0.035)  #tuned std
        m.bias.data.fill_(0)
        
def LoadDict (dic_name, dic_type = 'dict'):
    with open('./DICT/' + dic_name + '.' + dic_type, 'rb') as file:
        dic = pickle.load(file)
    return dic
    
    
def SaveDict (dic, dic_name, dic_type = 'dict'):
    path = './DICT'
    if os.path.isdir(path) == False:
        os.mkdir(path)
    with open(path + dic_name + '.' + dic_type, 'wb') as file:
        pickle.dump(dic, file)
    return 0


def countParam(dic = 'A_best'):
    model = LoadDict(dic)

    num_param = 0
    for name, param in model.items():
        shape = list(model[name].shape)
        p = 1
        for i in shape:
            p *= i
        num_param += p

    return num_param

def getValAcc(dic = 'Val_acc_A'):
    acc = LoadDict(dic)
    return [float(i) for i in acc]
    