import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pickle, os

def imshow(img):
    npimg = img.numpy()
    fig = plt.figure(figsize = (10,10))
    ax = fig.add_subplot(111)
    
    ax.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
  
    
def get_infinite_batches(data_loader):
    while True:
        for i, (images, _) in enumerate(data_loader):
            yield images
            
            
def weights_init(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
        
        
def save_model(G, D, name):
    torch.save(G.state_dict(), './generator_' + name + '.pkl')
    torch.save(D.state_dict(), './discriminator_' + name + '.pkl')
    
    
def load_model(G, D, name):
    G_model_path = os.path.join(os.getcwd(), 'generator_' + name + '.pkl')
    D_model_path = os.path.join(os.getcwd(), 'discriminator_' + name + '.pkl')
    G.load_state_dict(torch.load(G_model_path))
    D.load_state_dict(torch.load(D_model_path))
    
    
def save_dict (dic, dic_name):
    with open(dic_name + '.pkl', 'wb') as file:
        pickle.dump(dic, file)
    return 0


def load_dict (dic_name):
    with open(dic_name + '.pkl', 'rb') as file:
        dic = pickle.load(file)
    return dic