import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import pickle
import pandas as pd

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


def imshow(img):
    npimg = img.numpy()
    fig = plt.figure(figsize = (50,50))
    ax = fig.add_subplot(111)
    
    ax.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()    
        
        
#initialize upsampling to bilinear interpolation
def upsampling_init_weight(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
            center = factor - 1
    else:
            center = factor - 0.5
    og = np.ogrid[:kernel_size, :kernel_size]
    filt = (1 - abs(og[0] - center) / factor) * (1 - abs(og[1] - center) / factor)
    weight = np.zeros((in_channels, out_channels, kernel_size, kernel_size), dtype=np.float64)
    weight[list(range(in_channels)), list(range(out_channels)), :, :] = filt
    return torch.from_numpy(weight).float()


def decode_segmap(image, nc=21):
  
    label_colors = np.array([(0, 0, 0),  # 0=background
               # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
               (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
               # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
               (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
               # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
               (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
               # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
               (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for l in range(0, nc):
        white = image == 255
        r[white], g[white], b[white] = 255, 255, 255
        idx = image == l
        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]

    rgb = np.stack([r, g, b], axis=2)
    return rgb


def showResult(model, dataiter, length):
    
    for i in range(length):
        image, label = dataiter.next()

        # original
        img = image.squeeze(0).permute(1, 2, 0).numpy()

        # fcn output
        output = model(image)
        om = torch.argmax(output.squeeze(), dim=0).detach().cpu().numpy()
        rgb = decode_segmap(om)

        #true mask
        label = np.asarray(label.squeeze(0))
        label_img = decode_segmap(label)

        fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(50, 50))
        ax0.imshow(img)
        ax1.imshow(rgb)
        ax2.imshow(label_img)

        
def LoadDict (dic_name, dic_type = 'dict'):
    with open('./DICT/' + dic_name + '.' + dic_type, 'rb') as file:
        dic = pickle.load(file)
    return dic
    
    
def SaveDict (dic, dic_name, dic_type = 'dict'):
    path = './DICT'
    if os.path.isdir(path) == False:
        os.mkdir(path)
    with open('./DICT/' + dic_name + '.' + dic_type, 'wb') as file:
        pickle.dump(dic, file)
    return 0
    
    
def plotCurve(dic, epoch = 100, typ = 'loss'):
    df = {'x': range(epoch)}
    for key in dic.keys():
        df[key] = dic[key]
    df=pd.DataFrame(df)

    palette = plt.get_cmap('Set1')

    # multiple line plot
    num=0
    plt.figure(figsize=(10,5))

    for column in df.drop('x', axis=1):
        num+=1
        plt.plot(df['x'], df[column], marker='', color=palette(num), linewidth=2, alpha=0.9, label=column,)

    # Add legend
    plt.legend(loc=1, ncol=1, fontsize = 'large')
    plt.xlabel("Epoch")
    plt.ylabel(typ)
