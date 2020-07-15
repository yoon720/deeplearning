import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pickle, os, random



def imshow(real, fake):
    if real.device.type == 'cuda':
        real = real.cpu().detach()
    if fake.device.type == 'cuda':
        fake = fake.cpu().detach()
    
    real_np = real.mul(0.5).add(0.5).numpy()
    fake_np = fake.mul(0.5).add(0.5).numpy()

    fig = plt.figure(figsize = (8,4))
    ax = fig.add_subplot(121)
    bx = fig.add_subplot(122)
    
    ax.imshow(np.transpose(real_np.squeeze(), (1, 2, 0)))
    bx.imshow(np.transpose(fake_np.squeeze(), (1, 2, 0)))
    plt.show()


def weights_init(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            nn.init.normal_(m.bias.data, 0.0, 0.02)
        elif isinstance(m, nn.ConvTranspose2d):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            nn.init.normal_(m.bias.data, 0.0, 0.02)
        
        
def decayLR(epoch):
    if epoch < 100:
        return 1
    else:
        return 1-0.01*(epoch-100+1)
    
        
def buffer(buffer_list, img):
    if len(buffer_list) < 50:
        buffer_list.append(img)
        return_img = img
    else:
        p = random.uniform(0, 1)
        rand_idx = random.randint(0, 49)
        tmp = buffer_list[rand_idx]
        buffer_list[rand_idx] = img  # buffer에 새 이미지를 업데이트
        if p > 0.5:
            return_img = tmp  # 확률이 0.5보다 클때 buffer image를 return
        else:
            return_img = img  # 그렇지 않으면 현재 이미지를 사용
    return return_img 

        
def save_model(G, F, Dx, Dy, name):
    if not os.path.exists('DICT/'):
        os.makedirs('DICT/')
    torch.save(G.state_dict(), './DICT/' + name + '_G.pkl')
    torch.save(F.state_dict(), './DICT/' + name + '_F.pkl')
    torch.save(Dx.state_dict(), './DICT/' + name + '_Dx.pkl')
    torch.save(Dy.state_dict(), './DICT/' + name + '_Dy.pkl')
    
    
def load_model(G, F, Dx, Dy, name):
    G_model_path = os.path.join(os.getcwd(), 'DICT/' + name + '_G.pkl')
    F_model_path = os.path.join(os.getcwd(), 'DICT/' + name + '_F.pkl')
    Dx_model_path = os.path.join(os.getcwd(), 'DICT/' + name + '_Dx.pkl')
    Dy_model_path = os.path.join(os.getcwd(), 'DICT/' + name + '_Dy.pkl')
    
    G.load_state_dict(torch.load(G_model_path))
    F.load_state_dict(torch.load(F_model_path))
    Dx.load_state_dict(torch.load(Dx_model_path))
    Dy.load_state_dict(torch.load(Dy_model_path))
    
    
def save_dict (dic, dic_name):
    with open(dic_name + '.pkl', 'wb') as file:
        pickle.dump(dic, file)
    return 0


def load_dict (dic_name):
    with open(dic_name + '.pkl', 'rb') as file:
        dic = pickle.load(file)
    return dic