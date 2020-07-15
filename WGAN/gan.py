import torch
import torch.nn as nn

# DCGAN generator
class DCGAN_G(nn.Module):
    def __init__(self):
        super(DCGAN_G, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(100, 1024, kernel_size=4, stride=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh())
        
    def forward(self, x):
        return self.deconv(x)

# DCGAN generator without batch normalization
class DCGAN_BN_G(nn.Module):
    def __init__(self):
        super(DCGAN_BN_G, self).__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(100, 1024, kernel_size=4, stride=1, bias=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh())
        
    def forward(self, x):
        return self.deconv(x)
    
    
# MLP 4
class MLP_G(nn.Module):
    def __init__(self):
        super(MLP_G, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(100, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 64*64*3),
            nn.Tanh())
        
    def forward(self, x):
        x = x.view(batch_size, 100)
        return self.layers(x)

# MLP 5
class MLP_G2(nn.Module):
    def __init__(self):
        super(MLP_G2, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(100, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, 64*64*3),
            nn.Tanh())
        
    def forward(self, x):
        x = x.view(batch_size, 100)
        return self.layers(x)
    

# DCGAN discriminator
class DCGAN_D(nn.Module):
    def __init__(self):
        super(DCGAN_D, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(1024, 1, kernel_size=4, stride=1, bias=False))
        
    def forward(self, x):
        return self.conv(x)