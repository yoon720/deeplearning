import torch
import torch.nn as nn
from utils import *

# MobileNet network class
class MobileNet(nn.Module):
    def __init__(self, a, p, img_shape = 32):
        super(MobileNet, self).__init__()
        
        def depthwiseSeparable(in_ch, out_ch, stride):
            layers = []
            layers += [nn.Conv2d(in_ch, in_ch, kernel_size=3, stride = stride, padding=1, groups=in_ch),
                       nn.BatchNorm2d(in_ch),
                       nn.ReLU(inplace = True),
                       nn.Conv2d(in_ch, out_ch, kernel_size=1),
                       nn.BatchNorm2d(out_ch),
                       nn.ReLU(inplace = True)]            
            return nn.Sequential(*layers)
        
        # Width multiplier
        width = []
        for i in [32, 64, 128, 256, 512, 1024]:
            width.append(int(i*a))
        
        # Resolution multiplier
        avg = int(img_shape*p//32)
            
        self.conv = nn.Sequential(nn.Conv2d(3, width[0], 3, stride = 2, padding = 1),
                                   nn.BatchNorm2d(width[0]),
                                   nn.ReLU(inplace = True))
        self.convdw = nn.Sequential(depthwiseSeparable(width[0], width[1], 1),
                                    depthwiseSeparable(width[1], width[2], 2),
                                    depthwiseSeparable(width[2], width[2], 1),
                                    depthwiseSeparable(width[2], width[3], 2),
                                    depthwiseSeparable(width[3], width[3], 1),
                                    depthwiseSeparable(width[3], width[4], 2),
                                    nn.Sequential(
                                        *[depthwiseSeparable(width[4], width[4], 1) for i in range(5)]),
                                    depthwiseSeparable(width[4], width[5], 2),
                                    depthwiseSeparable(width[5], width[5], 1))
        self.avgpool = nn.AvgPool2d(avg)
        self.fc = nn.Linear(width[5], 10)

        
    def forward(self, x):
        x = self.conv(x)
        x = self.convdw(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
    

# Conv MobileNet in Table 4
class convMN(nn.Module):
    def __init__(self):
        super(convMN, self).__init__()

        def conv(in_ch, out_ch, stride):
            return nn.Sequential(*[nn.Conv2d(in_ch, out_ch, 3, stride, padding = 1),
                                    nn.BatchNorm2d(out_ch),
                                    nn.ReLU(inplace = True)])
            
        self.conv = nn.Sequential(conv(3, 32, 2),
                                    conv(32, 64, 1),
                                    conv(64, 128, 2),
                                    conv(128, 128, 1),
                                    conv(128, 256, 2),
                                    conv(256, 256, 1),
                                    conv(256, 512, 2),
                                    nn.Sequential(
                                        *[conv(512, 512, 1) for i in range(5)]),
                                    conv(512,1024,2),
                                    conv(1024, 1024, 1))
        #self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(1024, 10)

        
    def forward(self, x):
        x = self.conv(x)
        #x = avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# MobileNet network class
class MobileNet_shall(nn.Module):
    def __init__(self, a, p, img_shape = 32):
        super(MobileNet_shall, self).__init__()
        
        def depthwiseSeparable(in_ch, out_ch, stride):
            layers = []
            layers += [nn.Conv2d(in_ch, in_ch, kernel_size=3, stride = stride, padding=1, groups=in_ch),
                       nn.BatchNorm2d(in_ch),
                       nn.ReLU(inplace = True),
                       nn.Conv2d(in_ch, out_ch, kernel_size=1),
                       nn.BatchNorm2d(out_ch),
                       nn.ReLU(inplace = True)]            
            return nn.Sequential(*layers)
        
        # Width multiplier
        width = []
        for i in [32, 64, 128, 256, 512, 1024]:
            width.append(int(i*a))
        
        # Resolution multiplier
        avg = img_shape*p//32
            
        self.conv = nn.Sequential(nn.Conv2d(3, width[0], 3, stride = 2, padding = 1),
                                   nn.BatchNorm2d(width[0]),
                                   nn.ReLU(inplace = True))
        self.convdw = nn.Sequential(depthwiseSeparable(width[0], width[1], 1),
                                    depthwiseSeparable(width[1], width[2], 2),
                                    depthwiseSeparable(width[2], width[2], 1),
                                    depthwiseSeparable(width[2], width[3], 2),
                                    depthwiseSeparable(width[3], width[3], 1),
                                    depthwiseSeparable(width[3], width[4], 2),
                                    depthwiseSeparable(width[4], width[5], 2),
                                    depthwiseSeparable(width[5], width[5], 1))
        self.avgpool = nn.AvgPool2d(avg)
        self.fc = nn.Linear(width[5], 10)

        
    def forward(self, x):
        x = self.conv(x)
        x = self.convdw(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x