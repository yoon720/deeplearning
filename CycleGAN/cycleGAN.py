import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, num_res = 6):
        super(Generator, self).__init__()
        global n_res
        n_res = num_res
        
        self.conv = nn.Sequential(*(self.conv_block(3, 64, 7, 1) + self.conv_block(64, 128, 3, 2) + self.conv_block(128, 256, 3, 2)))
        
        self.res1 = nn.Sequential(*self.res_block(256))
        self.res2 = nn.Sequential(*self.res_block(256))
        self.res3 = nn.Sequential(*self.res_block(256))
        self.res4 = nn.Sequential(*self.res_block(256))
        self.res5 = nn.Sequential(*self.res_block(256))
        self.res6 = nn.Sequential(*self.res_block(256))
        
        if num_res == 9:
            self.res7 = nn.Sequential(*self.res_block(256))
            self.res8 = nn.Sequential(*self.res_block(256))
            self.res9 = nn.Sequential(*self.res_block(256))
            
        self.deconv = nn.Sequential(*(self.deconv_block(256, 128)
                                     + self.deconv_block(128, 64)))
        self.conv2 = nn.Sequential(nn.ReflectionPad2d(3),
                                   nn.Conv2d(64, 3, kernel_size=7, stride = 1, padding=0),
                                   nn.Tanh())
        

    def conv_block(self, in_ch, out_ch, kernel_size, stride):
        layers = [nn.ReflectionPad2d(kernel_size//2),
                  nn.Conv2d(in_ch, out_ch, kernel_size, stride = stride),
                  nn.InstanceNorm2d(out_ch),
                  nn.ReLU(True)]
        return layers

    def res_block(self, in_ch):
        layers = [nn.ReflectionPad2d(1),
                  nn.Conv2d(in_ch, in_ch, kernel_size=3, stride = 1, padding=0, bias=True),
                  nn.InstanceNorm2d(in_ch)]
        return self.conv_block(in_ch, in_ch, 3, 1) + layers

    def deconv_block(self, in_ch, out_ch):
        layers = [nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True),
                  nn.InstanceNorm2d(out_ch),
                  nn.ReLU(True)]
        return layers


    def forward(self, x):
        x = self.conv(x)
        x = x + self.res1(x)
        x = x + self.res2(x)
        x = x + self.res3(x)
        x = x + self.res4(x)
        x = x + self.res5(x)
        x = x + self.res6(x)
        
        if n_res == 9:
            x = x + self.res7(x)
            x = x + self.res8(x)
            x = x + self.res9(x)
        
        x = self.deconv(x)
        return self.conv2(x)
    
    
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=4, stride = 2, padding=1),
                                   nn.LeakyReLU(0.2, inplace=True))
        self.conv2 = nn.Sequential(*self.conv_block(64, 128, 2))
        self.conv3 = nn.Sequential(*self.conv_block(128, 256, 2))
        self.conv4 = nn.Sequential(*self.conv_block(256, 512, 2))
        self.conv5 = nn.Conv2d(512, 1, kernel_size=4, stride = 1, padding=1)

        
    def conv_block(self, in_ch, out_ch, stride):
        layers = [nn.Conv2d(in_ch, out_ch, kernel_size=4, stride = stride, padding=1),
                  nn.InstanceNorm2d(out_ch),
                  nn.LeakyReLU(0.2, inplace=True)]
        return layers
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return self.conv5(x)