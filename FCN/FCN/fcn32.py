import torch.nn as nn
import torchvision.models as models
from FCN.fcn_utils import *

class FCN32(nn.Module):
    def __init__(self, num_classes = 21, pretrained=models.vgg16(pretrained=True), fixed = False):
        super(FCN32, self).__init__()
        pretrained = pretrained
        features = list(pretrained.features.children())
        classifier = list(pretrained.classifier.children())

        # Pad the input to enable small inputs and allow matching feature maps
        features[0].padding = (100, 100)

        # Enbale ceil in max pool, to avoid different sizes when upsampling
        for layer in features:
            if 'MaxPool' in layer.__class__.__name__:
                layer.ceil_mode = True

        # Extract pool3, pool4 and pool5 from the VGG net
        self.pool3 = nn.Sequential(*features[:17])
        self.pool4 = nn.Sequential(*features[17:24])
        self.pool5 = nn.Sequential(*features[24:])

        # Replace the FC layer with conv layers
        conv6 = nn.Conv2d(512, 4096, kernel_size=7)
        conv7 = nn.Conv2d(4096, 4096, kernel_size=1)
        output = nn.Conv2d(4096, num_classes, kernel_size=1)

        # Copy the weights from VGG's FC pretrained layers
        conv6.weight.data.copy_(classifier[0].weight.data.view(
            conv6.weight.data.size()))
        conv6.bias.data.copy_(classifier[0].bias.data)
        
        conv7.weight.data.copy_(classifier[3].weight.data.view(
            conv7.weight.data.size()))
        conv7.bias.data.copy_(classifier[3].bias.data)
        
        # Get the outputs
        self.output = nn.Sequential(conv6, nn.ReLU(inplace=True), nn.Dropout(),
                                    conv7, nn.ReLU(inplace=True), nn.Dropout(), 
                                    output)

        # upsampling 
        self.up_final = nn.ConvTranspose2d(num_classes, num_classes, 
                                            kernel_size=64, stride=32, bias=False)

        #initialize upsampling to bilinear interpolation and fix
        self.up_final.weight.data.copy_(
            upsampling_init_weight(num_classes, num_classes, 64))
        
        # Fix final layer deconv filter
        self.up_final.weight.requires_grad = False
        
        # zero-initialize the class scoring layer
        self.output[6].weight.data.fill_(0)
        
        #fcn32-fixed
        if fixed:
            for layer in [self.pool3, self.pool4, self.pool5]:
                if 'Conv2d' in layer.__class__.__name__:
                    layer.weight.requires_grad = False
            self.output[0].weight.requires_grad = False  #conv6
            self.output[3].weight.requires_grad = False  #conv7
            
        
    def forward(self, x):
        img_H, img_W = x.size()[2], x.size()[3]
        
        # Forward the image
        pool3 = self.pool3(x)
        pool4 = self.pool4(pool3)
        pool5 = self.pool5(pool4)
        
        output = self.output(pool5)
        
        # upsmapling
        up_final = self.up_final(output)

        # Remove the corresponding padded regions to the input img size
        final_value = up_final[:, :, 31: (31 + img_H), 31: (31 + img_W)].contiguous()
        return final_value

