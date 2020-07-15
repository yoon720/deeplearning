import torch.nn as nn
import torchvision.models as models
from FCN.fcn_utils import *

class FCN32_Alex(nn.Module):
    def __init__(self, num_classes = 21, pretrained=models.alexnet(pretrained=True)):
        super(FCN32_Alex, self).__init__()
        pretrained = pretrained
        features = list(pretrained.features.children())
        classifier = list(pretrained.classifier.children())

        # Pad the input to enable small inputs and allow matching feature maps
        features[0].padding = (100, 100)

        # Enbale ceil in max pool, to avoid different sizes when upsampling
        for layer in features:
            if 'MaxPool' in layer.__class__.__name__:
                layer.ceil_mode = True

        self.features = nn.Sequential(*features[:])

        # Replace the FC layer with conv layers
        conv6 = nn.Conv2d(256, 4096, kernel_size=6)
        conv7 = nn.Conv2d(4096, 4096, kernel_size=1)
        output = nn.Conv2d(4096, num_classes, kernel_size=1)

        # Copy the weights from AlexNet's FC pretrained layers
        conv6.weight.data.copy_(classifier[1].weight.data.view(
            conv6.weight.data.size()))
        conv6.bias.data.copy_(classifier[1].bias.data)
        
        conv7.weight.data.copy_(classifier[4].weight.data.view(
            conv7.weight.data.size()))
        conv7.bias.data.copy_(classifier[4].bias.data)
        
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
        self.up_final.weight.requires_grad = False



    def forward(self, x):
        img_H, img_W = x.size()[2], x.size()[3]
        
        # Forward the image
        features = self.features(x)      
        output = self.output(features)
        
        # upsmapling
        up_final = self.up_final(output)

        # Remove the corresponding padded regions to the input img size
        final_value = up_final[:, :, 31: (31 + img_H), 31: (31 + img_W)].contiguous()
        return final_value

