import torch.nn as nn
import torchvision.models as models
from FCN.fcn_utils import *

class FCN32_google(nn.Module):
    def __init__(self, num_classes = 21, pretrained=models.googlenet(pretrained=True)):
        super(FCN32_google, self).__init__()
        pretrained = pretrained
        
        # Discard final avgpool
        del pretrained.avgpool
        del pretrained.dropout
        
        classifier = pretrained.fc
        del pretrained.fc
        
        features = list(pretrained.children())

        # Pad the input to enable small inputs and allow matching feature maps
        features[0].conv.padding = (100, 100)

        # Enbale ceil in max pool, to avoid different sizes when upsampling
        for layer in features:
            if 'MaxPool' in layer.__class__.__name__:
                layer.ceil_mode = True

        self.features = nn.Sequential(*features[:])

        # Replace the FC layer with conv layers
        conv6 = nn.Conv2d(1024, 1000, kernel_size=1)
        output = nn.Conv2d(1000, num_classes, kernel_size=1)

        # Copy the weights from GoogLeNet's FC pretrained layers
        conv6.weight.data.copy_(classifier.weight.data.view(
            conv6.weight.data.size()))
        conv6.bias.data.copy_(classifier.bias.data)
        
        # Get the outputs
        self.output = nn.Sequential(conv6, nn.ReLU(inplace=True), nn.Dropout(),
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
        self.output[3].weight.data.fill_(0)
                    
        
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

