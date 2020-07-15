import torch.nn as nn
import torchvision.models as models
from FCN.fcn_utils import *

class FCN8(nn.Module):
    def __init__(self, num_classes = 21, pretrained=models.vgg16(pretrained=True)):
        super(FCN8, self).__init__()
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

        # Adjust the depth of pool3 and pool4 to num_classes
        self.adj_pool3 = nn.Conv2d(256, num_classes, kernel_size=1)
        self.adj_pool4 = nn.Conv2d(512, num_classes, kernel_size=1)
        
        # Initialize adj layers
        self.adj_pool3.weight.data.fill_(0)
        self.adj_pool3.bias.data.fill_(0)
        self.adj_pool4.weight.data.fill_(0)
        self.adj_pool4.bias.data.fill_(0)
        
        # Replace the FC layer of VGG with conv layers
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

        # Make 3 upsampling layers
        self.up_output = nn.ConvTranspose2d(num_classes, num_classes,
                                            kernel_size=4, stride=2, bias=False)
        self.up_pool4_out = nn.ConvTranspose2d(num_classes, num_classes, 
                                            kernel_size=4, stride=2, bias=False)
        self.up_final = nn.ConvTranspose2d(num_classes, num_classes, 
                                            kernel_size=16, stride=8, bias=False)

        # Initialize upsampling layers to bilinear interpolation
        self.up_output.weight.data.copy_(
            upsampling_init_weight(num_classes, num_classes, 4))
        self.up_pool4_out.weight.data.copy_(
            upsampling_init_weight(num_classes, num_classes, 4))
        self.up_final.weight.data.copy_(
            upsampling_init_weight(num_classes, num_classes, 16))
        self.up_final.weight.requires_grad = False
        
        # zero-initialize the class scoring layer
        self.output[6].weight.data.fill_(0)
        
        
    def forward(self, x):
        img_H, img_W = x.size()[2], x.size()[3]
        
        # Forward the image
        pool3 = self.pool3(x)
        pool4 = self.pool4(pool3)
        pool5 = self.pool5(pool4)

        # Get the outputs and upsmaple them
        output = self.output(pool5)
        up_output = self.up_output(output)

        # Adjust pool4 and add the uped-outputs to pool4
        adjstd_pool4 = self.adj_pool4(0.01 * pool4)
        add_out_pool4 = self.up_pool4_out(adjstd_pool4[:, :, 5: (5 + up_output.size()[2]), 
                                            5: (5 + up_output.size()[3])]
                                           + up_output)

        # Adjust pool3 and add it to the uped last addition
        adjstd_pool3 = self.adj_pool3(0.0001 * pool3)
        final_value = self.up_final(adjstd_pool3[:, :, 9: (9 + add_out_pool4.size()[2]), 9: (9 + add_out_pool4.size()[3])]
                                 + add_out_pool4)

        # Remove the corresponding padded regions to the input img size
        final_value = final_value[:, :, 31: (31 + img_H), 31: (31 + img_W)].contiguous()
        return final_value
