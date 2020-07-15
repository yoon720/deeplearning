"""
Implementation of VGG Network for CIFAR10

Adjust FC layer for CIFAR10 images which are 32x32
* reduced the number of FC layers from 3 to 2
* changed the number of parameters of fc1 : 512x7x7 => 512
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import time, copy
from utils import *

# VGG network class
# also can be used to create larger network
class VGG(nn.Module):
    def __init__(self, conf_list, config):
        super(VGG, self).__init__()
        self.convNet = self.config_layers(conf_list[config])
        self.fc1 = nn.Sequential(
            nn.Linear(512, 512),   #cifar10에 맞게 조절하기
            nn.ReLU(inplace=True), 
            nn.Dropout(0.5))
        self.fc2 = nn.Sequential(
            nn.Linear(512, 512),   #cifar10에 맞게 조절하기
            nn.ReLU(inplace=True), 
            nn.Dropout(0.5))
        self.fc3 = nn.Linear(512, 10)

        
    def forward(self, x):
        x = self.convNet(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
        
    def config_layers(self, config):
        layers = []
        input_ch = 3
        for layer in config:
            #print(l)
            if layer == 'MP':
                layers += [nn.MaxPool2d(2, 2)]
            elif isinstance(layer, tuple):
                layers += [nn.Conv2d(input_ch, layer[1], layer[0], padding = (layer[0]-1)//2),
                           nn.ReLU(inplace = True)]
                input_ch = layer[1]
            else:
                layers += [ nn.Conv2d(input_ch, layer, 3, padding = 1),
                           nn.ReLU(inplace = True)]
                input_ch = layer
            
        return nn.Sequential(*layers)
    

    
#model training session
def train(model, device, train_loader, length, criterion, optimizer):
    model.train()

    running_loss = 0.0
    running_corrects = 0

    
    for i, (inputs, labels) in enumerate(train_loader):
        if device == 'cuda':
            inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        _, preds = outputs.max(1)
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        
        # if you want to check batch loss & acc
#         if i % 20 == 19:
#             print("Batch %d) loss: %f correct: %d" %(i, float(loss.data), running_corrects))

    epoch_loss = running_loss / length
    epoch_acc = running_corrects.double() / length
    
    print('LR: {}'.format(get_lr(optimizer)))
    print('Train Loss: {:.4f} Acc: {:.4f} '.format(epoch_loss, epoch_acc))

    
# for validate and test
def validate(model, device, val_loader, length, criterion, mode = 'val'):
    # switch to evaluate mode
    model.eval()

    running_loss = 0.0
    running_corrects = 0
    
    with torch.set_grad_enabled(False):
        for i, (inputs, labels) in enumerate(val_loader):
            if device == 'cuda':
                inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = outputs.max(1)
            loss = criterion(outputs, labels)

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            #print(labels.data)
        
        epoch_loss = running_loss / length
        epoch_acc = running_corrects.double() / length
        
        if mode == 'val':
            print('Val Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
        else:
            print('Test Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
    
    return epoch_loss, epoch_acc, copy.deepcopy(model.state_dict())
    
    
def train_model(model, device, train_loader, val_loader, lengths, num_epochs=50):
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=7, threshold=0.001)

    if device == 'cuda':
        model.to(device)
        cudnn.benchmark = True
        criterion = criterion.cuda()
    
    since = time.time()
    
    VAL_ACC = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        
        train(model, device, train_loader, lengths[0], criterion, optimizer)
        
        # evaluate on validation set
        val_loss, val_acc, epoch_model = validate(model, device, val_loader, lengths[1], criterion)
        VAL_ACC.append(val_acc)
        
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
        
        scheduler.step(float(val_loss))    
        print()
    
    print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    print(model.load_state_dict(best_model_wts))
    return best_model_wts, VAL_ACC
