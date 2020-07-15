import copy
import numpy as np
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from utils import *

def dataloader(rho, dataset = 'CIFAR10'):
    # Resize image using resolution multiplier
    transform_train = transforms.Compose([
        transforms.Resize(int(96*rho)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    if dataset == 'STL10':
        train_data = torchvision.datasets.STL10(root='~/data', split = 'train', download = True, transform = transform_train)
    else:
        #default dataset is CIFAR10
        train_data = torchvision.datasets.CIFAR10(root='~/data', train = True, download = True, transform = transform_train)


    # randomly split train and val
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(0.2 * num_train))
    np.random.shuffle(indices)

    train_idx, val_idx = indices[split:], indices[:split]

    train_sampler = data.sampler.SubsetRandomSampler(train_idx)
    val_sampler = data.sampler.SubsetRandomSampler(val_idx)

    lengths = [len(train_sampler), len(val_sampler)]

    # dataloader
    train_loader = data.DataLoader(train_data, sampler=train_sampler, batch_size=512, num_workers = 2)
    val_loader = data.DataLoader(train_data, sampler=val_sampler, batch_size=512, num_workers = 2)
    return train_loader, val_loader, lengths
    
    
# train session   
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
    return epoch_loss, epoch_acc
    
    
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