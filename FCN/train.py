import copy, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn


from utils import *

def train(model, device, train_loader, criterion, optimizer):
    model.train()

    running_loss = 0.0
    r_pacc, r_macc, r_mIU, r_fIU = 0, 0, 0, 0

    loss_sum = torch.zeros(1)
    pacc, macc, mIU, fIU = 0, 0, 0, 0
    optimizer.zero_grad()
    
    for i, (image, label) in enumerate(train_loader):
        if device == 'cuda':
            image, label = image.to(device), label.to(device)
            
        output = model(image)
        
        #calculate metrics
        out_mask = torch.argmax(output.cpu().squeeze(), dim=0).numpy()
        target_mask = label.cpu().squeeze()
        pacc += pixel_accuracy(out_mask, target_mask)/20
        macc += mean_accuracy(out_mask, target_mask)/20
        mIU += mean_IU(out_mask, target_mask)/20
        fIU += frequency_weighted_IU(out_mask, target_mask)/20
        
        #calculate loss
        o = torch.squeeze(output, 0).reshape(21, -1).permute(1, 0)
        l = torch.squeeze(label.reshape(1, -1).permute(1, 0), 1)
        loss = criterion(o, l)/20
        loss.backward()
        loss_sum += loss
        
        # Implement mini-batch by accumulate 
        # update gradient every 20 image
        if i%20 == 19 or i == 1111:
            optimizer.step()
            print("Batch %d)\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f" %(i//20, loss_sum, pacc, macc, mIU, fIU))
            
            running_loss += loss_sum.item() * 20
            r_pacc += pacc * 20
            r_macc += macc * 20
            r_mIU += mIU * 20
            r_fIU += fIU * 20
            
            # batch initialize
            loss_sum, pacc, macc, mIU, fIU = 0, 0, 0, 0, 0
            optimizer.zero_grad()
            
            # show output mask
#             img = decode_segmap(out_mask)
#             plt.imshow(img); plt.axis('off'); plt.show()
        
    epoch_metric = (running_loss, r_pacc, r_macc, r_mIU, r_fIU)
    epoch_metric = [m/1112 for m in epoch_metric]
    
    print('Train:\t%s' %epoch_metric)
    return epoch_metric


def validate(model, device, val_loader, criterion, mode = 'val'):
    # switch to evaluate mode
    model.eval()

    running_loss = 0.0
    r_pacc, r_macc, r_mIU, r_fIU = 0, 0, 0, 0
    pacc, macc, mIU, fIU = 0, 0, 0, 0
    
    for i, (image, label) in enumerate(train_loader):
        if device == 'cuda':
            image, label = image.to(device), label.to(device)
        output = model(image)
        
        #calculate metrics
        out_mask = torch.argmax(output.cpu().squeeze(), dim=0).numpy()
        target_mask = label.cpu().squeeze()
        pacc = pixel_accuracy(out_mask, target_mask)
        macc = mean_accuracy(out_mask, target_mask)
        mIU = mean_IU(out_mask, target_mask)
        fIU = frequency_weighted_IU(out_mask, target_mask)
        
        o = torch.squeeze(output, 0).reshape(21, -1).permute(1, 0)
        l = torch.squeeze(label.reshape(1, -1).permute(1, 0), 1)
        loss = criterion(o, l)
        
        if i % 100 == 99:
            print("Val %d)\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f" %(i, loss, pacc, macc, mIU, fIU))
        # statistics
        running_loss += loss.item()
        r_pacc += pacc
        r_macc += macc
        r_mIU += mIU
        r_fIU += fIU

    #epoch_loss = running_loss / 1111
    epoch_metric = (running_loss, r_pacc, r_macc, r_mIU, r_fIU)
    epoch_metric = [m/1111 for m in epoch_metric]
    
    if mode == 'val':
        print('Val metric:\t%s' %epoch_metric)
    else:
        print('Test metric:\t%s' %epoch_metric)
    
    return epoch_metric  #, copy.deepcopy(model.state_dict())


def train_model(model, device, train_loader, val_loader, lr = 1e-4, dict_name = 'fcn32', num_epochs=50):
    
    criterion = nn.CrossEntropyLoss(ignore_index = 255)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5**(-4))

    if device == 'cuda':
        model.to(device)
        cudnn.benchmark = True
        criterion = criterion.cuda()
    
    since = time.time()
    M = {'train' : [], 'val' : []}
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        
        train_metric = train(model, device, train_loader, criterion, optimizer)
        M['train'].append(train_metric)
        
        # evaluate on validation set
        val_metric = validate(model, device, val_loader, criterion)
        M['val'].append(val_metric)
        
        if val_metric[1] > best_acc:
            best_acc = val_metric[1]
            best_model_wts = copy.deepcopy(model.state_dict())
        
        if epoch % 20 == 19:
            SaveDict(model.state_dict(), dict_name+'_epoch'+str(epoch))
        print()
    
    print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    print(model.load_state_dict(best_model_wts))
    return best_model_wts, M