import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from load_cifar10 import load_train_val, cifar10
from alexnet import AlexNet

# execute the following code if the file is executed as a script
if __name__ == "__main__":
    BATCH_SIZE = 600
    NUM_WORKER = 1
    NUM_CLASSES = 10
    START_EPOCH = 0
    NUM_EPOCHS = 75
    CHANNELS = 3
    LEARNING_RATE = 0.005
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    CKPT_FOLDER = "/content/drive/MyDrive/AlexNet-Pytorch/ckpt/"
    RESUME = True

    # load training and validation sets
    train_x, train_y, val_x, val_y = load_train_val()

    # define transformation
    transform_train = transforms.Compose([
        transforms.Resize((227, 227)),       
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        )
    ])

    transform_val = transforms.Compose([
        transforms.Resize((227, 227)),  
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        )
    ])

    # create dataset
    train_set = cifar10(data=train_x, label=train_y, transform=transform_train)
    val_set = cifar10(data=val_x, label=val_y, transform=transform_val)

    # create dataloader
    train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=NUM_WORKER)

    val_loader = DataLoader(dataset=val_set, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=NUM_WORKER)

    # load AlexNet
    model = AlexNet(num_classes=NUM_CLASSES, channels=CHANNELS)
    model.to(DEVICE)

    # loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, weight_decay=0.005, momentum=0.9)

    # load checkpoint
    if RESUME:
        last_ckpt = os.listdir(CKPT_FOLDER)[-1]
        checkpoint = torch.load(os.path.join(CKPT_FOLDER, last_ckpt))
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        START_EPOCH = checkpoint['epoch'] + 1
        print("Checkpoint has been loaded!")
        
    for epoch in range(START_EPOCH, NUM_EPOCHS):
        # training
        for i, (images, labels) in enumerate(train_loader):  
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
        
            # forward
            outputs = model(images)
            loss = criterion(outputs, labels)
        
            # backward and optimize
            optimizer.zero_grad()     # avoid gradient accumulation
            loss.backward()           # compute gradient
            optimizer.step()          # update parameters

        print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, NUM_EPOCHS, i+1, len(train_loader), loss.item()))
            
        # validation
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                del images, labels, outputs
    
            print('Accuracy of AlexNet on the validation set: {} %'.format(100 * correct / total)) 
        
        # save model parameters
        if (epoch + 1) % 5 == 0:
            save_state = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }

            ckpt_file = "AlexNet_epoch_{:03d}.pth.tar".format(epoch + 1)
            torch.save(save_state, os.path.join(CKPT_FOLDER, ckpt_file))
            print("Model parameters have been saved!")


