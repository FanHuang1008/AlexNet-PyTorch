import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from load_cifar10 import load_test, cifar10
from alexnet import AlexNet

# execute the following code if the file is executed as a script
if __name__ == "__main__":
    BATCH_SIZE = 600
    NUM_WORKER = 1
    NUM_CLASSES = 10
    CHANNELS = 3
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    CKPT_FOLDER = "/content/drive/MyDrive/AlexNet-Pytorch/ckpt/"

    # load test set
    test_x, test_y = load_test()

    # define transformation
    transform_test = transforms.Compose([
        transforms.Resize((227, 227)),  
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        )
    ])

    # create dataset and dataloader
    test_set = cifar10(data=test_x, label=test_y, transform=transform_test)
    test_loader = DataLoader(dataset=test_set, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=NUM_WORKER)

    # load AlexNet
    model = AlexNet(num_classes=NUM_CLASSES, channels=CHANNELS)
    model.to(DEVICE)

    # load checkpoint
    last_ckpt = os.listdir(CKPT_FOLDER)[-1]
    checkpoint = torch.load(os.path.join(CKPT_FOLDER, last_ckpt))
    model.load_state_dict(checkpoint['state_dict'])
    print("Checkpoint has been loaded!")

    # testing
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            del images, labels, outputs

        print('Accuracy of AlexNet on the test set: {} %'.format(100 * correct / total)) 


