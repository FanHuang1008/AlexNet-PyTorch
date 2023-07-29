import pickle
import numpy as np
from PIL import Image

from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset
import torchvision.transforms as transforms




def load_cifar10(file_name):
    path = '/content/drive/MyDrive/cifar10/'
    with open(path+file_name, 'rb') as fo:
       data_dict = pickle.load(fo, encoding='bytes')

    images = data_dict[b'data']
    labels = data_dict[b'labels']

    return images, labels

def split_rgb(batch_images):
    assert batch_images.shape[1] == 3072  # (32, 32) x 3

    r = batch_images[:, 0:1024].reshape(batch_images.shape[0], 32, 32, 1)
    g = batch_images[:, 1024:2048].reshape(batch_images.shape[0], 32, 32, 1)
    b = batch_images[:, 2048:3072].reshape(batch_images.shape[0], 32, 32, 1)
    batch_images_rgb = np.concatenate([r,g,b], -1)

    return batch_images_rgb

def load_train_val():
    batch_1, labels_1 = load_cifar10('data_batch_1')
    batch_2, labels_2 = load_cifar10('data_batch_2')
    batch_3, labels_3 = load_cifar10('data_batch_3')
    batch_4, labels_4 = load_cifar10('data_batch_4')
    batch_5, labels_5 = load_cifar10('data_batch_5')

    train_x = np.concatenate([batch_1,batch_2,batch_3,batch_4,batch_5], 0)
    train_y = np.concatenate([labels_1,labels_2,labels_3,labels_4,labels_5], 0)

    train_x = split_rgb(train_x)  
    # train_y is already an np.array

    # split training and validation sets
    train_x, val_x, train_y, val_y = train_test_split(
        train_x, train_y, test_size=0.2, random_state=21)

    return train_x, train_y, val_x, val_y

def load_test():
    test_x, test_y = load_cifar10('test_batch')

    test_x = split_rgb(test_x)
    test_y = np.array(test_y) 

    return test_x, test_y

class cifar10(Dataset):
    def __init__(self, data, label, transform=None):
        self.data = data
        self.label = label
        self.transform = transform
        self.image_shape = data.shape
    
    def __len__(self):
        return len(self.data)


    def __getitem__(self, index):
        image = Image.fromarray(self.data[index])
        label = self.label[index]
        if self.transform is not None:
            image = self.transform(image)
        else:
            toTensor = transforms.ToTensor(image)
            image = toTensor(image)
        return image, label