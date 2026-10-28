# coding:utf-8
import os
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import numpy as np
import tifffile as tiff
import cv2
from util.augmentation import RandomFlip, RandomBrightness, RandomNoise,RandomScratch,RandomBlur,RandomDistortion,RandomRoll
# from augmentation import RandomFlip, RandomCrop, RandomCropOut, RandomBrightness, RandomNoise
# from util.image_process import resize_img, write_img
# from image_process import resize_img


class Forest_dataset(Dataset):

    def __init__(self, map_dir, map_seffix, label_dir, label_seffix, have_label, class_num = 2, input_h=512, input_w=512 ,transform=[]):
        super(Forest_dataset, self).__init__()

        map_set = []
        label_set = []
        labeltype_length = len(label_seffix)
        listfile = os.listdir(label_dir)
        for path in listfile:
            if path[(-labeltype_length):].upper() != label_seffix.upper():
                continue
            map_set.append(map_dir + path)
            label_set.append(label_dir + path)

        self.map_set = map_set 
        self.label_set = label_set
        self.input_h   = input_h
        self.input_w   = input_w
        self.transform = transform
        self.class_num = class_num
        self.is_train  = have_label
        self.n_data    = len(self.map_set)
        print(len(self.map_set))


    def read_image(self, name, folder):
        if folder =='images':
            # print(name)
            image = tiff.imread(name)
            image = image[np.newaxis,:,:]
            # if image.shape[2]<image.shape[0]:
            #     image = image.transpose(2,0,1)
        else :
            # print(name)
            image = tiff.imread(name)
            image = np.squeeze(image)
            image[image>1] = 1
            # print(image.max())
        image.flags.writeable = True
        return image

    def get_train_item(self, index):
        map_name  = self.map_set[index]
        name = map_name.split('/')[-1]
        label_name = self.label_set[index]
        image = self.read_image(map_name, 'images')
        label = self.read_image(label_name, 'labels')
        # if image.shape[2] > image.shape[0]:
        #     image = image.transpose(1, 2, 0)
        # image_show = image[0]
        # cv2.imshow('before',image_show)
        # cv2.waitKey(0)
        if self.transform !=None:
            for func in self.transform:
                image, label = func(image, label)
        # image = cv2.resize(image,(self.input_h, self.input_w))
        # label = cv2.resize(label,(self.input_h, self.input_w))
        # if image.shape[0] > image.shape[2]:
        #     image = image.transpose(2, 0, 1)
        # print(image.shape)
        # print(label.shape)
        # image = resize_img(image, self.input_h, self.input_w)
        # label = resize_img(label, self.input_h, self.input_w)
        # image_show = image[0]
        # cv2.imshow('after', image_show)
        # cv2.waitKey(0)
        # cv2.imwrite('show.jpg',image_show)
        # image = image[:512,:512]
        # label = label[:512, :512]
        # print(image.shape)
        # print(label.shape)
        # image_show = image[0]
        # cv2.imshow('image', image_show)
        # cv2.waitKey(0)
        # image_show = label*255
        # cv2.imshow('label', image_show)
        # cv2.waitKey(0)
        image = np.array(image, dtype="int32")
        label = np.array(label, dtype="int64")
        # exit()

        return torch.tensor(image), torch.tensor(label), name

    def get_test_item(self, index):
        map_name = self.map_set[index]
        name = map_name.split('/')[-1]
        image = self.read_image(name, 'images')
        # image = np.asarray(Image.fromarray(image).resize((self.input_w, self.input_h)), dtype=np.float32).transpose((2,0,1))/255

        return torch.tensor(image), name


    def __getitem__(self, index):

        if self.is_train is True:
            return self.get_train_item(index)
        else: 
            return self.get_test_item (index)

    def __len__(self):
        return self.n_data

if __name__ == '__main__':
    train_map = 'd:/switzerland/data/v5/train/map/'
    image = 'd:/switzerland/data/v5/train/label/'
    # augmentation_methods = [RandomBrightness(
    #     bright_range=0.1, prob=0), RandomNoise(noise_range=25, prob=1)]
    augmentation_methods = [RandomFlip(prob=0), RandomBrightness(bright_range=0.5, prob=0), RandomNoise(noise_range=25, prob=1), ]

    x = Forest_dataset(map_dir=train_map, map_seffix='.tif', label_dir=image, label_seffix='.tif', have_label=True, transform=augmentation_methods)
    image,label,name = x.get_train_item(1)
    print (image.shape)
    print(image.max(),image.min())
    print(label.shape)
    print(label.max(),label.min())
