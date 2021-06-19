import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from PIL import Image
import os


class DataFeeder(Dataset):
    def __init__(self, train_dir, label_dir):
        super(DataFeeder, self).__init__()
        self.train_dir = train_dir
        self.label_dir = label_dir
        self.img_rows = 400
        self.img_cols = 400

        sub_dirs = os.listdir(train_dir)
        sub_dirs.sort(key=lambda x: int(x))
        train_image_path_list = []
        label_image_path_list = []
        for sub_dir in sub_dirs:
            train_sub_dir_path = os.path.join(self.train_dir, sub_dir)
            label_sub_dir_path = os.path.join(self.label_dir, sub_dir)
            train_images_name = os.listdir(train_sub_dir_path)
            train_images_name.sort(key=lambda x: int(x.split('.')[0]))
            label_images_name = os.listdir(label_sub_dir_path)
            label_images_name.sort(key=lambda x: int(x.split('.')[0]))

            for image_name in train_images_name:
                train_image_path = os.path.join(train_sub_dir_path, image_name)
                label_image_path = os.path.join(label_sub_dir_path, image_name)
                train_image_path_list.append(train_image_path)
                label_image_path_list.append(label_image_path)
        self.train_image_path_list = train_image_path_list
        self.label_image_path_list = label_image_path_list

    def __getitem__(self, index):
        train_path = self.train_image_path_list[index]
        label_path = self.label_image_path_list[index]

        image = Image.open(train_path)
        image = np.array(image)
        image = np.reshape(image, (self.img_rows, self.img_cols, 1))
        image = image.astype("float32")
        image /= 255.0

        label = Image.open(label_path)
        label = np.array(label)
        label = np.reshape(label, (self.img_cols, self.img_rows, 1))
        label = label.astype('float32')
        label /= 255.0

        image_sample = ToTensor()(image).float()
        label_sample = ToTensor()(label).float()
        return image_sample, label_sample

    def __len__(self):
        return len(self.label_image_path_list)


