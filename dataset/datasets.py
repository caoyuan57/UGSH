import itertools
import torch
import json
import os
import numpy as np
from PIL import Image
import torchvision
from utils import select_idxs



def read_json(file_path):
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
    return data

def get_image_names(json_data):
    image_names_list = []
    for image_information in json_data['images']:
        image_names_list.append(image_information['filename'])
    return image_names_list

json_file_path = '/home/hzj/my-code/my-code-0.4/data/RSICD.json'
json_data = read_json(json_file_path)
# print(json_data['images'][1]['filename'])
image_names_list = get_image_names(json_data)
images_folder_path = '/home/hzj/my-code/my-code-0.4/data/RSICD_images/'



class AbstractDataset(torch.utils.data.Dataset):

    def __init__(self, images, captions, labels, idxs, captions_aug=None, images_aug=None, seed=42):
        self.seed = seed
        self.image_replication_factor = 1  # default value, how many times we need to replicate image

        self.images = images
        self.captions = captions
        self.labels = labels

        self.captions_aug = captions_aug
        self.images_aug = images_aug

        self.idxs = np.array(idxs[0])
        self.idxs_cap = np.array(idxs[1])

    def __getitem__(self, index):
        return

    def __len__(self):
        return


class DatasetQuadrupletAugmentedTxtImg(AbstractDataset):
    """
    Class for dataset representation.

    Quadruplet dataset sample - img-img-txt-txt
    """

    def __init__(self, images, captions, labels, idxs, captions_aug=None, images_aug=None, seed=42):
        """
        Initialization

        :param images: image embeddings vector
        :param captions: captions embeddings vector
        :param labels: labels vector
        """
        super().__init__(images, captions, labels, idxs, captions_aug, images_aug, seed)

        caption_idxs = select_idxs(len(self.captions), 1, 5, seed=self.seed)[0]
        self.captions = self.captions[caption_idxs]
        self.captions_aug = self.captions_aug[caption_idxs]
        self.idxs_cap = self.idxs_cap[caption_idxs]

        # self.image_names_list = image_names_list
        # self.images_folder_path = images_folder_path
        # self.toTensor = torchvision.transforms.ToTensor()

    def __getitem__(self, index):
        """
        Returns a tuple (img1, img2, txt1, txt2, label)

        :param index: index of sample
        :return: tuple (img1, img2, txt1, txt2, label)
        """
        # image_name = self.image_names_list[index]
        # raw_image = Image.open(os.path.join(self.images_folder_path, image_name))
        return (
            index,
            (self.idxs[index], self.idxs[index], self.idxs_cap[index], self.idxs_cap[index]),
            # self.toTensor(raw_image),
            torch.from_numpy(self.images[index].astype('float32')),
            torch.from_numpy(self.images_aug[index].astype('float32')),
            torch.from_numpy(self.captions[index].astype('float32')),
            torch.from_numpy(self.captions_aug[index].astype('float32')),
            self.labels[index]
        )

    def __len__(self):
        return len(self.images)

class DatasetDuplet1(AbstractDataset):
    """
    Class for dataset representation.

    Each image has 5 corresponding captions

    Duplet dataset sample - img-txt (image and corresponding caption)
    """

    def __init__(self, images, captions, labels, idxs, captions_aug=None, images_aug=None, seed=42):
        """
        Initialization.

        :param images: image embeddings vector
        :param captions: captions embeddings vector
        :param labels: labels vector
        """
        super().__init__(images, captions, labels, idxs, captions_aug, images_aug, seed)

        caption_idxs = select_idxs(len(self.captions), 1, 5, seed=self.seed)[0]
        self.captions = self.captions[caption_idxs]

        # self.image_names_list = image_names_list
        # self.images_folder_path = images_folder_path
        # self.toTensor = torchvision.transforms.ToTensor()

    def __getitem__(self, index):
        """
        Returns a tuple (img, txt, label) - image and corresponding caption

        :param index: index of sample
        :return: tuple (img, txt, label)
        """
        # image_name = self.image_names_list[index]
        # raw_image = Image.open(os.path.join(self.images_folder_path, image_name))
        return (
            index,
            (self.idxs[index], self.idxs_cap[index]),
            # self.toTensor(raw_image),
            torch.from_numpy(self.images[index].astype('float32')),
            torch.from_numpy(self.captions[index].astype('float32')),
            self.labels[index]
        )

    def __len__(self):
        return len(self.images)
