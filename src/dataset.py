import functools, random
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from chainercv.visualizations import vis_bbox
from chainercv.datasets import voc_bbox_label_names

from src.box_coder import SSDBoxCoder
from src.voc_dataset import VOCBboxDataset
from transforms import random_distort, random_crop, random_flip, random_paste, resize


class VocTorchDataset(Dataset):
    def __init__(self, ssd_box_coder, shuffled_index, vocbbox_dataset, transform=None):
        self.ssd_box_coder = ssd_box_coder
        self.vocbbox_dataset = vocbbox_dataset
        self.index = shuffled_index
        self.transform = transform

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index, show_img=False):
        index = self.index[index]

        image = self.get_image_from_dataset(index)
        bbox, label, difficult = self.vocbbox_dataset._get_annotations(index)

        bbox, label = torch.Tensor(bbox), torch.LongTensor(label)

        if self.transform and not show_img:
            image, bbox, label = self.transform(image, bbox, label)
            bbox, label = self.ssd_box_coder.encode(bbox, label)

        return image, bbox, label

    def get_image_from_dataset(self, index):
        return self.vocbbox_dataset._get_image(index)

    def show_img(self, index):
        vis_bbox(*(self.__getitem__(index, True)[:-1]), label_names=voc_bbox_label_names)


def transform_image_w_bbox(img, boxes, labels, img_size=224):

    assert torch.is_tensor(boxes), 'type(boxes) : {}'.format(type(boxes))

    img = random_distort(img)
    if random.random() < 0.5:
        img, boxes = random_paste(img, boxes, max_ratio=4, fill=(123,116,103))
    img, boxes, labels = random_crop(img, boxes, labels)
    img, boxes = resize(img, boxes, size=(img_size, img_size), random_interpolation=True)
    img, boxes = random_flip(img, boxes)
    img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
    ])(img)
    return img, boxes, labels


PCT_TRAIN=0.9
def train_validate_split(length, split_pct=PCT_TRAIN):
    '''
    :param length: length of the total dataset
    :param split_pct: percentage to use for training
    :return np.array of indices for train, test
    '''
    total = np.arange(length)
    np.random.shuffle(total)  # inplace

    cutoff = int(length * split_pct)
    train, validate = total[:cutoff], total[cutoff:]

    return train, validate


def get_dl(ssd_model):

    encoder = SSDBoxCoder(ssd_model)
    chainercv_voc_bbox = VOCBboxDataset()

    voc_torch_dataset = functools.partial(VocTorchDataset,
                                          ssd_box_coder=encoder,
                                          transform=transform_image_w_bbox,
                                          vocbbox_dataset=chainercv_voc_bbox)

    train, validate = train_validate_split(5717, PCT_TRAIN)

    train_dataloader = DataLoader(voc_torch_dataset(shuffled_index=train),
                                  num_workers=1, batch_size=20)
    val_dataloader = DataLoader(voc_torch_dataset(shuffled_index=validate),
                                num_workers=1, batch_size=20)

    return train_dataloader, val_dataloader