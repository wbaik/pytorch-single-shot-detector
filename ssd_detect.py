from chainercv.datasets import voc_bbox_label_names
from chainercv.visualizations import vis_bbox

from src import VOCBboxDataset, SSDBoxCoder

import numpy as np

import torch
import torch.nn.functional as F
from torchvision import transforms

import matplotlib.pyplot as plt


net = torch.load('./ssd.pth')
net.eval()

def get_instance(img, transform, box_coder,
                  sizes=(224,224), CLS_SCORE=0.25):
    '''
    :param img: PIL.Image object of an image
    :param transform: torchvision.transforms.Compose
    :param box_coder: SSDBoxCoder
    :return: image, bboxs, labels, scores
    '''
    # resize
    img_resized = img.resize(sizes)
    img_transposed = np.array(img_resized).transpose(2, 0, 1)

    # transform
    x = transform(img_resized)

    # predict
    loc_preds, cls_preds = net(x.unsqueeze(0))

    # decode
    boxes, labels, scores = box_coder.decode(
        loc_preds.data.squeeze(),
        F.softmax(cls_preds.squeeze(), dim=1).data, CLS_SCORE)

    return img_transposed, boxes, labels, scores


if __name__ == '__main__':
    box_coder = SSDBoxCoder(net)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))
    ])

    dataset = VOCBboxDataset()

    random_numbers = np.random.randint(0,200,8)

    pil_imgs = map(lambda x: dataset[x][0], random_numbers)

    results = [[*get_instance(pil_img, transform, box_coder)]
               for pil_img in pil_imgs]

    fig, axes = plt.subplots(2,4)
    for i, ax in enumerate(axes.flat):
        vis_bbox(*results[i], label_names=voc_bbox_label_names, ax=ax)

    plt.tight_layout()
    plt.show()
