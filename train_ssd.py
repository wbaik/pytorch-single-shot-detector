import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from chainercv.visualizations import vis_bbox
from chainercv.datasets import voc_bbox_label_names

from transforms import random_distort, random_crop, random_flip, random_paste, resize

from src.box_coder import SSDBoxCoder
from src.voc_dataset import VOCBboxDataset
import src.models_pretrained as models
import src.loss as loss

import numpy as np
import time, random
import functools


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
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))
    ])(img)
    return img, boxes, labels

def _if_tensor_to_numpy(given):
    assert type(given) in [torch.tensor, torch.Tensor, np.array, np.ndarray]
    return given.float().numpy() if torch.is_tensor(given) else given.copy().astype(np.float32)


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


# models can be multiples, so are, criterions, optimizers, schedulers...
def train_model(models, criterions, optimizers, schedulers, num_epochs=5):
    since = time.time()

    if not hasattr(models, '__iter__'):
        models, criterions, optimizers, schedulers = [models], [criterions], [optimizers], [schedulers]

    for epoch in range(num_epochs):
        print('-' * 10)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'validate']:
            if phase == 'train':
                for scheduler, model in zip(schedulers, models):
                    scheduler.step()
                    model.train()
            else:
                for model in models:
                    model.eval()

            train_dl, val_dl = get_dl(model)
            dl = train_dl if phase == 'train' else val_dl

            running_losses = [torch.tensor([0.0], dtype=torch.float).cuda()
                              for _ in range(len(models))]

            for cur_idx, (image, loc_label, cls_label) in enumerate(dl):

                image, loc_label, cls_label = map(lambda x: x.cuda(), 
                                                  (image, loc_label, cls_label))

                for optimizer in optimizers:
                    optimizer.zero_grad()

                outs = [model(image) for model in models]

                losses = [criterion(out[0], loc_label, out[1], cls_label.long()) 
                          for criterion, out in zip(criterions, outs)]

                if phase == 'train':
                    for loss, optimizer in zip(losses, optimizers):
                        loss.backward()
                        optimizer.step()
                
                for loss, running_loss, out in zip(losses, running_losses, outs):
                    running_loss += loss.item() * image.size(0)
                    print(' Average loss: {:.4f}'.format(running_loss.item() /
                                                        (image.size(0)*(cur_idx+1))))

            # epoch_losses = [running_loss / len(dl) for running_loss in running_losses]
            epoch_losses = []
            for running_loss in running_losses:
                epoch_losses.append(running_loss / len(dl))

            for n_iter, epoch_loss in enumerate(epoch_losses):
                print('{}th Model: {} loss: {:.4f}'.format(n_iter, phase, 
                                                           epoch_loss.cpu().item()))
                      
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))


def train():
    try:
        ssd = torch.load('./ssd.pth').cuda()
    except:
        ssd = models.SSD300(21).float().cuda()

    loss_fn = loss.SSDLoss(21)

    optimizer_fn = optim.SGD(ssd.parameters(), lr=0.0005, momentum=0.9, weight_decay=1e-6)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer_fn, step_size=7, gamma=0.1)

    train_model(ssd, loss_fn, optimizer_fn, exp_lr_scheduler, 14)

    ssd = ssd.cpu()
    torch.save(ssd, './ssd.pth')


if __name__ == '__main__':
    train()
