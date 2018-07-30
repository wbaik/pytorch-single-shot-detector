import torch
import torch.optim as optim

import numpy as np

from src import SSD224, SSDLoss, train_model


def train():
    try:
        ssd = torch.load('./ssd.pth').cuda()
    except Exception as inst:
        print('Exception Raised: {}'.format(type(inst)))
        print('inst.args : {}'.format(inst.args))
        print('inst: {}'.format(inst))
        ssd = SSD224(21).float().cuda()

    loss_fn = SSDLoss(21)

    optimizer_fn = optim.SGD(ssd.parameters(),
                             lr=0.0002, momentum=0.9, weight_decay=1e-7)

    def sin_lr(x):
        return np.abs(np.sin((x + 0.01) * 0.2))

    exp_lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer_fn,
                                                   lr_lambda=[sin_lr])

    train_model(ssd, loss_fn, optimizer_fn, exp_lr_scheduler, 30)

    ssd = ssd.cpu()
    torch.save(ssd, './ssd.pth')


if __name__ == '__main__':
    train()
