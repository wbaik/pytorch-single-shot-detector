import torch
import time

from src.dataset import get_dl


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
