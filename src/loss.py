import torch.nn as nn
import torch.nn.functional as F


# Reference:
#   https://github.com/chainer/chainercv/blob/master/chainercv/links/model/ssd/multibox_loss.py
#   https://github.com/kuangliu/torchcv/blob/master/torchcv/loss/ssd_loss.py
class SSDLoss(nn.Module):
    def __init__(self, n_classes):
        super(SSDLoss, self).__init__()
        self.n_classes = n_classes

    def _hard_negative_mining(self, cls_loss, pos, s=3):
        '''
        Return negative indices that is s times the number as positive indices.
        Originally from the SSD Paper(arXiv:1512.02325v5), s = 3.

        Args:
          cls_loss: (tensor) cross entropy loss between cls_preds and cls_targets, sized [N, #anchors].
          pos: (tensor) positive class mask, sized [N, #anchors].

        Return:
          (tensor) negative indices, sized [N, #anchors].
        '''
        cls_loss = cls_loss * (pos.float() - 1)

        _, idx = cls_loss.sort(1)   # sort by negative losses
        _, rank = idx.sort(1)       # [N, #anchors]

        num_neg = s * pos.sum(1)    # [N,]
        return rank < num_neg[:, None]

    def forward(self, loc_preds, loc_targets, cls_preds, cls_targets):
        '''Compute loss between (loc_preds, loc_targets) and (cls_preds, cls_targets).

        Args:
          loc_preds: (tensor) predicted locations, sized [N, #anchors, 4].
          loc_targets: (tensor) encoded target locations, sized [N, #anchors, 4].
          cls_preds: (tensor) predicted class confidences, sized [N, #anchors, #classes].
          cls_targets: (tensor) encoded target labels, sized [N, #anchors].

        loss:
          (tensor) loss = SmoothL1Loss(loc_preds, loc_targets) + CrossEntropyLoss(cls_preds, cls_targets).
        '''
        pos = cls_targets > 0                                           # [N, #anchors]
        batch_size = pos.size(0)
        num_pos = pos.sum().item()

        mask = pos.unsqueeze(2).expand_as(loc_preds)                    # [N, #anchors, 4]
        loc_loss = F.smooth_l1_loss(loc_preds[mask], loc_targets[mask],
                                    size_average=False)

        cls_loss = F.cross_entropy(cls_preds.view(-1, self.n_classes),
                                   cls_targets.view(-1), reduce=False)  # [N * #anchors,]
        cls_loss = cls_loss.view(batch_size, -1)
        cls_loss[cls_targets < 0] = 0                                   # set ignored loss to 0

        neg = self._hard_negative_mining(cls_loss, pos, 3)              # [N, #anchors]
        cls_loss = cls_loss[pos | neg].sum()

        print('| loc_loss: {:.3f} | cls_loss: {:.3f} |'.format(
            loc_loss.item()/num_pos, cls_loss.item()/num_pos), end='|')

        loss = (loc_loss+cls_loss)/num_pos
        return loss

