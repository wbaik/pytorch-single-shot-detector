import torch
import numpy as np
import itertools
import math


def _if_numpy_to_tensor(given):
    if not torch.is_tensor(given):
        given = torch.Tensor(given)
    return given


def _if_tensor_to_numpy(given):
    assert type(given) in [torch.tensor, torch.Tensor, np.array, np.ndarray]
    return given.numpy() if torch.is_tensor(given) else given.copy()


def box_clamp(boxes, xmin, ymin, xmax, ymax):
    '''Clamp boxes.

    Args:
      boxes: (tensor) bounding boxes of (xmin,ymin,xmax,ymax), sized [N,4].
      xmin: (number) min value of x.
      ymin: (number) min value of y.
      xmax: (number) max value of x.
      ymax: (number) max value of y.

    Returns:
      (tensor) clamped boxes.
    '''
    boxes[:,0].clamp_(min=xmin, max=xmax)
    boxes[:,1].clamp_(min=ymin, max=ymax)
    boxes[:,2].clamp_(min=xmin, max=xmax)
    boxes[:,3].clamp_(min=ymin, max=ymax)
    return boxes


def change_box_order(boxes, order):
    '''
    Args:
      boxes: (tensor) bounding boxes, sized [N,4].
      order: (str) either 'xyxy2xywh' or 'xywh2xyxy'.

    Returns:
      (tensor) converted bounding boxes, sized [N,4].

    Reference:
      https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py
    '''
    assert order in ['xyxy2xywh', 'xywh2xyxy']

    boxes = _if_numpy_to_tensor(boxes)

    a = boxes[:, :2]
    b = boxes[:, 2:]

    return torch.cat([(a + b) / 2, b - a], 1) if order == 'xyxy2xywh' \
        else torch.cat([a - b / 2, a + b / 2], 1)


def box_iou(box1, box2):
    '''
    The box order must be (xmin, ymin, xmax, ymax).

    Args:
      box1: (tensor) bounding boxes, sized [N,4].
      box2: (tensor) bounding boxes, sized [M,4].

    Return:
      (tensor) iou, sized [N,M].

    Reference:
      https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
      https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/box.py

    '''

    assert torch.is_tensor(box1), f'Box1 expected to be tensor, found {type(box1)} instead...'
    assert torch.is_tensor(box2), f'Box2 expected to be tensor, found {type(box2)} instead...'

    top_left = torch.max(box1[:, None, :2], box2[:, :2])        # [N,M,2]
    bottom_right = torch.min(box1[:, None, 2:], box2[:, 2:])    # [N,M,2]

    width_height = (bottom_right - top_left).clamp(min=0)       # [N,M,2]

    intersect = width_height[:, :, 0] * width_height[:, :, 1]   # [N,M]

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]

    iou = intersect / (area1[:, None] + area2 - intersect)

    assert iou.shape == (box1.shape[0], box2.shape[0])

    return iou


def box_nms(bboxes, scores, threshold=0.5):
    '''Non maximum suppression.

    Args:
      bboxes: (tensor) bounding boxes, sized [N,4].
      scores: (tensor) confidence scores, sized [N,].
      threshold: (float) overlap threshold.

    Returns:
      keep: (tensor) selected indices.

    Reference:
      https://github.com/rbgirshick/py-faster-rcnn/blob/master/lib/nms/py_cpu_nms.py
    '''
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 2]
    y2 = bboxes[:, 3]

    areas = (x2-x1) * (y2-y1)
    _, order = scores.sort(0, descending=True)

    keep = []
    while order.numel() > 0:
        i = order[0]
        keep.append(i)

        if order.numel() == 1:
            break

        xx1 = x1[order[1:]].clamp(min=x1[i].item())
        yy1 = y1[order[1:]].clamp(min=y1[i].item())
        xx2 = x2[order[1:]].clamp(max=x2[i].item())
        yy2 = y2[order[1:]].clamp(max=y2[i].item())

        w = (xx2-xx1).clamp(min=0)
        h = (yy2-yy1).clamp(min=0)
        inter = w * h

        overlap = inter / (areas[i] + areas[order[1:]] - inter)
        ids = (overlap<=threshold).nonzero().squeeze()
        if ids.numel() == 0:
            break
        order = order[ids+1]
    return torch.LongTensor(keep)


class SSDBoxCoder:
    def __init__(self, ssd_model):
        self.feature_to_n_pixels = ssd_model.feature_to_n_pixels
        self.n_features = ssd_model.n_features
        self.box_sizes = ssd_model.box_sizes
        self.aspect_ratios = ssd_model.aspect_ratios
        self.default_boxes = self._get_default_boxes()

    def _get_default_boxes(self):
        boxes = []
        for i, fm_size in enumerate(self.n_features):
            for h, w in itertools.product(range(fm_size), repeat=2):
                cx = (w + 0.5) * self.feature_to_n_pixels[i]
                cy = (h + 0.5) * self.feature_to_n_pixels[i]

                s = self.box_sizes[i]
                boxes.append((cx, cy, s, s))

                s = math.sqrt(self.box_sizes[i] * self.box_sizes[i+1])
                boxes.append((cx, cy, s, s))

                s = self.box_sizes[i]
                for ar in self.aspect_ratios[i]:
                    boxes.append((cx, cy, s * math.sqrt(ar), s / math.sqrt(ar)))
                    boxes.append((cx, cy, s / math.sqrt(ar), s * math.sqrt(ar)))

        return torch.Tensor(boxes)  # xywh

    def encode(self, boxes, labels, threshold=0.5):
        '''Encode target bounding boxes and class labels.

        SSD coding rules:
          tx = (x - anchor_x) / (variance[0] * anchor_w)
          ty = (y - anchor_y) / (variance[0] * anchor_h)
          tw = log(w / anchor_w) / variance[1]
          th = log(h / anchor_h) / variance[1]

        Args:
          boxes: (tensor) bounding boxes of (xmin,ymin,xmax,ymax), sized [#obj, 4].
          labels: (tensor) object class labels, sized [#obj,].

        Returns:
          loc_targets: (tensor) encoded bounding boxes, sized [#anchors,4].
          cls_targets: (tensor) encoded class labels, sized [#anchors,].

        Revised from the original version:
          Originally was implemented in kuangliu/torchcv:  kuangliu's assignments for boxes were off...

        Reference:
          https://github.com/chainer/chainercv/blob/master/chainercv/links/model/ssd/multibox_coder.py
          https://github.com/kuangliu/torchcv/blob/master/torchcv/models/ssd/box_coder.py
        '''

        def argmax(x):
            v, i = x.max(0)
            j = v.max(0)[1][0]
            return (i[j], j)

        default_boxes = self.default_boxes  # xywh
        default_boxes = change_box_order(default_boxes, 'xywh2xyxy')

        test_box = _if_numpy_to_tensor(boxes)

        if test_box.size() == torch.tensor([]).size():
            return torch.zeros(default_boxes.shape, dtype=torch.float32), \
                    torch.zeros(default_boxes.shape[0], dtype=torch.int32)


        ious = box_iou(default_boxes, boxes)  # [#anchors, #obj]
        index = torch.LongTensor(len(default_boxes)).fill_(-1)
        masked_ious = ious.clone()
        while True:
            i, j = argmax(masked_ious)
            if masked_ious[i,j] < 1e-6:
                break
            index[i] = j
            masked_ious[i,:] = 0
            masked_ious[:,j] = 0

        mask = (index<0) & (ious.max(1)[0]>=threshold)
        if mask.any():
            try:
                index[mask] = ious[mask.nonzero().squeeze()].max(1)[1]
            except:
                print('Exception Raised: {}'.format(mask.nonzero().squeeze().shape))

        boxes = boxes[index.clamp(min=0)]  # negative index not supported
        boxes = change_box_order(boxes, 'xyxy2xywh')
        default_boxes = change_box_order(default_boxes, 'xyxy2xywh')

        variances = (0.1, 0.2)
        loc_xy = (boxes[:,:2]-default_boxes[:,:2]) / default_boxes[:,2:] / variances[0]
        loc_wh = torch.log(boxes[:,2:]/default_boxes[:,2:]) / variances[1]
        loc_targets = torch.cat([loc_xy,loc_wh], 1)
        cls_targets = 1 + labels[index.clamp(min=0)]
        cls_targets[index<0] = 0
        return loc_targets, cls_targets

    def decode(self, loc_preds, cls_preds, score_thresh=0.63, nms_thresh=0.4):
        '''Decode predicted loc/cls back to real box locations and class labels.

        Args:
          loc_preds: (tensor) predicted loc, sized [8732,4].
          cls_preds: (tensor) predicted conf, sized [8732,21].
          score_thresh: (float) threshold for object confidence score.
          nms_thresh: (float) threshold for box nms.

        Returns:
          boxes: (tensor) bbox locations, sized [#obj,4].
          labels: (tensor) class labels, sized [#obj,].
        '''
        variances = (0.1, 0.2)
        xy = loc_preds[:,:2] * variances[0] * self.default_boxes[:,2:] + self.default_boxes[:,:2]
        wh = torch.exp(loc_preds[:,2:]*variances[1]) * self.default_boxes[:,2:]
        box_preds = torch.cat([xy-wh/2, xy+wh/2], 1)

        boxes = []
        labels = []
        scores = []
        num_classes = cls_preds.size(1)
        for i in range(num_classes-1):
            score = cls_preds[:, i+1]  # class i corresponds to (i+1) column

            mask = score > score_thresh
            if not mask.any():
                continue
            box = box_preds[mask.nonzero().squeeze()]
            score = score[mask]

            try:
                keep = box_nms(box, score, nms_thresh)

                boxes.append(box[keep])
                labels.append(torch.LongTensor(len(box[keep])).fill_(i))
                scores.append(score[keep])
            except Exception as inst:
                print('Exception Raised: {}'.format(type(inst)))
                print('inst.args : {}'.format(inst.args))
                print('inst: {}'.format(inst))

        try:
            boxes = torch.cat(boxes, 0)
            labels = torch.cat(labels, 0)
            scores = torch.cat(scores, 0)
        except Exception as inst:
            print('Exception Raised: {}'.format(type(inst)))
            print('inst.args : {}'.format(inst.args))
            print('inst: {}'.format(inst))

        return boxes, labels, scores
