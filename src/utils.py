import torch


def _if_numpy_to_tensor(given):
    if not torch.is_tensor(given):
        given = torch.Tensor(given)
    return given


def _if_tensor_to_numpy(given):
    assert type(given) in [torch.tensor, torch.Tensor, np.array, np.ndarray]
    return given.numpy() if torch.is_tensor(given) else given.copy()


# Reference:
#     https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/
#     https://github.com/kuangliu/torchcv/blob/master/torchcv/utils/
def box_clamp(boxes, xmin, ymin, xmax, ymax):
    '''
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
