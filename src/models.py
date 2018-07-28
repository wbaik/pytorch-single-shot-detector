import torch
import torch.nn as nn
import torchvision.models as models


# Not used for training...
#   Scratch model with 300x300 pixel assumptions.
#   Training from scratch is bad idea. If wanted, we should use
#   ImageNet images to pretrain the model, then use it as SSD Head.
class VGG16SCRATCH(nn.Module):
    def __init__(self):
        super(VGG16SCRATCH, self).__init__()
        self.s1 = self._first_sequentials()
        self.layer_norm = nn.LayerNorm((512, 38, 38))
        self.amplifier = 20.0

        self.features_1 = nn.Sequential(
            # input size 38x38, output size 19x19
            nn.MaxPool2d(2, 2, ceil_mode=True),
            nn.Conv2d(512, 512, kernel_size=3,
                      stride=1, padding=1, dilation=1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1, 1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1, 1), nn.ReLU(),
            nn.MaxPool2d(3, 1, 1),
            nn.Conv2d(512, 1024, 3, 1, 6, 6), nn.ReLU(),
            nn.Conv2d(1024, 1024, 1), nn.ReLU(),
        )

        self.features_2 = nn.Sequential(
            # input size 19x19, output size 10x10
            nn.Conv2d(1024, 256, 1), nn.ReLU(),
            nn.Conv2d(256, 512, 3, 2, 1), nn.ReLU(),
        )

        self.features_3 = nn.Sequential(
            # input size 10x10, output size 5x5
            nn.Conv2d(512, 128, 1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1), nn.ReLU(),
        )

        self.features_4 = nn.Sequential(
            # input size 5x5, output size 3x3
            nn.Conv2d(256, 128, 1), nn.ReLU(),
            nn.Conv2d(128, 256, 3), nn.ReLU(),
        )

        self.features_5 = nn.Sequential(
            # input size 3x3, output size 1x1
            nn.Conv2d(256, 128, 1), nn.ReLU(),
            nn.Conv2d(128, 256, 3), nn.ReLU(),
        )

    def forward(self, x):
        outs = []
        x = self.s1(x) # 38x38
        outs += [x]
        x = self.layer_norm(x) * self.amplifier
        x = self.features_1(x) # 19x19
        outs += [x]
        x = self.features_2(x) # 10x10
        outs += [x]
        x = self.features_3(x) # 5x5
        outs += [x]
        x = self.features_4(x) # 3x3
        outs += [x]
        x = self.features_5(x) # 1x1
        outs += [x]

        return outs

    # 300x300 image outputs 512x38x38 features
    def _first_sequentials(self):
        outputs = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        layers = []
        in_channels = 3
        for output in outputs:
            if output == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
            else:
                layers += [nn.Conv2d(in_channels, output, kernel_size=3, padding=1),
                           nn.ReLU(True)]
                in_channels = output
        return nn.Sequential(*layers)


class SSD300SCRATCH(nn.Module):

    # Each feature corresponds to n pixels s.t.
    #       n_features * feature_to_n_pixels ~= 300
    feature_to_n_pixels = (8, 16, 32, 64, 100, 300)
    n_features = (38, 19, 10, 5, 3, 1)
    box_sizes = (30, 60, 111, 162, 213, 264, 315)
    aspect_ratios = ((2,), (2, 3), (2, 3), (2, 3), (2,), (2,))

    def __init__(self, num_classes):
        super(SSD300SCRATCH, self).__init__()
        self.n_classes = num_classes
        #    n_anchors increases respect to `aspect_ratios`
        self.n_anchors = (4, 6, 6, 6, 4, 4)
        self.n_channels = (512, 1024, 512, 256, 256, 256)

        self.features = VGG16SCRATCH()

        self.loc_layers = nn.ModuleList()
        self.cls_layers = nn.ModuleList()

        self._create_loc_cls_layers()

    def forward(self, input):
        loc_preds = []
        cls_preds = []

        features = self.features(input)
        for i, feature in enumerate(features):
            loc_pred = self.loc_layers[i](feature)
            loc_pred = loc_pred.permute(0, 2, 3, 1).contiguous()
            loc_preds += [loc_pred.view(loc_pred.shape[0], -1, 4)]

            cls_pred = self.cls_layers[i](feature)
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous()
            cls_preds += [cls_pred.view(cls_pred.shape[0], -1, self.n_classes)]

        loc_preds = torch.cat(loc_preds, 1)
        cls_preds = torch.cat(cls_preds, 1)

        return loc_preds, cls_preds

    def _create_loc_cls_layers(self):
        for n_channel, n_anchor in zip(self.n_channels, self.n_anchors):
            self.loc_layers += [nn.Conv2d(n_channel,
                                          n_anchor * 4,
                                          kernel_size=3,
                                          padding=1)]
            self.cls_layers += [nn.Conv2d(n_channel,
                                          n_anchor * self.n_classes,
                                          kernel_size=3,
                                          padding=1)]


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.s1 = models.vgg16(pretrained=True).features[:29]
        self.layer_norm = nn.LayerNorm((512, 14, 14))
        self.amplifier = 20.0

        self.features_1 = nn.Sequential(
            # input size 14x14, output size 7x7
            nn.MaxPool2d(2, 2, ceil_mode=True),
            nn.Conv2d(512, 512, kernel_size=3,
                      stride=1, padding=1, dilation=1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1, 1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1, 1), nn.ReLU(),
            nn.MaxPool2d(3, 1, 1),
            nn.Conv2d(512, 1024, 3, 1, 6, 6), nn.ReLU(),
            nn.Conv2d(1024, 1024, 1), nn.ReLU(),
        )

        self.features_2 = nn.Sequential(
            # input size 7x7, output size 5x5
            nn.Conv2d(1024, 256, 1, padding=1), nn.ReLU(),
            nn.Conv2d(256, 512, 3, 2, 1), nn.ReLU(),
        )

        self.features_3 = nn.Sequential(
            # input size 5x5, output size 3x3
            nn.Conv2d(512, 128, 1), nn.ReLU(),
            nn.Conv2d(128, 256, 3), nn.ReLU(),
        )

        self.features_4 = nn.Sequential(
            # input size 3x3, output size 1x1
            nn.Conv2d(256, 128, 1), nn.ReLU(),
            nn.Conv2d(128, 256, 3), nn.ReLU(),
        )


    def forward(self, x):
        outs = []
        x = self.s1(x) # 14x14
        outs += [x]
        x = self.layer_norm(x) * self.amplifier
        x = self.features_1(x) # 7x7
        outs += [x]
        x = self.features_2(x) # 5x5
        outs += [x]
        x = self.features_3(x) # 3x3
        outs += [x]
        x = self.features_4(x) # 1x1
        outs += [x]

        return outs


class SSD224(nn.Module):
    # Each feature corresponds to n pixels s.t.
    #       n_features * feature_to_n_pixels ~= 224
    feature_to_n_pixels = (16, 32, 45, 75, 224)
    n_features = (14, 7, 5, 3, 1)
    box_sizes = (44, 82, 120, 160, 200, 235)
    aspect_ratios = ((2,), (2, 3), (2, 3), (2, 3), (2,))

    def __init__(self, num_classes):
        super(SSD224, self).__init__()
        self.n_classes = num_classes
        #    n_anchors increases respect to `aspect_ratios`
        self.n_anchors = (4, 6, 6, 6, 4, 4)
        self.n_channels = (512, 1024, 512, 256, 256, 256)

        self.features = VGG16()

        self.loc_layers = nn.ModuleList()
        self.cls_layers = nn.ModuleList()

        self._create_loc_cls_layers()

    def forward(self, input):
        loc_preds = []
        cls_preds = []

        features = self.features(input)
        for i, feature in enumerate(features):
            loc_pred = self.loc_layers[i](feature)
            loc_pred = loc_pred.permute(0, 2, 3, 1).contiguous()
            loc_preds += [loc_pred.view(loc_pred.shape[0], -1, 4)]

            cls_pred = self.cls_layers[i](feature)
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous()
            cls_preds += [cls_pred.view(cls_pred.shape[0], -1, self.n_classes)]

        loc_preds = torch.cat(loc_preds, 1)
        cls_preds = torch.cat(cls_preds, 1)

        return loc_preds, cls_preds

    def _create_loc_cls_layers(self):
        for n_channel, n_anchor in zip(self.n_channels, self.n_anchors):
            self.loc_layers += [nn.Conv2d(n_channel,
                                          n_anchor * 4,
                                          kernel_size=3,
                                          padding=1)]
            self.cls_layers += [nn.Conv2d(n_channel,
                                          n_anchor * self.n_classes,
                                          kernel_size=3,
                                          padding=1)]
