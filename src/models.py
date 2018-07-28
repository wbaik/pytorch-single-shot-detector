import torch
import torch.nn as nn


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
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


class SSD300(nn.Module):

    # Each feature corresponds to n pixels s.t.
    #       n_features * feature_to_n_pixels ~= 300
    feature_to_n_pixels = (8, 16, 32, 64, 100, 300)
    n_features = (38, 19, 10, 5, 3, 1)
    box_sizes = (30, 60, 111, 162, 213, 264, 315)
    aspect_ratios = ((2,), (2, 3), (2, 3), (2, 3), (2,), (2,))

    def __init__(self, num_classes):
        super(SSD300, self).__init__()
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


class GDD(nn.Module):
    def __init__(self, n_layer, n_filter, image):
        super(GDD, self).__init__()
        self.model = models.vgg19(pretrained=True).features
        self.n_layer = n_layer
        self.n_filter = n_filter
        self.image = self.image_to_tensor(image)
        self.hooked_out = 0
        self.create_hook()

    def image_to_tensor(self, img):
        img = cv2.resize(np.array(img).copy(), (224, 224))
        img = np.array(img).transpose(2, 0, 1).astype(np.float32)
        img /= 255
        img = np.expand_dims(img, 0)
        img = np.ascontiguousarray(img)
        image = torch.tensor(img, requires_grad=True).float()
        return image

    def tensor_to_image(self, img):
        img = img.squeeze()
        img = img.data.cpu().numpy().transpose(1, 2, 0)
        return img

    def create_hook(self):
        def hook_this(x, in_, out_):
            self.hooked_out = out_[0, self.n_filter]

        self.model[self.n_layer].register_forward_hook(hook_this)

    def deep_dream(self):
        opt = optim.SGD([self.image], lr=1, weight_decay=1e-4)
        self.model.eval()
        self.resulting_img = []
        for i in range(1, 50):
            opt.zero_grad()
            # This is terrible, but must be done here to avoid Nonleaf error ...
            x = self.image
            x = self.modules(x)

            loss = -torch.mean(self.hooked_out)
            loss.backward()
            opt.step()

            if i % 10 == 0:
                self.resulting_img += [self.tensor_to_image(self.image)]


