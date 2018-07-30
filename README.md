## SSD: Single Shot MultiBox Detector

Implementing [SSD](https://arxiv.org/abs/1512.02325) with few tweaks, in `PyTorch`.

The original paper uses 300x300 pixel inputs for VGG16 architecture. Here, we re-implement the model 
with 224x224 pixel input. Training model for 300px requires pretrained VGG16 (with the ImageNet data). 
Because `vgg16(pretrained=True)` from `torchvision` is 224px, this model uses 224px input. 
Naturally, number of bounding boxes used is decreased.

With GTX 1060, training time of 10 hours produces not so satisfying results; there is still a room for 
improvement, as the scores are improving. Since my original intention was to reproduce an SSD and understand 
the details, `git stash`ing for the time being. Training data is from 
[chainercv](https://chainercv.readthedocs.io/en/stable/), which utilizes 
[Pascal VOC data](http://host.robots.ox.ac.uk/pascal/VOC/).

### Training
Simply run the following to build a model and run the training:
```
python ssd_train.py
```
This works, because `chainercv` provides api that automatically downloads `VOC data`.

### Detect
Simply run the following:
```
python ssd_detect.py
```

Few results after some training:
![sample_pic](/images/Figure_1-10.png)
![sample_pic](/images/Figure_1-11.png)

### Credit
There has been a number of implementations for SSD. These are some of the
implementations I have used to learn the details of the original SSD.
`chainercv` relies on its own framework, where as `torchcv` relies on `pytorch`.

[chainercv](http://chainercv.readthedocs.io/en/stable/)

[torchcv](https://github.com/kuangliu/torchcv)
