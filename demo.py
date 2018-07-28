import torch
import torch.nn.functional as F
from torchvision import transforms


from PIL import Image, ImageDraw
from src.box_coder import SSDBoxCoder


net = torch.load('./ssd.pth')
net.eval()

img = Image.open('./images/000084.jpg')
img = img.resize((224, 224))

transform = transforms.Compose([
    transforms.ToTensor()
])

x = transform(img)
loc_preds, cls_preds = net(x.unsqueeze(0))

print('Decoding..')
box_coder = SSDBoxCoder(net)
boxes, labels, scores = box_coder.decode(
    loc_preds.data.squeeze(), F.softmax(cls_preds.squeeze(), dim=1).data, 0.15)
print(labels)
print(scores)

draw = ImageDraw.Draw(img)
for box in boxes:
    draw.rectangle(list(box), outline='red')
img.show()
