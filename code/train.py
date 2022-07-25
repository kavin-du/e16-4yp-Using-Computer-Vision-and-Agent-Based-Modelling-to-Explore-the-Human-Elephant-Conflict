import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from imutils import paths
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from config.dataset import SegmentationDataset
#from config.model import UNet
from config.model import UNetResNet
from config import config  

"""
Note :- https://discuss.pytorch.org/t/semantic-segmentation-loss-function-data-format-help/111486
        https://www.jeremyjordan.me/semantic-segmentation/

The input to my network is a bunch of images in the form:
[B, C, H, W]

This is currently [12, 3, 512, 512]

The output from my network is in the form:
[B, NUMCLASSES, H, W]

This is currently [12, 20, 512, 512]

I am loading in my normal images and segmentation truths using PIL.
So my input images have 3 channels, and my segmentation images have 1 channel (in PILs P mode).

So my input image goes into the network and outputs a shape of [12, 20, 512, 512]
My ground truth images are in the shape of [12, 1, 512, 512]

My question:
What loss function should I use, and what format / shapes should my data be in?
Is is possible to just shove my outputs into a loss function as they are and calculate a loss, or do I need to reshape them in any way?
Do I have to calculate an output prediction with output.argmax(1) for input into a loss function?

The shapes look almost right. For a multi-class segmentation use case you could use 
nn.CrossEntropyLoss as the criterion, which expects the model output to contain logits in the shape 
[batch_size, nb_classes, height, width]. The target should have the shape [batch_size, height, width] 
(remove dim1 in your script via target = target.squeeze(1)) and should contain the class indices in the 
range [0, nb_classes-1].

Assuming you are dealing with 20 classes, here is a small code example:

output = torch.randn(2, 20, 24, 24, requires_grad=True)
target = torch.randint(0, 20, (2, 24, 24))
criterion = nn.CrossEntropyLoss()
loss = criterion(output, target)

Do I have to calculate an output prediction with output.argmax(1) for input into a loss function?

No, nn.CrossEntropyLoss expects the logits for each class. You could use torch.argmax(output, dim=1) 
to compute the predictions, where each pixel would contain the the predicted class index.
"""
"""
#unet = UNet(n_channels = 3, n_classes=8)
unet = UNetResNet(num_classes=8)
unet.train()
cross_entropy_loss = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(unet.parameters(), lr=0.0001, weight_decay=0.0001)   


imagePaths = sorted(list(paths.list_images(config.IMAGE_DATASET_PATH)))
maskPaths = sorted(list(paths.list_images(config.MASK_DATASET_PATH)))
split = train_test_split(imagePaths, maskPaths, test_size=config.TEST_SPLIT, random_state=42)
(trainImages, testImages) = split[:2]
(trainMasks, testMasks) = split[2:]

print("[INFO] saving testing image paths...")
f = open(config.TEST_PATHS, "w")
f.write("\n".join(testImages))
f.close()

transforms1 = transforms.Compose([transforms.ToPILImage(), transforms.Resize((config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH)), transforms.ToTensor()])
transforms2 = transforms.Compose([transforms.ToPILImage(), transforms.Resize((config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH))])
transforms = [transforms1, transforms2]
trainDS = SegmentationDataset(imagePaths=trainImages, maskPaths=trainMasks, transforms=transforms)

print(f"[INFO] found {len(trainDS)} examples in the training set...")

train_dataloader = DataLoader(trainDS, shuffle=True, batch_size=32)
loss_values = []

for epoch_idx in range(3):
    
    loss_batches = []
    for batch_idx, data in enumerate(train_dataloader):
    
        imgs, masks = data
        imgs = torch.autograd.Variable(imgs)
        masks = torch.autograd.Variable(masks)

        masks = torch.squeeze(masks.to(dtype=torch.long), dim=1) # {batch, height, width}, long

        y = unet(imgs)
        preds = y.to(dtype=torch.float32) # {batch, channels, height, width}, float

        loss = cross_entropy_loss(preds, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_batches.append(loss.data.cpu().numpy())

    print ('epoch: ' + str(epoch_idx) + ' training loss: ' + str(np.sum(loss_batches)))
    loss_values.append(np.sum(loss_batches))

model_file = './models/unet-' + str(epoch_idx)
torch.save(unet.state_dict(), model_file)
print ('model saved')

"""
unet_resnet = UNetResNet(num_classes=8)
model_path= './models/unet-14'
pretrained_model = torch.load(model_path)
for name, tensor in pretrained_model.items():
    unet_resnet.state_dict()[name].copy_(tensor)

unet_resnet.eval()


img = cv2.imread('./3530_img.png')

transforms = transforms.Compose([transforms.ToPILImage(), transforms.Resize((config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDTH)), transforms.ToTensor()])
img = transforms(img)
img = torch.unsqueeze(img, dim=0)

with torch.no_grad():
    pred = unet_resnet(img)
    pred = torch.softmax(pred, dim=1)
    pred = torch.squeeze(pred, dim=0) # reduce one dimension

class_idx = torch.argmax(pred, dim=0)   

mapping = {0: (16  , 16, 245),     #building = red
           1: (18, 213, 230  ),     #road = yellow
           2: (230, 18  , 29),     #water = blue
           3: (220  , 18, 230  ),     #barren = purple
           4: (29  , 230  , 18),     #forest = green
           5: (5, 58, 202),     #agriculture = brown
           6: (255  , 255  , 255),     #background = white    
           7: (0, 0, 0)}            # none = black


image = torch.zeros(256, 256, 3, dtype=torch.uint8)

for k in mapping:
    idx = (class_idx == torch.tensor(k, dtype=torch.uint8))
    validx = (idx == 1)
    image[validx,:] = torch.tensor(mapping[k], dtype=torch.uint8)

cv2.imwrite('./output/mask.png', image.numpy())