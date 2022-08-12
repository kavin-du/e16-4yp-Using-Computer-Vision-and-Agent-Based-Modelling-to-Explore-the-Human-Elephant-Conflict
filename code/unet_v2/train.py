from asyncio.log import logger
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn.functional as F
import torch.optim as optim
from model import UNET, UNetWithResnet50Encoder
from loss_functions import mIoULoss, FocalLoss
import os
import numpy as np
from matplotlib import pyplot as plt
from loguru import logger

from utils import get_train_loaders, get_test_loaders, get_arguments, save_image

args = get_arguments()

# Hyperparameters etc
LEARNING_RATE = 1e-4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE =  args.batches #1
NUM_EPOCHS =  args.epochs #4 
NUM_WORKERS = 2
IMAGE_HEIGHT = args.resolution # 350 # 1024 originally
IMAGE_WIDTH = args.resolution # 350 # 1024 originally
PIN_MEMORY = True
TEST_MODE = args.testing 
MODEL_PATH = args.model
BASE_DIR = args.dataset
ARCHITECTURE = args.architecture

TRAIN_IMG_DIR = os.path.join(BASE_DIR, 'Train/Rural/images_png')
TRAIN_MASK_DIR = os.path.join(BASE_DIR, 'Train/Rural/masks_png')
VAL_IMG_DIR = os.path.join(BASE_DIR, 'Val/Rural/images_png')
VAL_MASK_DIR = os.path.join(BASE_DIR, 'Val/Rural/masks_png')
TEST_DATA_DIR = os.path.join(BASE_DIR, 'Test/Rural/images_png')

MODEL_FOLDER = 'saved_models' # saved models in each epoch
SAVED_IMAGE_FOLDER = 'visualizations'

def acc(y, pred_mask):
  seg_acc = (y.cpu() == torch.argmax(pred_mask, axis=1).cpu()).sum() / torch.numel(y.cpu())
  return seg_acc

def main():
  # out_channels = no of segmentation classes
  # model = UNET(in_channels=3, out_channels=8).to(DEVICE)
  
  if ARCHITECTURE == 'UNet':
    unet = UNET(in_channels=3, out_channels=8)
  elif ARCHITECTURE == 'UNet-Resnet50':  
    unet = UNetWithResnet50Encoder(n_classes=8)
  else:  
    logger.debug("no architecture is selected")
  model = torch.nn.DataParallel(unet).to(DEVICE)
  
  if TEST_MODE is False:
    # loss_fn = mIoULoss(n_classes=8).to(DEVICE)
    loss_fn = FocalLoss(gamma=3/4).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    min_loss = torch.tensor(float('inf'))

    os.makedirs(MODEL_FOLDER, exist_ok=True)

    loss_history = []
    scheduler_counter = 0

    train_transform = A.Compose([
      A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
      A.Rotate(limit=35, p=1.0),
      A.HorizontalFlip(p=0.5),
      A.VerticalFlip(p=0.1),
      A.Normalize(
        mean=[0.0, 0.0, 0.0],
        std=[1.0, 1.0, 1.0],
        max_pixel_value=255.0,
      ),
      ToTensorV2(),
    ])

    val_transforms = A.Compose([
      A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
      A.Normalize(
        mean=[0.0, 0.0, 0.0],
        std=[1.0, 1.0, 1.0],
        max_pixel_value=255.0,
      ),
      ToTensorV2(),
    ])

    train_loader, val_loader = get_train_loaders(
      TRAIN_IMG_DIR, 
      TRAIN_MASK_DIR, 
      VAL_IMG_DIR, 
      VAL_MASK_DIR,
      BATCH_SIZE,
      train_transform,
      val_transforms,
      NUM_WORKERS,
      PIN_MEMORY
    )

    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(NUM_EPOCHS):
      torch.cuda.empty_cache()

      model.train()
      loss_list = []
      acc_list = []

      loop = tqdm(train_loader)

      for batch_idx, (data, targets) in enumerate(loop):
        torch.cuda.empty_cache()
        data = data.to(DEVICE)
        targets = targets.to(DEVICE)

        # forward   
        # with torch.cuda.amp.autocast(): # automatically type cast to smaller memory footprint
        predictions = model(data)

        # probs = F.softmax(output, dim=1)
        # predictions = torch.argmax(probs, axis=1)


        loss = loss_fn(predictions, targets.type(torch.LongTensor).to(DEVICE))
        # loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

        loss_list.append(loss.cpu().detach().numpy())
        acc_list.append(acc(targets, predictions).numpy())

      scheduler_counter += 1

      # ===================
      # validation
      model.eval()
      val_loss_list = []
      val_acc_list = []
      for batch_idx, (x, y) in enumerate(val_loader):
        with torch.no_grad():
          pred_mask = model(x.to(DEVICE)) 
        val_loss = loss_fn(pred_mask, y.type(torch.LongTensor).to(DEVICE))
        val_loss_list.append(val_loss.cpu().detach().numpy())
        val_acc_list.append(acc(y, pred_mask).numpy())
      
      print('epoch {} - loss : {:.5f} - acc : {:.2f} - val loss : {:.5f} - val acc : {:.2f}'.format(epoch, 
                                                                                                  np.mean(loss_list), 
                                                                                                  np.mean(acc_list), 
                                                                                                  np.mean(val_loss_list),
                                                                                                  np.mean(val_acc_list)))
      loss_history.append([epoch, np.mean(loss_list), np.mean(val_loss_list)])

      compare_loss = np.mean(val_loss_list)
      is_best = compare_loss < min_loss
      if is_best:
        scheduler_counter = 0
        min_loss = min(compare_loss, min_loss)
        torch.save(model.module.state_dict(), '{}/unet_epoch_{}_{:.5f}.pt'.format(MODEL_FOLDER, epoch, np.mean(val_loss_list)))

      if scheduler_counter > 5:
        lr_scheduler.step()
        print(f'Lowering learning rate to {optimizer.param_groups[0]["lr"]}')
        scheduler_counter = 0
    
    # plotting losses
    loss_history = np.array(loss_history)
    plt.figure(figsize=(15, 10))
    plt.plot(loss_history[:,0], loss_history[:,1], color='b', linewidth=3)
    plt.plot(loss_history[:,0], loss_history[:,2], color='r', linewidth=3)
    plt.title('Focal Loss')
    plt.xlabel('epoch'), plt.ylabel('loss')
    plt.grid()
    plt.legend(['training', 'validation'])
    plt.savefig('loss_history.png')

  else:  # loading mode

    test_transforms = A.Compose([
      A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
      A.Normalize(
        mean=[0.0, 0.0, 0.0],
        std=[1.0, 1.0, 1.0],
        max_pixel_value=255.0,
      ),
      ToTensorV2(),
    ])

    test_loader = get_test_loaders(TEST_DATA_DIR, test_transforms, NUM_WORKERS, PIN_MEMORY)

    # model.load_state_dict(torch.load(MODEL_PATH))

    unet.load_state_dict(torch.load(MODEL_PATH))
    model = torch.nn.DataParallel(unet).to(DEVICE)

    os.makedirs(f'{SAVED_IMAGE_FOLDER}', exist_ok=True)

    model.eval()

    for batch_idx, (x, _) in enumerate(test_loader):
      x = x.to(DEVICE) # tensor: [1, n_classes, height, width]
      with torch.no_grad():
        out = model(x) # tensor: [1, n_classes, height, width]

        probs = F.softmax(out, dim=1)
        mask = torch.argmax(probs, axis=1).cpu().detach().squeeze()
        # mask = torch.argmax(probs, axis=1).cpu().detach().squeeze().numpy()

        im = np.moveaxis(x.squeeze().cpu().detach().squeeze().numpy(), 0, -1).copy()*255
        im = im.astype(int)

        save_image(SAVED_IMAGE_FOLDER, batch_idx, im, mask)

    torch.cuda.empty_cache()

if __name__ == '__main__':
    torch.cuda.empty_cache()
    main()
    
