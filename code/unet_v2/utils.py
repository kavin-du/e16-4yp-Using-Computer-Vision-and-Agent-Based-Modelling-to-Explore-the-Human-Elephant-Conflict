import torch
import torchvision
from dataset import SatelliteDataset
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F
from matplotlib import pyplot as plt
import os, argparse

def get_arguments():
  parser = argparse.ArgumentParser()
  parser.add_argument('-e', '--epochs', type=int, default=50, help='number of epochs')
  parser.add_argument('-b', '--batches', type=int, default=8, help='batch size')
  parser.add_argument('-r', '--resolution', type=int, default=1024, help='resolution of image SxS')
  parser.add_argument('-t', '--testing', action='store_true', help='switch between training and testing')
  parser.add_argument('-m', '--model', type=str,default=None, help='path for the saved model')
  parser.add_argument('-d', '--dataset', type=str,default='.', help='dataset folder path')
  return parser.parse_args()

def save_checkpoint(state, filename='my_checkpoint.pth.tar'):
  print('=> Saving checkpoint')
  torch.save(state, filename)

def load_checkpoint(checkpoint, model):
  print('=> Loading checkpoint')
  model.load_state_dict(checkpoint['state_dict'])

def get_train_loaders(
  train_dir,
  train_maskdir,
  val_dir,
  val_maskdir,
  batch_size,
  train_transform,
  val_transform,
  num_workers=4,
  pin_memory=True
):
  
  train_ds = SatelliteDataset(
    image_dir=train_dir,
    mask_dir=train_maskdir,
    transform=train_transform
  )

  train_loader = DataLoader(
    train_ds,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=pin_memory,
    shuffle=True
  )

  val_ds = SatelliteDataset(
    image_dir=val_dir,
    mask_dir=val_maskdir,
    transform=val_transform
  )

  val_loader = DataLoader(
    val_ds,
    batch_size=batch_size,
    num_workers=num_workers,
    pin_memory=pin_memory,
    shuffle=False  
  )

  return train_loader, val_loader

def get_test_loaders(test_data_dir, test_transform, num_workers=4, pin_memory=True):
  test_ds = SatelliteDataset(
    image_dir=test_data_dir,
    mask_dir=test_data_dir,
    transform=test_transform
  )

  test_loader = DataLoader(
    test_ds,
    num_workers=num_workers,
    pin_memory=pin_memory,
    shuffle=False  
  )

  return test_loader

def save_image(folder, index, original, mask):
  height, width = mask.shape[0], mask.shape[1]

  mapping = {
      0: (0, 0, 0), # ignore
      1: (255, 255, 255), # background
      2: (255, 0, 0), # building
      3: (255, 255, 0), # road
      4: (0, 0, 255), # water
      5: (159, 129, 183), # barren
      6: (0, 255, 0), # forest
      7: (255, 195, 128)} # agricultural


  colored_mask = torch.zeros(height, width, 3, dtype=torch.uint8)

  for k in mapping:   
    idx = (mask == torch.tensor(k, dtype=torch.uint8))
    validx = (idx == 1)
    colored_mask[validx,:] = torch.tensor(mapping[k], dtype=torch.uint8)
            
  # colored_mask = colored_mask.permute(2, 0, 1)

  colored_mask = colored_mask.squeeze().cpu().numpy()

  plt.figure(figsize=(25,25))
        
  plt.subplot(1,2,1)
  plt.imshow(original)

  plt.subplot(1,2,2)
  plt.imshow(colored_mask)
  
  plt.savefig(f'{folder}/{index}.png')
  print(f'{folder}/{index}.png')


def save_predictions_as_imgs(loader, model, folder='saved_images', device='cuda'):
  os.makedirs(folder, exist_ok=True)
  model.eval()
  for index, (x, y) in enumerate(loader):

    # x = x.unsqueeze(0)
    orig = torch.clone(x)
    x = x.to(device=device)
    with torch.no_grad():
      output = model(x)
      preds = torch.sigmoid(output)
      #preds = preds.squeeze()
      preds = torch.mean(preds, 0)
      print(preds.shape)
      height, width = preds.shape[1], preds.shape[2]

      # COLOR_MAP = dict(
      #     IGNORE=(0, 0, 0),
      #     Background=(255, 255, 255),
      #     Building=(255, 0, 0),
      #     Road=(255, 255, 0),
      #     Water=(0, 0, 255),
      #     Barren=(159, 129, 183),
      #     Forest=(0, 255, 0),
      #     Agricultural=(255, 195, 128),
      # )

      mapping = {
          0: (0, 0, 0),
          1: (255, 255, 255),
          2: (255, 0, 0),
          3: (255, 255, 0),
          4: (0, 0, 255),
          5: (159, 129, 183),
          6: (0, 255, 0),
          7: (255, 195, 128)}

      class_idx = torch.argmax(preds, dim=0)

      image = torch.zeros(height, width, 3, dtype=torch.uint8)

      for k in mapping:
        
        idx = (class_idx == torch.tensor(k, dtype=torch.uint8))
        validx = (idx == 1)
        image[validx,:] = torch.tensor(mapping[k], dtype=torch.uint8)
                
      image = image.permute(2, 0, 1)

      # image = image.permute(1,2,0)
      image = image.squeeze().cpu().numpy()

      #preds = (preds > 0.5).float()
    # torchvision.utils.save_image(preds, f'{folder}/pred_{index}.png')
    print(image.shape)
    image = torch.from_numpy(image/255) 
    torchvision.utils.save_image(image, f'{folder}/pred_{index}.png')
    # torchvision.utils.save_image(orig, f'{folder}/{index}.png')

  model.train()
