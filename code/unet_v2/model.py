import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(DoubleConv, self).__init__()
    self.conv = nn.Sequential(
      nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
      nn.BatchNorm2d(out_channels), # no bias bcz batch normalization
      nn.ReLU(inplace=True),

      nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
      nn.BatchNorm2d(out_channels), # no bias bcz batch normalization
      nn.ReLU(inplace=True),
    )

  def forward(self, x):
    return self.conv(x)


class UNET(nn.Module):
  # outchannels=1, bcz binary classification
  def __init__(
    self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]
  ):
    super(UNET, self).__init__()
    self.ups = nn.ModuleList()
    self.downs = nn.ModuleList()
    self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    # down part of UNET
    for feature in features:
      self.downs.append(DoubleConv(in_channels, feature))
      in_channels = feature

    # up part of UNET
    for feature in reversed(features):
      # upsampling part
      self.ups.append(
        nn.ConvTranspose2d(
          feature*2, feature, kernel_size=2, stride=2
        )
      )
      # double convolution
      self.ups.append(DoubleConv(feature*2, feature))

    # bottom layer
    self.bottleneck = DoubleConv(features[-1], features[-1]*2)
    # last output layer, 1x1 convolution
    self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

  def forward(self, x):
    skip_connections = []

    for down in self.downs:
      x = down(x)
      skip_connections.append(x)
      x = self.pool(x)

    x = self.bottleneck(x)
    skip_connections = skip_connections[::-1] # reversing the list

    for idx in range(0, len(self.ups), 2):
      x = self.ups[idx](x)
      skip_connection = skip_connections[idx//2]

      if x.shape != skip_connection.shape: # shape reduce due to max pooling
        x = TF.resize(x, size=skip_connection.shape[2:]) # skipping batch size and no of channels

      concat_skip = torch.cat((skip_connection, x), dim=1)
      x = self.ups[idx+1](concat_skip)
    
    return self.final_conv(x)

def test():
  # batch size, no of channels
  x = torch.randn((3, 1, 160, 160))
  model = UNET(in_channels=1, out_channels=1)
  preds = model(x)
  print(x.shape)
  print(preds.shape)
  assert preds.shape == x.shape

if __name__ == '__main__':
    test()
    