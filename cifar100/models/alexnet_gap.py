import torch
import torch.nn as nn

class AlexNet_GAP(nn.Module):
      def __init__(self, classes=100):
    super(AlexNet_GAP, self).__init__()
    self.features = nn.Sequential(
      nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2),
      nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=3, stride=2),
      nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
      nn.ReLU(inplace=True),
    )

    self.pool =  nn.Sequential(
      nn.AvgPool2d(3)
    )
    self.classifier = nn.Sequential(
      nn.Linear(256, classes),
    )

  def forward(self, x):
    x = self.features(x)
    x = self.pool(x)
    x = torch.flatten(x, 1)
    x = self.classifier(x)
    return x

