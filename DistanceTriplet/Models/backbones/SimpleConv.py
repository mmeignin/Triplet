import torch.nn as nn
import torch.nn.functional as F
from ipdb import set_trace
import torch

class SimpleConv(nn.Module):
    def __init__(self):
        super().__init__()

        self.pool =  nn.MaxPool2d(2)
        self.conv1 = nn.Conv2d(3, 6, 3,stride=1)
        self.nm1 = nn.BatchNorm2d(6, track_running_stats=False)

        self.conv2 = nn.Conv2d(6, 12, 3,stride=1)
        self.nm2 = nn.BatchNorm2d(12, track_running_stats=False)

        self.conv3 = nn.Conv2d(12, 16, 3,stride=1)
        self.nm3 = nn.BatchNorm2d(16, track_running_stats=False)

        self.conv4 = nn.Conv2d(16, 24, 3,stride=1)
        self.nm4 = nn.BatchNorm2d(24, track_running_stats=False)


    def forward(self, x):
        """Extract features from image

        Parameters
        ----------
        x : tensor (Bimage, Channels, W , H)

        Returns
        -------
        features : tensor (B, C*W*H)

        """
        x = F.relu(self.pool(self.nm1(self.conv1(x))))
        x = F.relu(self.pool(self.nm2(self.conv2(x))))
        x = F.relu(self.pool(self.nm3(self.conv3(x))))
        x = F.relu(self.pool(self.nm4(self.conv4(x))))
        x = x.flatten(1) # (B, C*W*H)
        return x


    def get_output_feats(self, img_size) :
        inp = torch.rand((1,3,img_size[0], img_size[1]))
        feats = self.forward(inp)
        return feats.shape[1]
