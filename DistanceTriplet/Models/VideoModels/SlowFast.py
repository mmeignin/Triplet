import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from pytorchvideo.models import slowfast
from pytorchvideo.models import net


class SlowFast(nn.Module):
    def __init__(self, pretrained_vm=True, **kwargs):
        super().__init__()

       
        print(f'Using pretraining : {pretrained_vm}')
        self.model= net.Net()


    def forward(self, x):
        """Extract features from Video

        Parameters
        ----------
        x : tensor (Nvideos, Channels, W , H)

        Returns
        -------
        features : tensor (B, C*W*H)
        """
        x=self.model(x)
        return x #

