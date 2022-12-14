import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()

        m = models.resnet18(pretrained=True)
        self.out_features = m.fc.in_features
        self.feature_extractor = nn.Sequential(*list(m.children())[:-1])

    def forward(self, x):
        """Extract features from image

        Parameters
        ----------
        x : tensor (Bimage, Channels, W , H)

        Returns
        -------
        features : tensor (B, C*W*H)


        """
        x=self.feature_extractor(x)
        x=x.flatten(1)
        return x #

    def get_output_feats(self) :
        return self.out_features
