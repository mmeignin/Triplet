import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class Vit(nn.Module):
    def __init__(self, pretrained_backbone=False, **kwargs):
        super().__init__()
        m = models.vit_b_16(pretrained=pretrained_backbone)
        print(f'Using pretraining : {pretrained_backbone}')
        self.out_features = m.heads.head.out_features
        self.feature_extractor = m

    def forward(self, x):
        """Extract features from image

        Parameters
        ----------
        x : tensor (Bimage, Channels, W , H)

        Returns
        -------
        features : tensor (B, C*W*H)
        classification is done for this transformer module

        """
        x=self.feature_extractor(x)
        x=x.flatten(1)
        return x #

    def get_output_feats(self, img_size) :
        return self.out_features
