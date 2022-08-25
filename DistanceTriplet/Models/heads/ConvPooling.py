import torch.nn as nn
from einops import rearrange
import torch
from ipdb import set_trace
from csvflowdatamodule.utils import NBClass


class ConvPooling(nn.Module):
    def __init__(self, input_feats):
        super().__init__()
        self.input_feats = input_feats
        self.fc1 = nn.Linear(input_feats, input_feats//4) # 4096 for (125,125)
        self.fc2 = nn.Linear(input_feats//4, NBClass)

    def forward(self, x):
        """Extract features from image

        Parameters
        ----------
        x : tensor (Nvideos, Nfeats, NFrames)

        Returns
        -------
        output : tensor (Nvideos, 8)
        """
        Nvideos = x.shape[0]
        x, _ = torch.max(x, dim=-1) # (Nvideos, Nfeats)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        assert x.shape==(Nvideos, NBClass), f'Dimension output should be ({Nvideos},{NBClass}) and is ({x.shape})'
        return x
