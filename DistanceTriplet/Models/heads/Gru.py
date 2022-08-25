import torch.nn as nn
from einops import rearrange
from csvflowdatamodule.utils import NBClass

class Gru(nn.Module):
    def __init__(self, input_feats):
        super().__init__()
        self.input_feats = input_feats
        self.gru = nn.GRU(input_feats, NBClass, batch_first=True) # 4096 for (125,125)

    def forward(self, x):
        """Extract features from image

        Parameters
        ----------
        x : tensor (Nvideos, Nfeats, NFrames)

        Returns
        -------
        output : tensor (Nvideos, NBClass)
        """
        Nvideos = x.shape[0]
        x =  rearrange(x, 'videos features frames -> videos frames features', features=self.input_feats) #(Nvideos, Nfeats, NFrames)
        x, h = self.gru(x) # x : (Nvideos, NFrames, NBClass)
        x = x[:, -1, :] # (Nvideos, NBClass)
        assert x.shape==(Nvideos, NBClass), f'Dimension output should be ({Nvideos},{NBClass}) and is ({x.shape})'
        return x
