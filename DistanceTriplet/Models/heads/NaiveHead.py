import torch.nn as nn
from csvflowdatamodule.utils import NBClass

class NaiveHead(nn.Module):
    def __init__(self, input_feats):
        super().__init__()

        self.conv1 = nn.Conv1d(input_feats, 20, kernel_size=1) # 4096 for (125,125)
        self.fc = nn.Linear(300*20, NBClass)

    def forward(self, x):
        """Extract features from image

        Parameters
        ----------
        x : tensor (Nvideos, Nfeats, NFrames)

        Returns
        -------
        output : tensor (Nvideos, NBClass)
        """
        x = self.conv1(x)
        x = x.flatten(1) # (Nvideos, NFrames*Dfeats)
        x = self.fc(x) #(Nvideos, NBClass)
        return x
