import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch


from torchmetrics import ConfusionMatrix
class WeightedClassificationError(torch.nn.Module):
    n_classes = 8
    Wmax=10
    W = torch.tensor (
            [[0, 1, 6, 10, 10, 10, 10, 10],
                [1, 0, 3, 10, 10, 10, 10, 10],
                [6, 3, 0, 2, 9, 10, 10, 10],
                [10, 10, 2, 0, 9, 9, 10, 10],
                [10, 10, 9, 9, 0, 8, 8, 8],
                [10, 10, 10, 9, 8, 0, 9, 8],
                [10, 10, 10, 10, 8, 9, 0, 9],
                [10, 10, 10, 10, 8, 8, 9, 0],]
        )
    def __init__(
        self, name="WeightedClassificationError", precision=2, time_idx=0
    ):
        self.precision = precision

    def compute(self, y_true, y_pred,device):
        confmat = ConfusionMatrix(num_classes=8).to(device)
        loss = torch.sum(torch.multiply(confmat(y_pred,y_true),self.W))/ (self.n_classes * self.Wmax)
        return loss

    def __call__(self, y_true, y_pred,device):
        y_pred = y_pred.to(device)
        y_true = y_true.to(device)
        return self.compute(y_true, y_pred,device) 





import torch
wce = WeightedClassificationError()
x1 = torch.tensor([[1 , 0 ,0 ,0 ,0 ,0 ,0 ,0]],requires_grad=True,dtype=torch.float32)
y= torch.tensor([[0,1,0,0,1,1,1,0]],dtype=torch.int32)
a=wce.compute(y,x1,y.device)
print(a)















class Vit(nn.Module):
    def __init__(self, pretrained_backbone=False, **kwargs):
        super().__init__()
        m = models.vit_b_16(pretrained=pretrained_backbone)
        print(f'Using pretraining : {pretrained_backbone}')
        self.out_features = m.heads.head.out_features
        self.feature_extractor = m
        """er tensor
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




#m=Vit()
#import torch
#x=torch.rand(1,3,256,256)

#print(m(x).shape)


#for name, layer in m.named_modules():
#    print(name, layer)
