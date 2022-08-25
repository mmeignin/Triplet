import pytorch_lightning as pl
from einops import rearrange
from Models.backbones.SimpleConv import SimpleConv
from Models.backbones.ResNet18 import ResNet18
from Models.backbones.ResNet34 import ResNet34
from Models.backbones.Vgg11 import Vgg11
from Models.backbones.Vit import Vit

from argparse import ArgumentParser


class LitBackbone(pl.LightningModule) :
    def __init__(self, backbone, **kwargs) :
        super().__init__()
        self.model = self.init_model(backbone, **kwargs)

    def init_model(self, backbone, **kwargs) :
        if backbone == 'SimpleConv' :
            return SimpleConv()
        elif backbone == 'ResNet18':
            return ResNet18(**kwargs)
        elif backbone == 'ResNet34':
            return ResNet34(**kwargs)
        elif backbone == 'Vgg11':
            return Vgg11(**kwargs)
        elif backbone == 'Vit' :
            return Vit(**kwargs)
        else :
            print(f'Backbone {backbone} not available')

    def forward(self, batch) :
        """Compute feature for each input frame.
        Parameters
        ----------
        batch : Dict containing
            'Video' (Nvideos, Nframes, channels, w, H)

        Returns
        -------
        batch : add field 'Features' ( Nvideos, NFrames, Nfeats)
        """
        b, f, c , w, h = batch['Video'].shape
        input =  rearrange(batch['Video'], 'b f c w h -> (b f) c w h') # (Nvideos*NFrames, channels, w, h)
        features = self.model(input) # (Nvideos*NFrames, Nfeats)
        output = rearrange(features, ' (b f) s -> b f s', b=b , f=f)
        return output

    @staticmethod
    def add_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--backbone', '-bb', type=str, choices=['SimpleConv', 'ResNet18', 'ResNet34', 'Vgg11','Vit'], default='ResNet18')
        parser.add_argument('--pretrained_backbone', '-pb', action='store_true', help='Use pretrained backbone')
        return parser

    def get_output_feats(self, img_size) :
        return self.model.get_output_feats(img_size)
