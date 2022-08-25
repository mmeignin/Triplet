from Models.heads.Lstm import Lstm
import pytorch_lightning as pl
from einops import rearrange
from Models.heads.NaiveHead import NaiveHead
from Models.heads.Gru import Gru
from Models.heads.ConvPooling import ConvPooling
from argparse import ArgumentParser

class LitHead(pl.LightningModule) :
    def __init__(self, input_feats, head, **kwargs) :
        super().__init__()
        self.input_feats = input_feats
        self.model = self.init_model(head, input_feats)

    def init_model(self, head, input_feats) :
        if head == 'NaiveHead' :
            return NaiveHead(input_feats)
        elif head == 'Gru':
            return Gru(input_feats)
        elif head == 'Lstm':
            return Lstm(input_feats)
        elif head == 'ConvPooling':
            return ConvPooling(input_feats)
        else :
            print(f'Head {Head} not available')

    def forward(self, batch) :
        """Compute feature for each input frame.

        Parameters
        ----------
        batch : 'FeatureBackbone' (Nvideos, NFrames, Nfeats)

        Returns
        -------
        batch : add field 'Features' ( Nvideos, 8)
        """
        input =  rearrange(batch['FeatureBackbone'], 'videos frames features -> videos features frames', features=self.input_feats) #(Nvideos, Nfeats, NFrames)
        output = self.model(input) # ( Nvideos, 8)
        return output

    @staticmethod
    def add_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--head', '-he', type=str, choices=['NaiveHead', 'Gru', 'ConvPooling','Lstm'], default='ConvPooling')
        return parser


