import pytorch_lightning as pl
from LitBackbone import LitBackbone
from LitHead import LitHead
import torch.nn as nn
import torch
from ipdb import set_trace
from argparse import ArgumentParser
import wandb
from csvflowdatamodule.utils import NBClass

from losses.WeightedClassificationError import WeightedClassificationError
"""
	A LightningModule organizes your PyTorch code into 6 sections:
	-Computations (init).
	-Train Loop (training_step)
	-Validation Loop (validation_step)
	-Test Loop (test_step)
	-Prediction Loop (predict_step)
	-Optimizers and LR Schedulers (configure_optimizers)
"""

class LitClassificationModel(pl.LightningModule) :
    def __init__(self, criterion_name, **kwargs) :
        super().__init__()
        self.backbone_model = LitBackbone(**kwargs)
        self.head_model = LitHead(input_feats=self.backbone_model.get_output_feats(kwargs['img_size']), **kwargs)
        self.criterion_name = criterion_name
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        print(f'Optimizer : {optimizer}')
        return optimizer

    def prediction(self, batch) :
        """
        Produce a classification prediction using the model and the batch
        """
        batch['FeatureBackbone'] = self.backbone_model.forward(batch)
        batch['Pred'] = self.head_model.forward(batch)
        return batch

    def Criterion(self, batch) :
        """"wandb_logger = self.logger.experiment
        Comput loss for the batch
        """
        if self.criterion_name == 'bce' :
            losses = nn.functional.cross_entropy(batch['Pred'], batch['Class'], reduction='none')
        if self.criterion_name == 'bce_balanced' :
            if NBClass == 8 :
                class_weight = torch.tensor([1.21875,1.17672414,1.1375,1.625,1.1375,0.875,0.89802632,0.58836207]).to(batch['Class'].device)
            elif NBClass == 2 :
                #class_weight = torch.tensor([0.77118644,1.421875]).to(batch['Class'].device)
                class_weight = torch.tensor([1.2967032978432556, 0.7032967032967034]).to(batch['Class'].device)
            else :
                pass
            losses = nn.functional.cross_entropy(batch['Pred'], batch['Class'], reduction='none', weight=class_weight)
        if self.criterion_name == 'custom_loss':
            if NBClass == 8 :
                print("not working for now")
                #wce = WeightedClassificationError()
                #losses = wce.compute(batch['Pred'],batch['Class'],batch['Class'].device)
            else :
                print("Custom loss not available for Binary Classification!\n")
        return {'losses' : losses}

    def Evaluations(self, batch, evals) :
        """"Args : batch with at least keys
                'Class'
            Update
            Evals : 'loss','accs','Class'
        """
        evals['loss'] = evals['losses'].mean() # Loss is necessary for backprop in Pytorch Lightning
        a = batch['Pred'].argmax(axis=-1) #(Nvideos, 1)
        evals['preds'] =  a
        evals['accs'] = (a == batch['Class']).to(torch.float)
        evals['Class'] =  batch['Class']
        if NBClass == 8 and self.criterion_name != 'custom_loss' :
            wce = WeightedClassificationError()
            evals['WCE'] = wce.compute(batch['Pred'],batch['Class'],batch['Class'].device)
        for i in range(NBClass) :
            evals[f'preds_{i}'] = (a == i).to(torch.float)
      
    def Logging(self, evals, step_label) :
        """
        pl logger for wandb
        """
        bs = evals['losses'].shape[0]
        #if self.criterion_name != 'custom_loss' :
        log = lambda k : self.log(f'{step_label}/{k}', evals[k].mean(), on_epoch=True, on_step=False)
        #
        for k in ['losses', 'accs'] + [f'preds_{i}' for i in range(NBClass)]:
            evals[k] = evals[k].detach()
            log(k)
        if NBClass == 8:
            self.log('WCE custom loss',evals['WCE'].detach(),on_epoch=True, on_step=False)
        #else :
        #    log = lambda k : self.log(f'{step_label}/{k}', evals[k].mean(), on_epoch=True, on_step=False)
        #
        #    for k in ['accs'] + [f'preds_{i}' for i in range(NBClass)]:
        #        evals[k] = evals[k].detach()
        #        log(k)
        #    self.log('WCE custom loss as Main loss',evals['losses'].detach(),on_epoch=True, on_step=False)

    def step(self, batch, step_label):
        batch = self.prediction(batch)
        evals = self.Criterion(batch)
        self.Evaluations(batch, evals)
        self.Logging(evals, step_label)
        return evals, batch
    
    def training_step(self, batch, batch_idx) :
        evals, batch =  self.step(batch, 'train')
        return evals

    def validation_step(self, batch, batch_idx) :
        evals, batch =  self.step(batch, 'val')
        return evals

    def test_step(self, batch, batch_idx) :
        evals, batch =  self.step(batch, 'test')
        return evals

    @staticmethod
    def add_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = LitBackbone.add_specific_args(parser)
        parser = LitHead.add_specific_args(parser)
        parser.add_argument('--criterion_name', type=str, choices=['bce', 'bce_balanced','custom_loss'], default='bce')
        return parser
