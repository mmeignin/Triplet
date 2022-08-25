from torchvision import transforms
import torch
from argparse import ArgumentParser
from ipdb import set_trace
import numpy as np

class TransformsComposer():
    """
    Compose and setup the transforms depending command line arguments.
    Define a series of transforms, each transform takes a dictionnary
    containing a subset of keys from [ 'Video'] and
    has to return the same dictionnary with content elements transformed.
    """
    def __init__(self, augmentation) :
        transfs = []
        self.augmentations = TrAugmentVideo(augmentation)
        transfs.append(self.augmentations)
        self.TrCompose = transforms.Compose(transfs)

    def __call__(self, ret) :
        return self.TrCompose(ret)

    @staticmethod
    def add_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = TrAugmentVideo.add_specific_args(parser)
        return parser


class TrAugmentVideo() :
    """
    Data augmentation techniques for videos

    Args :
        Name augmentation (list str) : data augmentation to return
    """
    def __init__(self, augmentation) :
        self.augs = []
        augs_names =  augmentation
        if (augmentation =='none') or (augmentation ==''):
            pass
        else : 
            for name in augs_names :
                self.interpret_name(name)
        self.declare()

    def interpret_name(self, name) :
        if 'randombrightness' == name :
            self.augs.append(self.randombrightness)
        elif 'hflip' == name :
            self.augs.append(self.hflip)
        elif 'vflip' == name :
            self.augs.append(self.vflip)
        elif 'fill_background' == name :
            self.augs.append(self.fill_background) 
        elif (name == 'none') or (name=='') :
            pass
        else :
            raise Exception(f'Video augmentation {name} is unknown')

    def __call__(self, ret) :
        """
        Call all augmentations defined in the init
        """
        for aug in self.augs :
            ret=aug(ret)
        return ret

    def declare(self):
         print(f'Video Transformations : {[aug for aug in self.augs]}')

    @staticmethod
    def randombrightness(ret) :
        """
        Apply a random brightness adjustment to all the video
        Args :
          ret : dictionnary containing at least "Video"
        Return :
          ret dictionnary containg video with adjusted brightness
              'Flow' : (Nframes, Channels ,W, H)
        """
        fct = torch.rand(1) + 1
        ret['Video'] += 0.5
        ret['Video'] = transforms.functional.adjust_brightness(ret['Video'], fct)
        ret['Video'] -= 0.5
        return ret

    @staticmethod
    def hflip(ret) :
        """
        Horizontal Flip
        Args :
          ret : dictionnary containing at least "Video"
        Return :
          ret dictionnary with Video horizontally flipped the same way for all images
              'Video' : (Nframes, Channels ,W, H)
        """
        hflipper = transforms.RandomHorizontalFlip(p=0.5)
        ret['Video'] = hflipper(ret['Video'])
        return ret

    @staticmethod
    def vflip(ret) :
        """
        Vertical Flip
        Args :
          ret : dictionnary containing at least "Video"
        Return :
          ret dictionnary with Video vertically flipped the same way for all images
              'Video' : (Nframes, Channels ,W, H)
        """
        vflipper = transforms.RandomVerticalFlip(p=0.5)
        ret['Video'] = vflipper(ret['Video'])
        return ret

    @staticmethod
    def fill_background(ret) :
        """
        fill the background of segmented images with random values
        Args :
          ret : dictionnary containing at least "Video"
        Return :
          ret dictionnary with randomized background the same way for all images
              'Video' : (Nframes, Channels ,W, H)
        """
        for i in range(ret['Video'].shape[0]):
            if ret['Video'][i,0,:,:][ret['Video'][i,0,:,:] == - 0.5].any() and ret['Video'][i,1,:,:][ret['Video'][i,1,:,:] == - 0.5].any() and ret['Video'][i,2,:,:][ret['Video'][i,2,:,:] == - 0.5].any() :
                fct=torch.rand(ret['Video'][i,0,:,:][ret['Video'][i,0,:,:] == - 0.5].shape[0]) -0.5
                ret['Video'][i,0,:,:][ret['Video'][i,0,:,:] == - 0.5] = fct 
                ret['Video'][i,1,:,:][ret['Video'][i,1,:,:] == - 0.5] = fct 
                ret['Video'][i,2,:,:][ret['Video'][i,2,:,:] == - 0.5] = fct 
        return ret

    

    @staticmethod
    def add_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--augmentation',type=str,nargs='+', default='none')

        return parser
