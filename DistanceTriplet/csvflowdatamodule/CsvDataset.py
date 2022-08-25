from torch.utils.data import DataLoader, Dataset
import pandas as pd
import torch
from argparse import ArgumentParser
import flowiz
from PIL import Image
import numpy as np
from ipdb import set_trace
import pytorch_lightning as pl
import os
from pathlib import Path
from torchvision import transforms, io
from torch.utils.data.dataloader import default_collate
from .Transforms import TransformsComposer
import torchvision
from tqdm import tqdm
import psutil
from natsort import natsort_keygen

class FilesLoaders() :
    def __init__(self) :
        self.loaders = {
                        'Flow' :  self.load_flow,
                        'Image' :  self.load_image,
                        'GtMask' :  self.load_mask,
                        'FlowRGB' : self.load_image
                       }

    def load_file(self, path, type, img_size=None) :
        file = self.loaders[type](path)
        if img_size is not None:
            if not file.shape[-2:] == img_size:
                resize = transforms.Resize(img_size)
                file = resize(file[None])[0] # You need to add and remove a leading dimension because resize doesn't accept (W,H) inputs
        return file

    @staticmethod
    def load_flow(flow_path) :
        flow = torch.tensor(flowiz.read_flow(flow_path)).permute(2, 0, 1) # 2, i, j
        assert flow.ndim == 3, f'Wrong Number of dimension : {image_path}'
        return flow

    @staticmethod
    def load_image(image_path) :
        #im = (io.read_image(image_path)/255.) - 0.5 # c, i, j : [-0.5; 0.5]
        im = (torch.tensor(np.array(Image.open(image_path))).permute(2,0,1)/255.) - 0.5 # c, i, j : [-0.5; 0.5]
        assert im.ndim == 3, f'Wrong Number of dimension : {image_path}'
        return im

    @staticmethod
    def load_mask(image_path) :
        im = torch.tensor(np.array(Image.open(image_path))/255., dtype=torch.long) # i, j, c
        if im.ndim == 3 :
            im = im[:,:,0]
        assert im.ndim == 2, f'Wrong Number of dimension : {image_path}'
        return im



class CsvDataset(Dataset) :
    def __init__(self,
        data_path: str,
        base_dir: str,
        img_size: tuple, # i, j
        request: list, # List of fields you want in the folder
        subsample = 1,# percemtage of the dataset available ( if this is under 1 we subsample randomly )
        transform=None,
        preload_cache=False):
     super().__init__()

     self.img_size = img_size
     self.transform = transform
     self.base_dir = base_dir
     self.files = pd.read_csv(data_path)
     if subsample < 1 :
         self.files = self.files.sample(frac=subsample, random_state=123)
     print('request : ', request)
     assert set(request).issubset(self.files.columns), f'CSV is missing some requested columns columns : {list(self.files.columns)} Request : {request}'
     self.available_request = self.files.columns
     self.request = request.copy()
     if 'GtMask' in self.available_request : self.request.add('GtMask')
     self.fl = FilesLoaders()
     self.cache = {}
     if preload_cache :
         self.preload_cache()

    def __len__(self) :
        return len(self.files)

    def getter(self, idx, function) :
        ret = {}
        loc = self.files.iloc[idx]
        for type in self.request :
         try :
             if isinstance(loc[type], (np.integer, np.float)) :
                 assert loc[type] is not np.nan, f'Error in the Datasplit in {type}'
                 ret[type] = loc[type]
             elif isinstance(loc[type], (str)) :
                 ret[f'{type}Path'] =  loc[type]
                 ret[type] = function(loc[type], type, self.img_size)
             else :
                 raise Exception(f'Data type of {loc[type]} not handled')
         except Exception as e :
             print(e) # File does not exist
             return None
        return ret

    def __getitem__(self, idx):
        ret =  self.getter(idx, self.load)
        if ret is None : return None
        return self.transform(ret)

    @staticmethod
    def cache_key(filename, type, img_size) :
        return (filename, type, str(img_size)) # Key for dict

    def preload_cache(self) :
        for idx in tqdm(range(len(self.files)), desc='Preload Cache') :
             self.getter(idx, self.preload)
        print(f'Dataset preloaded\n\t Size : {len(self.cache)} RAM usage : {psutil.Process().memory_info().rss / (1024 * 1024 * 1000):.2f} Gb')

    def preload(self, filename, type, img_size) :
        #set_trace()
        key = self.cache_key(filename, type, img_size)
        self.cache[key] = self.fl.load_file(os.path.join(self.base_dir, filename), type, self.img_size)
        pass

    def load(self, filename, type, img_size) :
        key = self.cache_key(filename, type, img_size)
        if key in self.cache :
            return self.cache[key].clone()
        else :
            #print(f'Not in cache {key}')
            return self.fl.load_file(os.path.join(self.base_dir, filename), type, self.img_size)


class VideoDataset(CsvDataset) :
        def __init__(self, framestep=1, **kwargs) :
            super().__init__(**kwargs)
            videos = self.files['VideoIdx'].unique()
            self.framestep=framestep
            self.videos_idx = dict(zip(range(len(videos)), videos))

        def __getitem__(self, video_idx) :
            video_name = self.videos_idx[video_idx]
            samples = self.files[self.files['VideoIdx']  == video_name]
            samples_list = list(samples.sort_values('FrameIdx', key=natsort_keygen()).index)
            ret_list = []
            for isample  in samples_list[::self.framestep] :
                ret = self.getter(isample, self.load)
                if ret is None : return None
                ret_list.append(ret)
            video = torch.stack([ret['Image'] for ret in ret_list])
            classes = torch.tensor(np.stack([ret['Class'] for ret in ret_list]))
            assert len(classes.unique()) == 1, 'Several Labels for the same class'
            ret = {'Video' : video, 'Class' : classes[0], 'VideoName' : video_name}
            return self.transform(ret)

        def __len__(self) :
            return len(self.videos_idx)



class CsvDataModule(pl.LightningDataModule):
    def __init__(self, data_path: str,
                       base_dir: str,
                       batch_size: int,
                       request: list, # List of fields you want in the folder
                       img_size : tuple,
                       subsample_train=1, # percentage of the train data to use for training.
                       framestep=1,
                       shuffle_fit=True,
                       preload_cache=False,
                       **kwargs) :
        super().__init__()

        self.data_path = data_path
        self.batch_size = batch_size
        self.img_size = img_size
        self.base_dir = base_dir
        self.request = request
        self.subsample_train = subsample_train
        self.shuffle_fit = shuffle_fit
        self.preload_cache = preload_cache
        self.framestep = framestep
        self.kwargs_dataloader = {'batch_size':self.batch_size,
                                  'collate_fn' : self.collate_fn,
                                  'persistent_workers':False,
                                  'num_workers': 8,#0 for gpu
                                  'pin_memory':True,
                                  'drop_last': False}
        self.set_transformations(**kwargs)

    def set_transformations(self, augmentation, val_augment=False, **kwargs) :
        self.transforms = {}
        self.transforms['train'] = TransformsComposer(augmentation)
        self.transforms['val'] = TransformsComposer('')
        self.transforms['test'] = TransformsComposer('')

    def setup(self, stage=None):
        print(f'Loading data in : {self.data_path} ------ Stage : {stage}')
        if stage == 'fit' or stage is None:
             self.dtrain = VideoDataset(data_path=self.data_path.format('train'), base_dir=self.base_dir, img_size=self.img_size, request=self.request,
                                        subsample=self.subsample_train, transform=self.transforms['train'], preload_cache=self.preload_cache, framestep=self.framestep)
             self.dval = VideoDataset(data_path=self.data_path.format('val'), base_dir=self.base_dir, img_size=self.img_size, request=self.request,
                                        subsample=self.subsample_train, transform=self.transforms['val'], preload_cache=self.preload_cache, framestep=self.framestep)
        if stage == 'test' or stage is None: # For now the val and test are the same
             self.dtest = VideoDataset(data_path=self.data_path.format('test'), base_dir=self.base_dir, img_size=self.img_size, request=self.request,
                                        transform=self.transforms['test'], preload_cache=self.preload_cache, framestep=self.framestep)
        self.size(stage)

    def size(self, stage=None) :
        print('Size of dataset :')
        if stage == 'fit' or stage is None:
            print(f'\tTrain : {self.dtrain.__len__()} \t Val : {self.dval.__len__()}')
        if stage == 'test' or stage is None:
            print(f'\t Test : {self.dtest.__len__()}')

    @staticmethod
    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))
        return default_collate(batch)

    def train_dataloader(self):
        return DataLoader(self.dtrain, **self.kwargs_dataloader ,shuffle=self.shuffle_fit)

    def val_dataloader(self):
        return DataLoader(self.dval, **self.kwargs_dataloader, shuffle=self.shuffle_fit)

    def test_dataloader(self):
        return DataLoader(self.dtest, **self.kwargs_dataloader, shuffle=False)

    def get_sample(self, set=None) :
        if set == "train"  : return next(iter(self.train_dataloader()))
        elif set == "val"  : return next(iter(self.val_dataloader()))
        elif set == "test" : return next(iter(self.test_dataloader()))

    def get_dataloader(self, set=None) :
        if set == "train"  : return self.train_dataloader()
        elif set == "val"  : return self.val_dataloader()
        elif set == "test" : return self.test_dataloader()

    @staticmethod
    def add_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser = TransformsComposer.add_specific_args(parser)
        parser.add_argument('--batch_size', default=1, type=int)
        parser.add_argument('--framestep', default=1, type=int, help='Temporal subsampling in video')
        parser.add_argument('--subsample_train', default=1, type=float)
        parser.add_argument('--img_size', nargs='+', type=int, default=[256, 256])
        parser.add_argument('--base_dir', type=str, default=os.environ['PWD'])
        parser.add_argument('--data_file', type=str)
        parser.add_argument('--preload_cache', action='store_true', help='Enable data Augmentation in validation')
        return parser
        
##---------------------------------------
"""
 class tripletVideoDataset(CsvDataset) :
    def __init__(self, base_path, filenames_filename, triplets_file_name, transform=None,
                 loader=default_image_loader):

        self.base_path = base_path  
        self.filenamelist = []
        for line in open(filenames_filename):
            self.filenamelist.append(line.rstrip('\n'))
        triplets = []
        for line in open(triplets_file_name):
            triplets.append((line.split()[0], line.split()[1], line.split()[2])) # anchor, far, close
        self.triplets = triplets
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        path1, path2, path3 = self.triplets[index]
        img1 = self.loader(os.path.join(self.base_path,self.filenamelist[int(path1)]))
        img2 = self.loader(os.path.join(self.base_path,self.filenamelist[int(path2)]))
        img3 = self.loader(os.path.join(self.base_path,self.filenamelist[int(path3)]))
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
            img3 = self.transform(img3)

        return img1, img2, img3

    def __len__(self):
        return len(self.triplets)

"""