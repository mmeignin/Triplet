
from csvflowdatamodule.CsvDataset import CsvDataModule
import os
import matplotlib.pyplot as plt
from torchvision import transforms
import torch
from visualisation import displayvideo
import time
import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
#class_names=['A','B','C','D','E','F','G','H'])})
from PIL import Image
import torchvision.transforms as transforms



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.cuda.get_device_name()

PATH=os.getcwd()+"/DataSplit/Embryon_RandomSplit_"
data_path= "/home/maiage/mmeignin/Stage/"
train_df = pd.read_csv(PATH+"train.csv")
val_df = pd.read_csv(PATH+"val.csv")
test_df = pd.read_csv(PATH+"test.csv")
df=train_df

def piltotensor(images,data_path):
    
    
    im = (torch.tensor(np.array(Image.open(data_path+str(images)))).permute(2,0,1)/255.) - 0.5
    return im
class MNIST(Dataset):
    def __init__(self, df, transform=None,data_path=""):
        self.transform = transform
        self.labels = df['Class'].unique()
        self.files=df['Image'].unique()
        self.index=df['VideoIdx'].unique()
        self.df=df
        self.data_path=data_path

        
    def __len__(self):
        return len(self.index)
    
    def __getitem__(self, item):
        iitem=self.index[item]
        anchor_images = self.df[self.df['VideoIdx']==iitem]
        anchor_label = self.df["Class"][self.df['VideoIdx']==iitem].unique()[0]


        positive_list = self.df[self.df['Class']==anchor_label][self.df['VideoIdx']!=iitem]
        positive_item = random.choice(positive_list.values.tolist())[2]
        positive_label = self.df["Class"][self.df['VideoIdx']==positive_item].unique()[0]
        positive_images = self.df[self.df['VideoIdx']==positive_item]
        
        negative_list = self.df[self.df['Class']!=anchor_label][self.df['VideoIdx']!=iitem]
        negative_item = random.choice(negative_list.values.tolist())[2]
        negative_label = self.df["Class"][self.df['VideoIdx']==negative_item].unique()[0]
        negative_images = self.df[self.df['VideoIdx']==negative_item]
        
        
        anchor_classes = torch.tensor(np.stack([ret[4] for ret in anchor_images.values]))
        anchor_video = torch.stack([piltotensor(ret[1],self.data_path)for ret in anchor_images.values])
        assert len(anchor_classes.unique()) == 1, 'Several Labels for the same class'
        anchor_ret = {'Video' : anchor_video, 'Class' : anchor_classes[0], 'VideoName' : iitem}

        positive_classes = torch.tensor(np.stack([ret[4] for ret in positive_images.values]))
        positive_video = torch.stack([piltotensor(ret[1],self.data_path)for ret in positive_images.values])
        assert len(positive_classes.unique()) == 1, 'Several Labels for the same class'
        positive_ret = {'Video' : positive_video, 'Class' : positive_classes[0], 'VideoName' : positive_item}

        negative_classes = torch.tensor(np.stack([ret[4] for ret in negative_images.values]))
        negative_video = torch.stack([piltotensor(ret[1],self.data_path)for ret in negative_images.values])
        assert len(negative_classes.unique()) == 1, 'Several Labels for the same class'
        negative_ret = {'Video' : negative_video, 'Class' : negative_classes[0], 'VideoName' : negative_item}

        return anchor_ret,positive_ret,negative_ret




train_ds = MNIST(train_df,data_path=data_path)
train_loader=DataLoader(train_ds,shuffle=True,num_workers=4)

test_ds = MNIST(test_df, data_path=data_path)
test_loader = DataLoader(test_ds,shuffle=False, num_workers=4)

val_ds = MNIST(val_df, data_path=data_path)
val_loader = DataLoader(val_ds,shuffle=False, num_workers=4)

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def L2_dist(self, x1, x2):
        return torch.dist(x1,x2)
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        distance_positive = self.L2_dist(anchor, positive)
        distance_negative = self.L2_dist(anchor, negative)
        losses = torch.relu(distance_positive - distance_negative + self.margin)

        return losses.mean()


from Models.backbones.ResNet18 import ResNet18
from Models.heads.Gru import Gru
from einops import rearrange

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.resnet=ResNet18()
        self.gru=Gru(self.resnet.get_output_feats(256))
    def forward(self, inputs):
        b, f, c , w, h = inputs['Video'].shape
        input =  rearrange(inputs['Video'], 'b f c w h -> (b f) c w h') # (Nvideos*NFrames, channels, w, h)
        features = self.resnet(input) # (Nvideos*NFrames, Nfeats)
        features = rearrange(features, ' (b f) s -> b s f', b=b , f=f)
        output = self.gru(features) # ( Nvideos, 8)
        return output



model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = TripletLoss()


import wandb
wandb.init()
epochs=5
model.train()
best_loss = 99999999


for epoch in range(epochs):
    running_loss = []
    loss_per_epoch=0
    for step, (anchor_img, positive_img, negative_img) in enumerate(train_loader):
        anchor_im = anchor_img
        positive_im = positive_img
        negative_im = negative_img
        
        optimizer.zero_grad()
        anchor_out = model(anchor_im)
        positive_out = model(positive_im)
        negative_out = model(negative_im)
        
        loss = criterion(anchor_out, positive_out, negative_out)
        loss.backward()
        optimizer.step()
        
        running_loss.append(loss.cpu().detach().numpy())
        wandb.log({'loss': loss})
        if(loss < best_loss):
            wandb.run.summary["best_loss"] = loss
            best_loss = best_loss
        loss_per_epoch = loss_per_epoch +loss
    wandb.log({'loss_per_epoch':loss_per_epoch})
    print("Epoch: {}/{} - Loss: {:.4f}".format(epoch+1, epochs, np.mean(running_loss)))



torch.save({"model_state_dict": model.state_dict(),
            "optimzier_state_dict": optimizer.state_dict()
           }, "trained_model.pth")

train_results = []
labels = []
model.eval()

with torch.no_grad():
    for step, (anchor_img, positive_img, negative_img) in enumerate(train_loader):
        data  = anchor_img
        pred = model(data)
        print(step)
        train_results.append((data['VideoName'],data['Class'],pred))
    for step, (anchor_img, positive_img, negative_img) in enumerate(test_loader):
        data  = anchor_img
        pred = model(data)
        print(step)
        train_results.append((data['VideoName'],data['Class'],pred))
    for step, (anchor_img, positive_img, negative_img) in enumerate(val_loader):
        data  = anchor_img
        pred = model(data)
        print(step)
        train_results.append((data['VideoName'],data['Class'],pred))   


df = pd.DataFrame(train_results)
df.to_csv(os.getcwd()+"results.csv")


print("Everything done")


