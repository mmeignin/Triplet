import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt


def displayvideo(ret,nb_displayed=1,fig_size=(18,4)):

    if nb_displayed > ret['Video'].shape[1] :
        print("Impossible display, size format incorrect")
    else :
        fig, axs = plt.subplots(1,nb_displayed,figsize=fig_size)
        fig.suptitle("Video number:"+str(ret['VideoName'])+" Video Class: "+str(ret['Class'].item()))

        #Constant frame display rate
        frame_step = int(ret['Video'].shape[1] / nb_displayed)

        for i in range(nb_displayed):
            axs[i].imshow(ret['Video'][0,i*(frame_step)].permute(1,2,0)+0.5)
            axs[i].set_title(f"Frame_number: {i*(frame_step)}")
    return fig

