import torch
import torch.nn as nn
from torch import Tensor
from matplotlib import cm
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image
from PIL import Image, ImageOps

import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as pyplot
import cv2
import datetime
import random

# hooks
class My_hook_lstm:
    '''out, (h_n, c_n) = self.lstm1(x)'''
    def __init__(self):
        self.out_map = []
        self.inp_map = []
    def forward_hook(self, module, inp, outp):
        # defines hook for LSTM
        self.inp_map.append(inp)     # (torch.Size([12, 35, 2]))
        self.out_map.append(outp)    # (torch.Size([12, 35, 1]), (torch.Size([1, 35, 1]),torch.Size([1, 35, 1])))

class My_hook_GCN:
    def __init__(self):
        self.out_map = []
        self.inp_map = []
    def forward_hook(self, module, inp, outp):
        # defines hook for GCN
        self.inp_map.append(inp)     # ([torch.Size([35, 12]),torch.Size([35, 35]))
        self.out_map.append(outp)    # torch.Size([35, 35])

class My_hook_linear:
    def __init__(self):
        self.weights_map = []
        self.inp_map = []
    def forward_hook(self, module, inp, outp):
        # defines hook for linear
        self.inp_map.append(inp)      # (torch.Size([35, 35]))
        weights = module.weight.data  # torch.Size([1, 35])
        self.weights_map.append(weights)

class Hook_process:
    def overlay_mask(self, img: Image.Image, mask: Image.Image, colormap: str = 'jet', alpha: float = 0.6) -> Image.Image:
                """Overlay a colormapped mask on a background image
                Args:
                    img: background image
                    mask: mask to be overlayed in grayscale
                    colormap: colormap to be applied on the mask
                    alpha: transparency of the background image
                Returns:
                    overlayed image
                """
                if not isinstance(img, Image.Image) or not isinstance(mask, Image.Image):
                    raise TypeError('img and mask arguments need to be PIL.Image')
                if not isinstance(alpha, float) or alpha < 0 or alpha >= 1:
                    raise ValueError('alpha argument is expected to be of type float between 0 and 1')

                cmap = cm.get_cmap(colormap)    
                # Resize mask and apply colormap
                overlay = mask.resize(img.size, resample=Image.BICUBIC)
                overlay = (255 * cmap(np.asarray(overlay) ** 2)[:, :, 1:]).astype(np.uint8)
                # Overlay the image with the mask
                overlayed_img = Image.fromarray((alpha * np.asarray(img) + (1 - alpha) * overlay).astype(np.uint8))
                return overlayed_img

    def convert_to_3D(self, image):
        height, width = image.shape[:2]
        # Create a 3D matrix to hold the converted image
        new_image = np.zeros((height, width, 3), dtype=np.uint8)
    
        # Assigns each pixel value of the two-dimensional image to each channel of the new three-dimensional matrix
        for i in range(height):
            for j in range(width):
                new_image[i][j] = [image[i][j], image[i][j], image[i][j]]
    
        return new_image

    def min_max_normalize(self, arr):
        min_val = torch.min(arr)
        max_val = torch.max(arr)
        normalized_arr = (arr - min_val) / (max_val - min_val)
        return normalized_arr

    def process_hook(self, my_hook_lstm, my_hook_GCN1, my_hook_GCN2, my_hook_linear):
        # my_hook_lstm.inp_map     # (torch.Size([12, 35, 2]))
        # my_hook_lstm.out_map    # (torch.Size([12, 35, 1]), (torch.Size([1, 35, 1]),torch.Size([1, 35, 1])))
        # # defines hook for GCN
        # my_hook_GCN.inp_map     # ([torch.Size([35, 12]),torch.Size([35, 35]))
        # my_hook_GCN.out_map    # torch.Size([35, 35])
        # # defines hook for linear
        # my_hook_linear.inp_map      # (torch.Size([35, 35]))
        # my_hook_linear.weights_map  # torch.Size([1, 35])
        indexSameple = random.sample(range(1, 22), 5)
        now = datetime.datetime.now()
        strNow = now.strftime("%Y-%m-%d_%H-%M-%S.%f")
        for index in [1]:
            data1 = my_hook_lstm.inp_map[-index][0][:,:,0]
            data1_1 = my_hook_lstm.inp_map[-index][0][:,:,1]
            data2 = my_hook_GCN1.inp_map[-index][0]
            data2_1 = my_hook_GCN1.inp_map[-index][1]
            data3 = my_hook_GCN2.inp_map[-index][0]
            data3_1 = my_hook_GCN2.inp_map[-index][1]
            data4 = my_hook_linear.inp_map[-index][0]
            weights = my_hook_linear.weights_map[-index]
            self.arrayToIMG(data1,strNow,'A_',index)
            self.arrayToIMG(data1_1,strNow,'A_1_',index)
            self.arrayToIMG(data2,strNow,'B_',index)
            self.arrayToIMG(data2_1,strNow,'B_1_',index)
            self.arrayToIMG(data3,strNow,'C_',index)
            self.arrayToIMG(data3_1,strNow,'C_1_',index)
            self.arrayToIMG(data4,strNow,'D',index)
            self.arrayToIMG(weights,strNow,'W',index)

    def arrayToIMG(self, arr, strNow, step, index):
        file = open("example.txt", "a")

        save_path1 = 'D:\\pythoncode\\Paper\\stocks\\regressionDay-PredictNextday-MutilstockGnnbySignal-MyModel\\imgs\\{}_{}{}.png'.format(strNow,step,index)
        arr_norm = self.min_max_normalize(arr).cpu()
        data = (arr_norm * 255.0).type(torch.uint8)  # converts data types 
        result = ""
        i, j = 0, 0
        for row in  data:
            i+=1
            j=0
            for item in row:
                j+=1
                if j==1:
                    if i==1:
                        result = "\\node[minimum height=1em, minimum width=1em, fill=gray!{}]at(0,0)(A{}{}){};"\
                            .format(int(item), i, j, "{}")
                    else:
                        result = "\\node[minimum height=1em, minimum width=1em, fill=gray!{},below=(0cm of A{}{})](A{}{}){};"\
                            .format(int(item), i-1, 1, i, j, "{}")
                else:
                    result = "\\node[minimum height=1em, minimum width=1em, fill=gray!{},right=(0cm of A{}{})](A{}{}){};"\
                        .format(int(item), i, j-1, i, j, "{}")
                file.writelines(result)
                file.writelines('\n')
        file.close()
        data_3D = self.convert_to_3D(data.detach().numpy())
        orign_img = Image.fromarray(data_3D)
        # shows image
        # pyplot.imshow(data)
        pyplot.imshow(data_3D)
        # pyplot.imshow(orign_img)
        pyplot.savefig(save_path1)