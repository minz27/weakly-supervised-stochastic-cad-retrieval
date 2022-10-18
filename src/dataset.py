from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
import json

class OverfitDataset(torch.utils.data.Dataset):
    def __init__(self, mode, scannet_root = "../scannet_frames_25k/", shapenet_root = "../shapenet", scan2cad_root = '../scan2cad'):
        
        self.images = []
        self.cads = []

        self.scannet_root = scannet_root
        self.shapenet_root = shapenet_root

        if (mode == 'overfit'):
            self.images.append()