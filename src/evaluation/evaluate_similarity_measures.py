import torch
import torchvision
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import numpy as np
from torchsummary import summary
from pathlib import Path

from src.dataset import OverfitDatasetScannet, OverfitDatasetShapenet
from src.networks.basic_net import Encoder 
from src.util.normal_similarity import self_similarity_normal_histogram, calculate_histogram_iou, scale_tensor
from src.data.prepare_data import retrieve_instances, transform_normal_map

def evaluate_similarity_measure():
    
    config = {
    "width": 240,
    "height": 240,
    "device": 'cuda:0',
    "batch_size": 4,
    "n_instances": 2
    }

    # Download all the framenet files for my split first -> important
    # For now, I'll simply try with overfitdataset
    # Since I havent figured out how to retrieve the masks from scan2cad yet
    # my goal right now is to 
    #    1. retrieve the k nearest objects for each masked object in the image
    #    2. for each of the k objects - find the object in ground truth that is closest to it
    #    3. get chamfer distance of the normalised objects

    scannet = OverfitDatasetScannet(config)
    shapenet = OverfitDatasetShapenet(config)

    scannetloader = torch.utils.data.DataLoader(scannet, batch_size=len(scannet), shuffle=False)
    shapenetloader = torch.utils.data.DataLoader(shapenet, batch_size=len(shapenet), shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")