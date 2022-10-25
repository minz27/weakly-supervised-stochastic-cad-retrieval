import torch
import numpy as np

from src.dataset import OverfitDatasetScannet, OverfitDatasetShapenet
from src.networks.basic_net import Encoder 

def train(model, device, config, scannetloader, shapenetloader):
    
    optimizer = torch.optim.Adam(model.parameters(), lr = config['lr'])
    #Add criterion
    #Add training loop
    #Torch summary writer
    #Add gradient scaling in the actual model