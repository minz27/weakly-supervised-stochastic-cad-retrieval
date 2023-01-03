import torch
import torchvision
import matplotlib.pyplot as plt
import cv2
import pandas as pd
import numpy as np
from torchsummary import summary
from pathlib import Path
from tqdm import tqdm
import pickle
import sys

from src.dataset import OverfitDatasetScannet, OverfitDatasetShapenet
from src.networks.basic_net import Encoder 
from src.util.normal_similarity import self_similarity_normal_histogram, calculate_histogram_iou, scale_tensor
from src.data.prepare_data import retrieve_instances, transform_normal_map
from src.util.losses import VGGPerceptualLoss

#TODO: create modules instead of appending path like this
sys.path.append('../..')

K = 5 #Number of nearest neighbours to evaluate

def loss_to_evaluate(normal_1:torch.tensor, normal_2:torch.tensor, 
                    hist_1:np.ndarray, hist_2:np.ndarray, alpha:float=1, beta:float=1)->float:
    '''
    Define the loss to evaluate here
    '''
    perceptual_distance = vggloss(normal_1, normal_2)
    hist_distance = 1 - calculate_histogram_iou(hist_1, hist_2)
    weighted_distance = (alpha * perceptual_distance + beta * hist_distance)
    return weighted_distance

def find_k_nearest_neighbours(masked_imgs, masked_normals, views, shape_normals, shape_batch, alpha:float=1., beta:float=1.)->pd.DataFrame:
    #Calculate self-similarity histograms of scans and shapes
    scan_histograms = []
    shape_histograms = []

    print('Calculating scan histograms...')
    for i in tqdm(range(masked_imgs.shape[0])):
        scan_normal = masked_normals[i].permute(1,2,0).detach().cpu().numpy()
        scan_normal_mask = ((scan_normal[:,:,2] != 0) | (scan_normal[:,:,1] != 0) | (scan_normal[:,:,0] != 0))
        scan_hist = self_similarity_normal_histogram(scan_normal, scan_normal_mask)
        scan_histograms.append(scan_hist)

    print('Calculating shape histograms...')
    for i in tqdm(range(shape['normal_maps'].view(-1, 3, config["width"], config["height"]).shape[0])):
        shape_normal = shape['normal_maps'].view(-1, 3, config["width"], config["height"])[i].permute(1,2,0).detach().cpu().numpy()
        shape_normal_mask = ((shape_normal[:,:,2] != 0) | (shape_normal[:,:,1] != 0) | (shape_normal[:,:,0] != 0))
        shape_hist = self_similarity_normal_histogram(shape_normal, shape_normal_mask)
        shape_histograms.append(shape_hist)  

    #Dump histograms to pickle so that we don't have to compute it again
    with open('runs/histograms.pkl', 'wb') as file:
        all_histograms = (scan_histograms, shape_histograms)
        pickle.dump(all_histograms, file)    

    # with open('runs/histograms.pkl', 'rb') as file:
    #     scan_histograms, shape_histograms = pickle.load(file) 

    all_distances = {}
    print('Calculating distances...')
    for i in range(len(scan_histograms)):
        distances = []
        for j in range(len(shape_histograms)):
            loss = loss_to_evaluate(masked_normals[i].unsqueeze(0).cpu(), shape_normals[j].unsqueeze(0).cpu(),
                                    scan_histograms[i], shape_histograms[j],
                                    alpha, beta)
            distances.append(loss)
        all_distances[i] = distances
    
    df = pd.DataFrame().from_dict(all_distances, orient='index', columns = shape_names)
    df.to_csv('runs/distances.csv', index = False) #Saving it so that I don't have to calculate it everytime

    nearest_neighbours = {}
    for i, (scan, dist) in enumerate(tqdm(all_distances.items())):
        sorted_distances = np.argsort(dist)
        neighbours = []
        for j in range(K):
            idx = sorted_distances[j]
            shape_name = shape_batch['cat_id'][idx] + '/' + shape_batch['model_id'][idx]
            neighbours.append(shape_name)
        
        nearest_neighbours[i] = neighbours        
    
    neighbours_df = pd.DataFrame().from_dict(nearest_neighbours, orient='index')
    neighbours_df.to_csv('runs/neighbours.csv', index=False)
  
    return neighbours_df    

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
    #    3. get chamfer distance of the normalised objects and store it
    #    4. plot a few random images and retrieved neighbours to get a feel of what's happening

    scannet = OverfitDatasetScannet(config, split="src/splits/evaluate_scannet_split.txt")
    shapenet = OverfitDatasetShapenet(config, split="src/splits/evaluate_shapenet_split.txt")

    scannetloader = torch.utils.data.DataLoader(scannet, batch_size=len(scannet), shuffle=False)
    shapenetloader = torch.utils.data.DataLoader(shapenet, batch_size=len(shapenet), shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    valid_labels = [
    3,  4,    5,    6,    7,    10,    12,    14,    15,    35,    37,    39
    ]

    scan = next(iter(scannetloader))
    masked_imgs = retrieve_instances(scan["img"], scan["mask"], valid_labels, config["n_instances"]) #2xNxCxWxH
    masked_imgs = masked_imgs.view(-1, 3, config["width"], config["height"]) #(2N)xCxWxH
    masked_normals = retrieve_instances(scan["normal"], scan["mask"], valid_labels, config['n_instances']).view(-1, 3, config["width"], config["height"])
                    
    shape = next(iter(shapenetloader))
    views = shape['rendered_views'].view(-1,3,config["width"], config["height"])
    shape_normals = shape['normal_maps'].view(-1, 3, config["width"], config["height"])

    # DF Containing K-Nearest Neighbours
    nearest_neighbours = find_k_nearest_neighbours(masked_imgs, masked_normals, views, shape_normals, shape, alpha=1., beta=1.)    

if __name__=='__main__':
    evaluate_similarity_measure()   