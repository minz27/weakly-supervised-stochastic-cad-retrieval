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
from operator import add
import json
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops import corresponding_points_alignment
import warnings
warnings.filterwarnings("ignore") #TODO: change it to filter the warnings that I don't want instead of everything

from src.dataset import OverfitDatasetScannet, OverfitDatasetShapenet
from src.networks.basic_net import Encoder 
from src.util.normal_similarity import self_similarity_normal_histogram, calculate_histogram_iou, scale_tensor
from src.data.prepare_data import retrieve_instances, transform_normal_map
from src.util.losses import VGGPerceptualLoss

K = 5 #Number of nearest neighbours to evaluate
SCAN2CAD_PATH = '/mnt/raid/mdeka/scan2cad/scan2cad_image_alignments.json'
vggloss = VGGPerceptualLoss()

def loss_to_evaluate(normal_1:torch.tensor, normal_2:torch.tensor, 
                    hist_1:np.ndarray, hist_2:np.ndarray, alpha:float=1., beta:float=1.)->float:
    '''
    Define the loss to evaluate here
    '''
    content_loss, style_loss = vggloss(normal_1, normal_2)
    hist_distance = 1 - calculate_histogram_iou(hist_1, hist_2)
    weighted_distance = (alpha * (style_loss / 1000) + beta * hist_distance)
    return weighted_distance, hist_distance, content_loss, style_loss

def find_k_nearest_neighbours(masked_imgs, masked_normals, views, shape_normals, shape_batch, alpha:float=1., beta:float=1., gamma:float=0., n_views:int=3)->pd.DataFrame:
    # Create file names based on parameters
    filename = '_' + str(alpha) + '_' + str(beta) + '_' + str(gamma) + '.csv'
    #Calculate self-similarity histograms of scans and shapes
    scan_histograms = []
    shape_histograms = []

    # print('Calculating scan histograms...')
    # for i in tqdm(range(masked_imgs.shape[0])):
    #     scan_normal = masked_normals[i].permute(1,2,0).detach().cpu().numpy()
    #     scan_normal_mask = ((scan_normal[:,:,2] != 0) | (scan_normal[:,:,1] != 0) | (scan_normal[:,:,0] != 0))
    #     scan_hist = self_similarity_normal_histogram(scan_normal, scan_normal_mask)
    #     scan_histograms.append(scan_hist)

    # print('Calculating shape histograms...')
    # for i in tqdm(range(shape_normals.shape[0])):
    #     shape_normal = shape_normals[i].permute(1,2,0).detach().cpu().numpy()
    #     shape_normal_mask = ((shape_normal[:,:,2] != 0) | (shape_normal[:,:,1] != 0) | (shape_normal[:,:,0] != 0))
    #     shape_hist = self_similarity_normal_histogram(shape_normal, shape_normal_mask)
    #     shape_histograms.append(shape_hist)  

    # #Dump histograms to pickle so that we don't have to compute it again
    # with open('src/runs/histograms.pkl', 'wb') as file:
    #     all_histograms = (scan_histograms, shape_histograms)
    #     pickle.dump(all_histograms, file)    

    print('Retrieving cached self-similarity histograms...')
    with open('src/runs/histograms.pkl', 'rb') as file:
        scan_histograms, shape_histograms = pickle.load(file) 

    # Calculate the distances and cache them to quickly calculate different combinations
    all_distances = {}
    all_hist_distances = {}
    all_content_losses = {}
    all_style_losses = {}

    # print('Calculating distances...')
    # for i in tqdm(range(len(scan_histograms))):
    #     distances = []
    #     hist_distances = []
    #     content_losses = []
    #     style_losses = []

    #     for j in range(len(shape_histograms)):
    #         loss, hist_loss, content_loss, style_loss = loss_to_evaluate(masked_normals[i].unsqueeze(0).cpu(), shape_normals[j].unsqueeze(0).cpu(),
    #                                 scan_histograms[i], shape_histograms[j],
    #                                 alpha, beta)
    #         distances.append(loss)
    #         hist_distances.append(hist_loss)
    #         content_losses.append(content_loss)
    #         style_losses.append(style_loss / 1000)

    #     all_distances[i] = distances
    #     all_hist_distances[i] = hist_distances
    #     all_content_losses[i] = content_losses
    #     all_style_losses[i] = style_losses
    
    # with open('src/runs/cached_distances_vgg.pkl', 'wb') as file:
    #     all_losses = (all_hist_distances, all_content_losses, all_style_losses)
    #     pickle.dump(all_losses, file)

    print('Retrieving cached distances...')
    with open('src/runs/cached_distances_vgg.pkl', 'rb') as file:
        all_hist_distances, all_content_losses, all_style_losses = pickle.load(file)

    print('Calculating distance combination...')
    for k, v in all_hist_distances.items():
        hist_distances = [alpha*x for x in all_hist_distances[k]]
        style_losses = [beta*x for x in all_style_losses[k]]
        content_losses = [gamma*x for x in all_content_losses[k]]   
        # all_distances[k] = list(map(add, hist_distances, style_losses, content_losses))
        all_distances[k] = [x[0] + x[1] + x[2] for x in zip(hist_distances, style_losses, content_losses)]     

    df = pd.DataFrame().from_dict(all_distances, orient='index')
    df.to_csv('src/runs/distances.csv', index = False) #Saving it so that I don't have to calpculate it everytime

    nearest_neighbours = {}

    print('Calculating neighbours...')
    for i, (scan, dist) in enumerate(tqdm(all_distances.items())):
        sorted_distances = np.argsort(dist)
        neighbours = []
        j = 0
        j_loop = 0
        # Need another variable?
        while j < K:
            idx = sorted_distances[j_loop]
            shape_name = shape_batch['cat_id'][idx // n_views] + '/' + shape_batch['model_id'][idx // n_views]
            if shape_name not in neighbours:
                neighbours.append(shape_name)
                j+=1
                j_loop+=1
            else:
                # neighbours.append(shape_name)
                # j+=1
                j_loop+=1    

        # for j in range(K):
        #     idx = sorted_distances[j]
        #     shape_name = shape_batch['cat_id'][idx // n_views] + '/' + shape_batch['model_id'][idx // n_views]
        #     neighbours.append(shape_name)
        
        nearest_neighbours[i] = neighbours        
    
    neighbours_df = pd.DataFrame().from_dict(nearest_neighbours, orient='index')
    neighbours_df.to_csv(f'src/runs/neighbours_top{K}_min_' + filename, index=False)
  
    return nearest_neighbours    

def evaluate_similarity_measure(masked_imgs, masked_normals, views, shape_normals, shape, scan, n_views, labels, indices, alpha:float=1, beta:float=1, gamma:float=1):

    # DF Containing K-Nearest Neighbours
    nearest_neighbours = find_k_nearest_neighbours(masked_imgs, masked_normals, views, shape_normals, shape, alpha=alpha, beta=beta, gamma=gamma, n_views=n_views) 

    # Get objects from scan2cad dataset for each scannet frame
    print('Getting ground truth objects...')
    scan2cad_img = {}

    with open(SCAN2CAD_PATH) as f:
        scan2cad_img = json.load(f)   

    frames = scan['frame']
    gt_objects = {}

    for i in tqdm(range(len(labels))):
        # objects = scan2cad_img['alignments'][labels[i]]
        # objects_list = []
        # for obj in objects:
        #     objects_list.append(obj['catid_cad'] + '/' + obj['id_cad'])
        # gt_objects[i] = objects_list   
        obj = scan2cad_img['alignments'][labels[i]][indices[i] - 1]
        gt_objects[i] = obj['catid_cad'] + '/' + obj['id_cad']

    # Create a mapping between model hash and pointclouds
    mapping = [x[0] + '/'+ x[1] for x in zip(shape['cat_id'], shape['model_id'])] 
    # For each frame find the chamfer distance between ground truth and nearest neighbours
    chamfer_distances = []
    
    for i in tqdm(range(len(labels))):
        retrievals = nearest_neighbours[i]
        gt_obj = gt_objects[i]   
        min_chamfer = 1e5
        max_chamfer = 0
        # Use the model IDs to fetch the pointclouds and get the minimum distance
        for ret_obj in retrievals:
            # R, T, S = corresponding_points_alignment(shape['pointcloud'][mapping.index(obj)], shape['pointcloud'][mapping.index(ret_obj)])
            # transformed_pcl = S*((R.squeeze(0) @ torch.permute(shape['pointcloud'][mapping.index(obj)].squeeze(0), (1,0))).permute(1,0))
            chamfer = chamfer_distance(shape['pointcloud'][mapping.index(gt_obj)], shape['pointcloud'][mapping.index(ret_obj)])
            # chamfer = chamfer_distance(transformed_pcl.unsqueeze(0), shape['pointcloud'][mapping.index(ret_obj)])
            # sum_chamfer += chamfer[0].item()
            if chamfer[0].item() < min_chamfer:
                min_chamfer = chamfer[0].item()
    
        chamfer_distances.append(min_chamfer)
        # chamfer_distances.append(sum_chamfer / K)

    filename = '_' + str(alpha) + '_' + str(beta) + '_' + str(gamma) + '.pkl'
    
    # with open('src/runs/chamfer' + filename, 'wb') as file:
    #     pickle.dump(min_chamfer_distances, file)
    chamfer_distances = np.array(chamfer_distances)
    
    print("Mean chamfer distance%s"%(chamfer_distances.mean()))
    return chamfer_distances.mean()

if __name__=='__main__':
    config = {
    "width": 240,
    "height": 180,
    "device": 'cuda:0',
    "batch_size": 4,
    "n_instances": 2
    }

    results_file = f'results_top_{K}.csv'

    scannet = OverfitDatasetScannet(config, split="src/splits/evaluate_scannet_split.txt")
    shapenet = OverfitDatasetShapenet(config, split="src/splits/evaluate_shapenet_split.txt")

    scannetloader = torch.utils.data.DataLoader(scannet, batch_size=len(scannet), shuffle=False)
    shapenetloader = torch.utils.data.DataLoader(shapenet, batch_size=len(shapenet), shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    valid_labels = [
    3,  4,    5,    6,    7,    10,    12,    14,    15,    35,    37,    39
    ]

    scan = next(iter(scannetloader))
    masked_imgs, labels, indices = retrieve_instances(scan["img"], scan["mask"], valid_labels, config["n_instances"], scan['frame']) #2xNxCxWxH
    masked_imgs = masked_imgs.view(-1, 3, config["height"], config["width"]) #(2N)xCxWxH
    masked_normals = retrieve_instances(scan["normal"], scan["mask"], valid_labels, config['n_instances'], scan['frame'])[0].view(-1, 3, config["height"], config["width"])
    # masked_normals = retrieve_instances(scan["normal"], scan["mask"], valid_labels, config['n_instances'], scan['frame'])[0]  

    try:                
        shape = next(iter(shapenetloader))
    except:
        shape = next(iter(shapenetloader))

    views = shape['rendered_views'].view(-1,3,config["width"], config["width"])
    shape_normals = shape['normal_maps'].view(-1, 3, config["width"], config["width"])

    n_views = shape['rendered_views'].shape[2] #No of views for each model

    # Try out different parameters
    alphas = [0, 0.25, 0.5, 0.75, 1]
    betas = [0, 0.25, 0.5, 0.75, 1]
    gammas = [0, 0.25, 0.5, 0.75, 1]

    with open(results_file, 'w') as file:
        file.write('alpha,beta,gamma,chamfer_distance\n')

    for alpha in alphas:
        for beta in betas:
            for gamma in gammas:
                print('Evaluating %s %s %s'%(alpha, beta, gamma))    
                loss = evaluate_similarity_measure(masked_imgs, masked_normals, views, shape_normals, shape, scan, n_views, labels, indices, alpha, beta, gamma) 
                with open(results_file, 'a') as file:
                    file.write('%s,%s,%s,%s\n'%(alpha, beta, gamma, loss))  