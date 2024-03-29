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
from src.util.normal_similarity import self_similarity_normal_histogram, calculate_histogram_iou, scale_tensor, sharpen_normal, angular_loss
from src.data.prepare_data import retrieve_instances, transform_normal_map
from src.util.losses import VGGPerceptualLoss

K = 10 #Number of nearest neighbours to evaluate
SCAN2CAD_PATH = '/mnt/raid/mdeka/scan2cad/scan2cad_image_alignments.json'
category_list = ['02818832','04256520','03001627','02933112','04379243','02871439', '02747177']
vggloss = VGGPerceptualLoss()

chamfer_distance_thresholds = {
    "02818832" : 0.00012615480227395892,
    "04256520": 0.00010117214696947485,
    "03001627": 0.0010825844947248697,
    "02747177": 0.0002355660981265828,
    "02933112": 0.0008038535015657544,
    "03211117": 0.00017238210421055555,
    "04379243": 0.0004430929257068783,
    "02871439": 0.00015336726210080087,
}

vgg16 = torchvision.models.vgg16(pretrained=True).features[:4].eval()

def loss_to_evaluate(normal_1:torch.tensor, normal_2:torch.tensor, 
                     alpha:float=1., beta:float=1.)->float:
    '''
    Define the loss to evaluate here
    '''
    content_loss, style_loss = vggloss(abs(normal_1), abs(normal_2))

    # content_loss = torch.nn.functional.l1_loss(output_normal_1, output_normal_2).item()
    output_normal_1 = vgg16(abs(normal_1))
    output_normal_2 = vgg16(abs(normal_2))
    content_loss = torch.nn.functional.l1_loss(output_normal_1[:,9,:,:], output_normal_2[:,9,:,:]).item()

    return content_loss, style_loss

def find_k_nearest_neighbours(masked_imgs, masked_normals, views, shape_normals, shape_batch, alpha:float=1., beta:float=1., gamma:float=0., n_views:int=3)->pd.DataFrame:
    # Create file names based on parameters
    filename = '_' + str(alpha) + '_' + str(beta) + '_' + str(gamma) + '.csv'
    #Calculate self-similarity histograms of scans and shapes
    # scan_histograms = []
    # shape_histograms = []

    # print('Calculating scan histograms...')
    # for i in tqdm(range(masked_imgs.shape[0])):
    #     scan_normal = masked_normals[i].permute(1,2,0).detach().cpu().numpy()
    #     # scan_normal = sharpen_normal(scan_normal)
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
    # with open('src/runs/new_histograms_v7.pkl', 'wb') as file:
    #     all_histograms = (scan_histograms, shape_histograms)
    #     pickle.dump(all_histograms, file)    

    # print('Retrieving cached self-similarity histograms...')
    # with open('src/runs/new_histograms_v7.pkl', 'rb') as file:
    #     scan_histograms, shape_histograms = pickle.load(file) 

    # Calculate the distances and cache them to quickly calculate different combinations
    all_distances = {}
    # all_hist_distances = {}
    all_content_losses = {}
    all_style_losses = {}

    # print('Calculating distances...')
    # for i in tqdm(range(len(masked_imgs))):
    #     # distances = []
    #     # hist_distances = []
    #     content_losses = []
    #     style_losses = []

    #     for j in range(len(views)):
    #         content_loss, style_loss = loss_to_evaluate((masked_normals[i].unsqueeze(0).cpu()), shape_normals[j].unsqueeze(0).cpu(),
    #                                 alpha, beta)
            
    #         content_losses.append(content_loss)
    #         style_losses.append(style_loss / 10000)

    #     all_content_losses[i] = content_losses
    #     all_style_losses[i] = style_losses
    
    # with open('src/runs/cached_distances_new_large.pkl', 'wb') as file:
    #     all_losses = (all_content_losses, all_style_losses)
    #     pickle.dump(all_losses, file)

    print('Retrieving cached distances...')
    with open('src/runs/cached_distances_new_large.pkl', 'rb') as file:
        all_content_losses, all_style_losses = pickle.load(file)

    print('Calculating distance combination...')
    for k, v in all_content_losses.items():
        # hist_distances = [alpha*x for x in all_hist_distances[k]]
        style_losses = [(beta/10)*x for x in all_style_losses[k]]
        content_losses = [gamma*x for x in all_content_losses[k]]   
        # all_distances[k] = list(map(add, hist_distances, style_losses, content_losses))
        all_distances[k] = [x[0] + x[1] for x in zip(style_losses, content_losses)]     

    df = pd.DataFrame().from_dict(all_distances, orient='index')
    df.to_csv('src/runs/distances.csv', index = False) #Saving it so that I don't have to calpculate it everytime

    nearest_neighbours = {}
    nearest_neighbours_distances = {}

    print('Calculating neighbours...')
    for i, (scan, dist) in enumerate(tqdm(all_distances.items())):
        sorted_distances = np.argsort(dist)
        neighbours = []
        distances = []
        j = 0
        j_loop = 0
        
        while j < K:
            idx = sorted_distances[j_loop]
            if shape_batch['cat_id'][idx // n_views] not in category_list:
                j_loop+=1
                continue
            shape_name = shape_batch['cat_id'][idx // n_views] + '/' + shape_batch['model_id'][idx // n_views]
            if shape_name not in neighbours:
                neighbours.append(shape_name)
                distances.append(dist[idx])
                j+=1
                j_loop+=1
            else:
                neighbours.append(shape_name)
                distances.append(dist[idx])
                j+=1
                j_loop+=1    

        # for j in range(K):
        #     idx = sorted_distances[j]
        #     shape_name = shape_batch['cat_id'][idx // n_views] + '/' + shape_batch['model_id'][idx // n_views]
        #     neighbours.append(shape_name)
        
        nearest_neighbours[i] = neighbours  
        nearest_neighbours_distances[i] = distances      
    
    neighbours_df = pd.DataFrame().from_dict(nearest_neighbours, orient='index')
    neighbours_df.to_csv(f'src/runs/neighbours_top{K}_new_v2_' + filename, index=False)
  
    return nearest_neighbours, nearest_neighbours_distances    

def evaluate_similarity_measure(masked_imgs, masked_normals, views, shape_normals, shape, scan, n_views, labels, indices, alpha:float=1, beta:float=1, gamma:float=1):
    results_file = f'results_analysis_proxy_loss_large.csv'
    
    with open(results_file, 'w') as file:
        file.write('category,n_positive_retrievals,sum_dist_positive_retrievals,gt_retrieved,n_cat_retrievals,sum_dist_cat_retrievals\n')
    # DF Containing K-Nearest Neighbours
    nearest_neighbours, nearest_neighbours_distances = find_k_nearest_neighbours(masked_imgs, masked_normals, views, shape_normals, shape, alpha=alpha, beta=beta, gamma=gamma, n_views=n_views) 

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
        if obj['catid_cad'] in category_list:
            gt_objects[i] = obj['catid_cad'] + '/' + obj['id_cad']

    # Create a mapping between model hash and pointclouds
    mapping = [x[0] + '/'+ x[1] for x in zip(shape['cat_id'], shape['model_id'])] 

    #Cache chamfer distance for each ground truth object as well
    min_chamfer_gt = {}
    for key, value in gt_objects.items():
        min_chamfer = 1e5
        gt_pcl = shape['pointcloud'][mapping.index(value)]
        for i in range(shape["pointcloud"].shape[0]):
            if (shape["cat_id"][i] + '/' + shape["model_id"][i]) == value:
                continue 
            c_dist = chamfer_distance(gt_pcl, shape["pointcloud"][i])
            if c_dist[0].item() < min_chamfer:
                min_chamfer = c_dist[0].item()
        
        min_chamfer_gt[key] = min_chamfer   
    
    with open('src/runs/cached_min_chamfer_large.pkl', 'wb') as file:
        pickle.dump(min_chamfer_gt, file)                  

    print('Retrieving minimum distances...')
    with open('src/runs/cached_min_chamfer_large.pkl', 'rb') as file:
        min_chamfer_gt = pickle.load(file)

    # For each frame find the chamfer distance between ground truth and nearest neighbours
    chamfer_distances = []
    
    for i in tqdm(range(len(labels))):
        if i not in gt_objects:
            continue
        gt_retrieved = False    
        retrievals = nearest_neighbours[i]
        gt_obj = gt_objects[i]   
        
        category = gt_obj.split('/')[0]
        # threshold = chamfer_distance_thresholds[category]
        threshold = min_chamfer_gt[i]
        n_positive_retrievals = 0
        n_cat_retrievals = 0
        sum_dist_positive_retrievals = 0
        sum_dist_cat_retrievals = 0

        min_chamfer = 1e5
        # Use the model IDs to fetch the pointclouds and get the minimum distance
        for j in range(len(retrievals)):
            category_ret = retrievals[j].split('/')[0]
            if category_ret == category:
                n_cat_retrievals += 1
                sum_dist_cat_retrievals += nearest_neighbours_distances[i][j].item()
            if gt_obj == retrievals[j]:
                gt_retrieved = True

            chamfer = chamfer_distance(shape['pointcloud'][mapping.index(gt_obj)], shape['pointcloud'][mapping.index(retrievals[j])])
            if chamfer[0].item() <= threshold:
                n_positive_retrievals += 1
                sum_dist_positive_retrievals += nearest_neighbours_distances[i][j].item()
            
            # chamfer = chamfer_distance(transformed_pcl.unsqueeze(0), shape['pointcloud'][mapping.index(ret_obj)])
            # sum_chamfer += chamfer[0].item()
            if chamfer[0].item() < min_chamfer:
                min_chamfer = chamfer[0].item()
    
        chamfer_distances.append(min_chamfer)

        with open(results_file, 'a') as file:
            file.write('%s,%s,%s,%s,%s,%s\n'%(category,n_positive_retrievals,sum_dist_positive_retrievals,gt_retrieved,n_cat_retrievals,sum_dist_cat_retrievals))
        # chamfer_distances.append(sum_chamfer / K)

    filename = '_' + str(alpha) + '_' + str(beta) + '_' + str(gamma) + '.pkl'
    
    # with open('src/runs/chamfer' + filename, 'wb') as file:
    #     pickle.dump(min_chamfer_distances, file)
    chamfer_distances = np.array(chamfer_distances)
    
    print("Mean chamfer distance%s"%(chamfer_distances.mean()))
    return chamfer_distances.mean()
    # return 0

if __name__=='__main__':
    config = {
    "width": 224,
    "height": 224,
    "device": 'cuda:0',
    "batch_size": 4,
    "n_instances": 2,
    "alpha": 0,
    "beta": 0.8,
    "gamma": 0.1
    }

    results_file = f'results_analysis_proxy_loss_large.csv'

    scannet = OverfitDatasetScannet(config, split="src/splits/analyse_similarity_split_scannet.txt")
    shapenet = OverfitDatasetShapenet(config, split="src/splits/analyse_similarity_split_shapenet.txt")

    scannetloader = torch.utils.data.DataLoader(scannet, batch_size=len(scannet), shuffle=False)
    shapenetloader = torch.utils.data.DataLoader(shapenet, batch_size=len(shapenet), shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    valid_labels = [
    3,  4,    5,    6,    7,    10,    12,    14,    15,    35,    37,    39
    ]

    scan = next(iter(scannetloader))
    masked_imgs, labels, indices = retrieve_instances(scan["img"], scan["mask"], valid_labels, config["n_instances"], scan['frame'], scan["R"], rotate=False ) #2xNxCxWxH
    masked_imgs = masked_imgs.view(-1, 3, config["height"], config["width"]) #(2N)xCxWxH
    masked_normals = retrieve_instances(scan["normal"], scan["mask"], valid_labels, config['n_instances'], scan['frame'], scan["R"])[0].view(-1, 3, config["height"], config["width"]).float()
    # masked_normals = retrieve_instances(scan["normal"], scan["mask"], valid_labels, config['n_instances'], scan['frame'])[0]  

    try:                
        shape = next(iter(shapenetloader))
    except:
        shape = next(iter(shapenetloader))

    views = shape['rendered_views'].view(-1,3,config["width"], config["width"])
    shape_normals = shape['normal_maps'].view(-1, 3, config["width"], config["width"])

    n_views = shape['rendered_views'].shape[2] #No of views for each model

    loss = evaluate_similarity_measure(masked_imgs, masked_normals, views, shape_normals, shape, scan, n_views, labels, indices, config["alpha"], config["beta"], config["gamma"]) 
    

    # for param in parameter_set:
    #     alpha, beta, gamma = param

    #     print('Evaluating %s %s %s'%(alpha, beta, gamma))    
    #     loss = evaluate_similarity_measure(masked_imgs, masked_normals, views, shape_normals, shape, scan, n_views, labels, indices, alpha, beta, gamma) 
    #     with open(results_file, 'a') as file:
    #         file.write('%s,%s,%s,%s\n'%(alpha, beta, gamma, loss))

    # alphas = np.linspace(0,1,10)
    # # alphas = [0]
    # betas = np.linspace(0,1,10)
    # gammas = np.linspace(0,1,10)

    # for alpha in alphas:
    #     for beta in betas:
    #         for gamma in gammas:
    #             print('Evaluating %s %s %s'%(alpha, beta, gamma))    
    #             loss = evaluate_similarity_measure(masked_imgs, masked_normals, views, shape_normals, shape, scan, n_views, labels, indices, alpha, beta, gamma) 
    #             with open(results_file, 'a') as file:
    #                 file.write('%s,%s,%s,%s\n'%(alpha, beta, gamma, loss))         
