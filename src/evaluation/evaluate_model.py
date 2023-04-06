import pickle 
import numpy as np
import torch
from src.util.eval_utils import ResultsStorer, get_neighbours, iou
from src.data.prepare_data import mesh_to_voxel, custom_load_obj
from src.data.dataset import ValidationDatasetScannet,ValidationDatasetShapenet
from src.networks.basic_net import Encoder 
from pytorch3d.loss import chamfer_distance
from tqdm import tqdm
from typing import Tuple

K = 1
def collate_fn_val_scannet(batch):
    batch = list(filter(lambda x : x is not None, batch))
    if len(batch) == 0:
        return None
    imgs = [item['scannet_frame'] for item in batch]
    for i,img in enumerate(imgs):
        if img.dim() == 3:
            imgs[i] = img.unsqueeze(0)
    
    imgs = torch.cat(imgs, 0)
    
    original_img = [item['original_image'] for item in batch]
    original_img = torch.hstack(original_img).squeeze(0)
    gt_objects = [item['gt_objects'] for item in batch]
    gt_objects = [item for sublist in gt_objects for item in sublist]
    categories = [item['categories'] for item in batch]
    categories = [item for sublist in categories for item in sublist]
    pointclouds = [item['pointclouds'].unsqueeze(0) for item in batch]
    pointclouds = torch.hstack(pointclouds).squeeze(0)
    voxels = [item['voxels'].unsqueeze(0) for item in batch]
    voxels = torch.hstack(voxels).squeeze(0)
    frame_id = [item['frame_id'] for item in batch]
    frame_id = [item for sublist in frame_id for item in sublist]
    labels = [item['labels'] for item in batch]
    labels = [item for sublist in labels for item in sublist]
    random_voxels = [item['random_voxels'].unsqueeze(0) for item in batch]
    random_voxels = torch.hstack(random_voxels).squeeze(0)
    random_pointclouds = [item['random_pointclouds'].unsqueeze(0) for item in batch]
    random_pointclouds = torch.hstack(random_pointclouds).squeeze(0)
    
    return {'scannet_frame': imgs, 
            'original_image': original_img,
            'gt_objects': gt_objects,
            'categories': categories,
            'pointclouds': pointclouds,
            'voxels': voxels,
            'frame_id': frame_id,
            'labels': labels,
            'random_voxels': random_voxels,
            'random_pointclouds': random_pointclouds}

def get_dataloaders(config:dict)->Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    scan_dataset = ValidationDatasetScannet(config)
    shape_dataset = ValidationDatasetShapenet(config)

    scan_dataloader = torch.utils.data.DataLoader(scan_dataset, batch_size=64, shuffle=False, collate_fn=collate_fn_val_scannet)
    shape_dataloader = torch.utils.data.DataLoader(shape_dataset, batch_size=16, shuffle=False)

    return scan_dataloader, shape_dataloader

def evaluate(config):
    results_file = 'src/cached/val_results.pkl'
    category_list = ['02818832','04256520','03001627','02933112','04379243','02871439', '02747177']
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scan_model = Encoder().to(device)
    shape_model = Encoder().to(device)

    ckpt_scan = f'src/runs/scan_model.ckpt'
    scan_model.load_state_dict(torch.load(ckpt_scan))
    ckpt_shape = f'src/runs/shape_model.ckpt'
    shape_model.load_state_dict(torch.load(ckpt_shape))

    scan_model.eval()
    shape_model.eval()

    count = 0
    total_chamfer_dist = 0
    total_random_chamfer_dist = 0
    total_iou = 0
    total_random_iou = 0
    category_wise_total = {}
    category_wise_total_iou = {}
    category_wise_total_random = {}
    category_wise_total_iou_random = {}
    category_wise_count = {}

    scan_dataloader, shape_dataloader = get_dataloaders(config)

    try:
        next(iter(shape_dataloader))
    except:
        pass    

    for b, scan_batch in enumerate(tqdm(scan_dataloader)): 
        if b == 25:
            break
        results = []
        scan_embeddings = scan_model(scan_batch['scannet_frame'])
        
        for j in range(len(scan_embeddings)):
            results.append(ResultsStorer(K))
            
        for i, batch in enumerate(tqdm(shape_dataloader)):
            views = batch['rendered_views'].view(-1,3,config["width"], config["width"])
            shape_embeddings = shape_model(views)
            pointclouds = batch['pointclouds'].view(-1, 5000, 3)
            voxels = batch['voxels'].view(-1, 1, 32, 32, 32)
            shape_id = [item for item in batch['shape_id'] for j in range(6)]
            similarity_matrix, neighbours = get_neighbours(scan_embeddings, shape_embeddings)
            for j in range(len(scan_embeddings)):
                idx = neighbours[0:K, j]
                dist = similarity_matrix[idx, j]
                pcl = pointclouds[idx]
                s_id = [shape_id[index] for index in idx] 
                results[j].update_results(s_id, dist, list(pcl), list(views[idx]), list(voxels[idx]))

        for j in range(len(scan_embeddings)):
            category = scan_batch['categories'][j]
            if category not in category_list:
                continue
            min_chamfer = 1e5
            min_chamfer_random = 1e5
            max_iou = 0.
            max_iou_random = 0.
            for k in range(K):
                chamfer_dist = chamfer_distance(scan_batch['pointclouds'][j].unsqueeze(0) ,results[j].get_results()[2][k].unsqueeze(0))[0].item()
                iou_batch = iou(scan_batch['voxels'][j], results[j].get_voxel()[k])
                chamfer_dist_random = chamfer_distance(scan_batch['random_pointclouds'][j] ,results[j].get_results()[2][k].unsqueeze(0))[0].item()
                iou_batch_random = iou(scan_batch['random_voxels'][j], results[j].get_voxel()[k])
                
                if chamfer_dist < min_chamfer:
                    min_chamfer = chamfer_dist
                if chamfer_dist_random < min_chamfer_random:
                    min_chamfer_random = chamfer_dist_random    
                if iou_batch > max_iou:
                    max_iou = iou_batch
                if iou_batch_random > max_iou_random:
                    max_iou_random = iou_batch_random    
            #Get category from batch, and add to both count and total dicts
            if category not in category_wise_count.keys():
                category_wise_count[category] = 1
                category_wise_total[category] = min_chamfer
                category_wise_total_iou[category] = max_iou
                category_wise_total_random[category] = min_chamfer_random
                category_wise_total_iou_random[category] = max_iou_random
            else:
                category_wise_count[category] += 1
                category_wise_total[category] += min_chamfer
                category_wise_total_iou[category] += max_iou
                category_wise_total_random[category] += min_chamfer_random
                category_wise_total_iou_random[category] += max_iou_random
            total_chamfer_dist += min_chamfer
            total_iou += max_iou
            total_random_chamfer_dist += min_chamfer_random
            total_random_iou += max_iou_random
            count += 1

        print("IOU %f Chamfer Dist %f"%(total_iou/count, total_chamfer_dist/count))

    results = (count, total_chamfer_dist, total_iou, category_wise_total, category_wise_total_iou, category_wise_count, 
            total_random_chamfer_dist, total_random_iou, category_wise_total_random, category_wise_total_iou_random)
    with open(results_file, 'wb') as file:
        pickle.dump(results, file)

    print("Done")
    print("IOU per category: %s"%str(category_wise_total_iou))
    print("Count per category: %s"%str(category_wise_count))    
    return 

if __name__=='__main__':
    config = {
    "width": 224,
    "height": 224,
    "device": 'cuda:0',
    "batch_size": 32,
    'lr': 1e-5,
    "epochs": 20
    }
    evaluate(config)