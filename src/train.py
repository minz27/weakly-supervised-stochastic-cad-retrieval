import torch
import numpy as np
import torchvision
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from tqdm import tqdm
from pytorch_metric_learning.losses import TripletMarginLoss
from pytorch_metric_learning.miners import MultiSimilarityMiner
from src.util.eval_utils import (
    ResultsStorer, 
    get_neighbours, 
    iou
)
from src.data.dataset import (
    RetrievalDataset, 
    ValidationDatasetScannet,
    ValidationDatasetShapenet
)
from src.networks.basic_net import Encoder 
from src.util.losses import SupervisedContrastiveLoss
from src.data.prepare_data import retrieve_instances, transform_normal_map

import wandb

def train(scan_model, shape_model, device, config, dataloader, scan_dataloader, shape_dataloader):
    wandb.init(project='cad_retrieval',reinit=True,  config = config)
    category_list = ['02818832','04256520','03001627','02933112','04379243','02871439', '02747177']
    #Create optimizer
    optimizer = torch.optim.Adam([
        {
            # TODO: optimizer params and learning rate for model (lr provided in config)
            'params' : scan_model.parameters(),
            'lr': config['lr']
        },
        {
            # TODO: optimizer params and learning rate for latent code (lr provided in config)
            'params': shape_model.parameters(),
            'lr': config['lr']
        }
    ])
    #Add scheduler
    # scheduler = ExponentialLR(optimizer, gamma=0.99)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True)

    #Add criterion
    # criterion = SupervisedContrastiveLoss(temperature=0.07)
    criterion = TripletMarginLoss()
    miner = MultiSimilarityMiner()
    history = []
    best_error = 1e5
    best_iou = 0.
    transform = torch.nn.functional.interpolate

    scan_model.to(device)
    shape_model.to(device)
    #bugbypass
    try:
        next(iter(dataloader))
    except:
        pass
    #Add training loop
    for epoch in tqdm(range(config['epochs'])):

        scan_model.train()
        shape_model.train()

        train_loss = 0
        epoch_loss = 0
        n_steps = 0    
        for i, batch in enumerate(dataloader):
            if batch == None:
                continue
            masked_imgs, rendered_views = batch['scannet_frame'], batch['shapenet_target']

            #Compute embeddings
            scan_embedding = scan_model(masked_imgs)
            shape_embedding = shape_model(rendered_views)
            
            labels = torch.cat((torch.arange(scan_embedding.shape[0]), torch.arange(scan_embedding.shape[0])))
            embeddings = torch.cat([scan_embedding, shape_embedding], dim=0)
            hard_pairs = miner(embeddings, labels)
            loss = criterion(embeddings, labels, hard_pairs)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            epoch_loss += loss
            n_steps += 1

        avg_loss = train_loss / n_steps
        print("Epoch:", epoch, " Loss:", avg_loss)
        history.append(avg_loss)
        wandb.log({'Loss': avg_loss})
        scheduler.step(epoch_loss)
        del batch

        # print("Running validation...")
        # scan_model.eval()
        # shape_model.eval()
        # count = 0
        # total_iou = 0
        
        # # for _, scan_batch in enumerate(tqdm(scan_dataloader)):
        # scan_batch = next(iter(scan_dataloader))
        # results = []
        # scan_embeddings = scan_model(scan_batch['scannet_frame'])
        
        # for j in range(len(scan_embeddings)):
        #     results.append(ResultsStorer())
            
        # for i, batch in enumerate(tqdm(shape_dataloader)):
        #     views = batch['rendered_views'].view(-1,3,config["width"], config["width"])
        #     shape_embeddings = shape_model(views)
        #     pointclouds = batch['pointclouds'].view(-1, 5000, 3)
        #     voxels = batch['voxels'].view(-1, 1, 32, 32, 32)
        #     shape_id = [item for item in batch['shape_id'] for j in range(6)]
        #     similarity_matrix, neighbours = get_neighbours(scan_embeddings, shape_embeddings)
        #     for j in range(len(scan_embeddings)):
        #         idx = neighbours[0, j]
        #         dist = similarity_matrix[idx, j]
        #         pcl = pointclouds[idx]
        #         s_id = [shape_id[idx]]
        #         results[j].update_results(s_id, [dist], list(pcl), list(views[idx]), list(voxels[idx]))

        # for j in range(len(scan_embeddings)):
        #     category = scan_batch['categories'][j]
        #     if category not in category_list:
        #         continue
        #     iou_batch = iou(scan_batch['voxels'][j], results[j].get_voxel()[0])
        #     total_iou += iou_batch
        #     count += 1

        # print("IOU %f"%(total_iou/count))
        
        # avg_iou = total_iou / count
        # wandb.log({'Val IoU': avg_iou})
        if (avg_loss < best_error):
            print("Saving model...")
            best_error = avg_loss
            torch.save(shape_model.state_dict(), f'src/runs/shape_model.ckpt')
            torch.save(scan_model.state_dict(), f'src/runs/scan_model.ckpt')
                        
    #Add gradient scaling in the actual model

    return 
    
def collate_fn(batch):
    batch = list(filter(lambda x : x is not None, batch))
    if len(batch) == 0:
        return None
    imgs = [item['scannet_frame'] for item in batch]
    for i,img in enumerate(imgs):
        if img.dim() == 3:
            imgs[i] = img.unsqueeze(0)
    
    imgs = torch.cat(imgs, 0)
    targets = [item['shapenet_tensor'] for item in batch]
    for i,target in enumerate(targets):
        if target.dim() == 3:
            targets[i] = target.unsqueeze(0)
    targets = torch.cat(targets, 0)
    
    original_img = [item['original_image'] for item in batch]
    original_img = torch.hstack(original_img).squeeze(0)
    return {'scannet_frame': imgs, 'shapenet_target': targets, 'original_image': original_img}

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
    
    return {'scannet_frame': imgs, 
            'original_image': original_img,
            'gt_objects': gt_objects,
            'categories': categories,
            'pointclouds': pointclouds,
            'voxels': voxels,
            'frame_id': frame_id}

if __name__ == '__main__':
    config = {
    "width": 224,
    "height": 224,
    "device": 'cuda:0',
    "batch_size": 32,
    'lr': 1e-4,
    "epochs": 40
    }    

    dataset = RetrievalDataset(config)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)

    scan_dataset = ValidationDatasetScannet(config, scannet_split="src/splits/scannet_train.txt")
    shape_dataset = ValidationDatasetShapenet(config, shapenet_split="src/splits/shapenet_train.txt")

    scan_dataloader = torch.utils.data.DataLoader(scan_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn_val_scannet)
    shape_dataloader = torch.utils.data.DataLoader(shape_dataset, batch_size=8, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scan_model = Encoder().to(device)
    shape_model = Encoder().to(device)
    print('Training retrieval model...')
    history = train(scan_model, shape_model, device, config, dataloader, scan_dataloader, shape_dataloader)