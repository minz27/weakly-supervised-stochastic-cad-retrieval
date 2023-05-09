import torch
import numpy as np
import torchvision
from torch.optim.lr_scheduler import ExponentialLR, ReduceLROnPlateau
from tqdm import tqdm
from torch.nn import TripletMarginLoss
# from pytorch_metric_learning.losses import TripletMarginLoss
# from pytorch_metric_learning.miners import MultiSimilarityMiner
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
from src.util.losses import SupervisedContrastiveLoss, VGGPerceptualLoss, TripletMarginLossWithNegativeMining
from src.data.prepare_data import retrieve_instances, transform_normal_map
from src.data.populate_distance_cache import get_gram_matrix, get_embedding

import wandb

vggloss = VGGPerceptualLoss()
vgg16 = torchvision.models.vgg16(pretrained=True).features[:4].eval()

def get_proxy_loss(normal_1:torch.tensor, normal_2:torch.tensor)->float:
    content_loss, style_loss = vggloss(abs(normal_1), abs(normal_2))

    output_normal_1 = vgg16(abs(normal_1))
    output_normal_2 = vgg16(abs(normal_2))
    content_loss = torch.nn.functional.l1_loss(output_normal_1[:,9,:,:], output_normal_2[:,9,:,:]).item()

    return 0.1*content_loss + 0.08*1e-4*style_loss

def batched_distance(scan_embedding:torch.tensor, shape_embedding:torch.tensor, scan_gram_matrices:torch.tensor, shape_gram_matrices:torch.tensor)->torch.tensor:
    rows = scan_embedding[:, None, :]
    cols = shape_embedding[None, :, :]
    content_loss = 0.1*abs(rows - cols).mean(dim=(2,3))
    del rows 
    del cols

    rows = scan_gram_matrices[0][:, None, :]
    cols = shape_gram_matrices[0][None, :, :]
    style_loss = abs(rows - cols).mean(dim=(2,3))
    del rows 
    del cols

    rows = scan_gram_matrices[1][:, None, :]
    cols = shape_gram_matrices[1][None, :, :]
    style_loss +=  abs(rows - cols).mean(dim=(2,3))
    del rows 
    del cols

    rows = scan_gram_matrices[2][:, None, :]
    cols = shape_gram_matrices[2][None, :, :]
    style_loss += abs(rows - cols).mean(dim=(2,3))
    del rows 
    del cols

    rows = scan_gram_matrices[3][:, None, :]
    cols = shape_gram_matrices[3][None, :, :]
    style_loss += abs(rows - cols).mean(dim=(2,3))
    del rows 
    del cols

    style_loss = 0.08 * 1e-4 * style_loss
    
    return style_loss + content_loss

def get_labels(masked_normals:torch.tensor, rendered_normals:torch.tensor, scan_embeddings:torch.tensor, shape_embeddings:torch.tensor, pos_threshold:float=0.04, neg_threshold:float=0.06)->list:
    scan_embedding = get_embedding(masked_normals.cpu())
    scan_gram_matrices = get_gram_matrix(masked_normals.cpu())

    shape_embedding = get_embedding(rendered_normals.cpu())
    shape_gram_matrices = get_gram_matrix(rendered_normals.cpu())

    proxy_loss_matrix = batched_distance(scan_embedding, shape_embedding, scan_gram_matrices, shape_gram_matrices)
    
    anchors = []
    positives = []
    negatives = []
    n_frames = masked_normals.shape[0]
    for i in range(n_frames):
        neg_found = False
        pos = shape_embeddings[i]
        neg = None
        n_triples = 0
        for j in range(n_frames):
            if (proxy_loss_matrix[i, j] >= neg_threshold) and (i != j):
                neg = shape_embeddings[j]
                neg_found = True
            if neg_found:
                anchors.append(scan_embeddings[i])
                positives.append(pos)
                negatives.append(neg)
                pos_found = False
                neg_found = False
                n_triples += 1
            # if n_triples == 3:
            #     break    

    if len(anchors) == 0:
        return None, None, None
    anchors = torch.stack(anchors)
    positives = torch.stack(positives)
    negatives = torch.stack(negatives) 
    
    return anchors, positives, negatives

def train(scan_model, shape_model, device, config, dataloader, scan_dataloader, shape_dataloader, val_dataloader=None, small:bool=False):
    wandb.init(project='cad_retrieval',reinit=False,  config = config, mode="online")
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
    criterion = TripletMarginLoss(margin=0.05)
    # miner = MultiSimilarityMiner()
    history = []
    best_error = 1e5
    best_val_loss = 1e5
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
            optimizer.zero_grad()
            masked_imgs, rendered_views = batch['scannet_frame'], batch['shapenet_target']

            #Compute embeddings
            scan_embedding = scan_model(masked_imgs)
            shape_embedding = shape_model(rendered_views)
            
            # labels = torch.cat((torch.arange(scan_embedding.shape[0]), torch.arange(scan_embedding.shape[0])))
            anchors, positives, negatives =  get_labels(batch['scannet_normals'], batch['shapenet_normals'], scan_embedding, shape_embedding)
            if anchors == None:
                continue
            # embeddings = torch.cat([scan_embedding, shape_embedding], dim=0)
            # hard_pairs = miner(embeddings, labels)
            # loss = criterion(embeddings, labels, hard_pairs)
            loss = criterion(anchors, positives, negatives)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            epoch_loss += loss
            n_steps += 1
        if n_steps == 0:
            continue
        avg_loss = train_loss / n_steps
        print("Epoch:", epoch, " Loss:", avg_loss)
        history.append(avg_loss)
        # wandb.log({'Loss': avg_loss})
        scheduler.step(epoch_loss)
        del batch

        if val_dataloader == None:
            torch.save(shape_model.state_dict(), f'src/runs/shape_model.ckpt')
            torch.save(scan_model.state_dict(), f'src/runs/scan_model.ckpt')
            continue

        print("Running validation...")
        scan_model.eval()
        shape_model.eval()
        n_steps = 0
        val_loss = 0.
        with torch.no_grad():
            for i, batch in enumerate(val_dataloader):
                if batch == None:
                    continue
                masked_imgs, rendered_views = batch['scannet_frame'], batch['shapenet_target']

                #Compute embeddings
                scan_embedding = scan_model(masked_imgs)
                shape_embedding = shape_model(rendered_views)
                
                # labels = torch.cat((torch.arange(scan_embedding.shape[0]), torch.arange(scan_embedding.shape[0])))
                anchors, positives, negatives =  get_labels(batch['scannet_normals'], batch['shapenet_normals'], scan_embedding, shape_embedding)
                if anchors == None:
                    continue
                loss = criterion(anchors, positives, negatives)
                val_loss += loss.item()
                epoch_loss += loss
                n_steps += 1
            if n_steps == 0:
                continue
        avg_val_loss = val_loss / n_steps
        print("Val:", epoch, " Loss:", avg_val_loss)
        wandb.log({'Train Loss': avg_loss, 'Val Loss': avg_val_loss})
        print("Calculating IOU")
        scan_model.eval()
        shape_model.eval()
        count = 0
        total_iou = 0
        
        try:
            # for _, scan_batch in enumerate(tqdm(scan_dataloader)):
            scan_batch = next(iter(scan_dataloader))
            results = []
            scan_embeddings = scan_model(scan_batch['scannet_frame'])
            
            for j in range(len(scan_embeddings)):
                results.append(ResultsStorer())
                
            for i, batch in enumerate(tqdm(shape_dataloader)):
                views = batch['rendered_views'].view(-1,3,config["width"], config["width"])
                shape_embeddings = shape_model(views)
                pointclouds = batch['pointclouds'].view(-1, 5000, 3)
                voxels = batch['voxels'].view(-1, 1, 32, 32, 32)
                shape_id = [item for item in batch['shape_id'] for j in range(6)]
                similarity_matrix, neighbours = get_neighbours(scan_embeddings, shape_embeddings)
                for j in range(len(scan_embeddings)):
                    idx = neighbours[0, j]
                    dist = similarity_matrix[idx, j]
                    pcl = pointclouds[idx]
                    s_id = [shape_id[idx]]
                    results[j].update_results(s_id, [dist], list(pcl), list(views[idx]), list(voxels[idx]))

            for j in range(len(scan_embeddings)):
                category = scan_batch['categories'][j]
                if category not in category_list:
                    continue
                iou_batch = iou(scan_batch['voxels'][j], results[j].get_voxel()[0])
                total_iou += iou_batch
                count += 1

            print("IOU %f"%(total_iou/count))
            
            avg_iou = total_iou / count
        except Exception as e:
            print(e)
            avg_iou = 0

        wandb.log({'Val IoU': avg_iou})
        if small and (avg_val_loss < best_val_loss):
            print("Saving model...")
            # best_iou = avg_iou
            best_val_loss = avg_val_loss
            torch.save(shape_model.state_dict(), f'src/runs/shape_model_small.ckpt')
            torch.save(scan_model.state_dict(), f'src/runs/scan_model_small.ckpt')
        
        elif (avg_val_loss < best_val_loss):
        # if (avg_iou > best_iou):    
            print("Saving model...")
            # best_iou = avg_iou
            best_val_loss = avg_val_loss
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

    scan_normals = [item['scannet_normal'] for item in batch]
    for i,scan_normal in enumerate(scan_normals):
        if scan_normal.dim() == 3:
            scan_normals[i] = scan_normal.unsqueeze(0)
    
    scan_normals = torch.cat(scan_normals, 0)

    targets = [item['shapenet_tensor'] for item in batch]
    for i,target in enumerate(targets):
        if target.dim() == 3:
            targets[i] = target.unsqueeze(0)
    targets = torch.cat(targets, 0)

    target_normals = [item['shapenet_normal'] for item in batch]
    for i,target in enumerate(target_normals):
        if target.dim() == 3:
            target_normals[i] = target.unsqueeze(0)
    target_normals = torch.cat(target_normals, 0)
    
    original_img = [item['original_image'] for item in batch]
    original_img = torch.hstack(original_img).squeeze(0)
    return {'scannet_frame': imgs, 'shapenet_target': targets,'scannet_normals': scan_normals, 'shapenet_normals': target_normals, 'original_image': original_img}

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
    "batch_size": 64,
    'lr': 1e-3,
    "epochs": 100
    }    

    dataset = RetrievalDataset(config)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)

    val_dataset = RetrievalDataset(config, split='val')
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=config['batch_size']*2, shuffle=True, collate_fn=collate_fn)

    scan_dataset = ValidationDatasetScannet(config, scannet_split="src/splits/scannet_train_val.txt")
    shape_dataset = ValidationDatasetShapenet(config, shapenet_split="src/splits/shapenet_train_val.txt")

    scan_dataloader = torch.utils.data.DataLoader(scan_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn_val_scannet)
    shape_dataloader = torch.utils.data.DataLoader(shape_dataset, batch_size=8, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    scan_model = Encoder().to(device)
    shape_model = Encoder().to(device)
    print('Training retrieval model...')
    history = train(scan_model, shape_model, device, config, dataloader, scan_dataloader, shape_dataloader, val_dataloader, small=True)