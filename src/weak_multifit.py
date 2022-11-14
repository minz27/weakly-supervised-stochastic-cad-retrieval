import torch
import numpy as np

from src.dataset import OverfitDatasetScannet, OverfitDatasetShapenet
from src.networks.basic_net import Encoder 
from src.util.losses import SupervisedContrastiveLoss
from src.util.normal_similarity import self_similarity_normal_histogram, calculate_histogram_iou
from src.data.prepare_data import retrieve_instances, transform_normal_map

#TODO: import from config file
valid_labels = [
    4,    5,    6,    7,    10,    12,    14,    15,    35,    37,    39,    40
    ]

def train(scan_model, shape_model, device, config, scannetloader, shapenetloader):
    
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
    #Add criterion
    criterion = SupervisedContrastiveLoss(temperature=0.07)
    history = []
    best_error = 1e5

    scan_model.to(device)
    shape_model.to(device)

    #To speed up overfit, i will cache the histogram calculation for the one image we have
    # scan = next(iter(scannetloader))
    # masked_img = retrieve_instances(scan["img"], scan["mask"], 6001).to(device)
    # gt_normal_img = scan["normal"]
    # gt_normal_instance = retrieve_instances(scan["normal"], scan["mask"], 6001).squeeze(0).permute(1,2,0).detach().cpu().numpy()
    # gt_normal_mask = ((gt_normal_instance[:,:,2] != 0) | (gt_normal_instance[:,:,1] != 0) | (gt_normal_instance[:,:,0] != 0))
    # gt_normal_hist = self_similarity_normal_histogram(gt_normal_instance, gt_normal_mask)
    #End caching

    #Add training loop
    for epoch in range(config['epochs']):

        scan_model.train()
        shape_model.train()

        scannet_iterator = iter(scannetloader)
        
        train_loss = 0
        n_steps = 0
        
        for i, shape in enumerate(shapenetloader):
            try:
                scan = next(scannet_iterator)
            except StopIteration:
                scannet_iterator = iter(scannetloader)
                scan = next(scannet_iterator)

            rendered_views = shape["rendered_views"].squeeze(1).view(-1, 3, config['height'], config['height']).to(device)
            
            #Mask out instances
            masked_imgs = retrieve_instances(scan["img"], scan["mask"], valid_labels, 2).to(device) #2xNxCxWxH
            masked_imgs = masked_img.view(-1, 3, config["width"], config["height"]) #(2N)xCxWxH
            masked_normals = retrieve_instances(scan["normal"], scan["mask"], valid_labels, 2).view(-1, 3, config["width"], config["height"]) 
            
            #Compute embeddings
            scan_embedding = scan_model(masked_img)
            shape_embedding = shape_model(rendered_views)
            
            #6 should be configurable to number of rendered views
            # Currently doing this with a for loop and gt normal
            labels_images = torch.tensor([1])
            '''Cached, so commented out
            gt_normal_img = scan["normal"]
            gt_normal_instance = retrieve_instances(scan["normal"], scan["mask"], 6001).squeeze(0).permute(1,2,0).detach().cpu().numpy()
            #transform shapenet normal to world space
#             gt_normal_instance = gt_normal_instance.squeeze(0).permute(1,2,0).detach().cpu().numpy()
            gt_normal_mask = ((gt_normal_instance[:,:,2] != 0) | (gt_normal_instance[:,:,1] != 0) | (gt_normal_instance[:,:,0] != 0))
            gt_normal_hist = self_similarity_normal_histogram(gt_normal_instance, gt_normal_mask)
           '''
            #Create labels now:
            labels_shapes = []
            for i in range(shape['normal_maps'].squeeze(1).squeeze(1).shape[0]):
#                 shape_normal = shape['normal_maps'].squeeze(1).squeeze(1)[i].permute(1,2,0).detach().cpu().numpy()
                shape_normal = transform_normal_map(shape['normal_maps'].squeeze(1).squeeze(1)[i].permute(1,2,0).detach().cpu().numpy(), 
                                                    shape['R'].squeeze(1).squeeze(1)[1].detach().cpu().numpy(), 
                                                    )
                shape_normal_mask = ((shape_normal[:,:,2] != 0) | (shape_normal[:,:,1] != 0) | (shape_normal[:,:,0] != 0))
                shape_hist = self_similarity_normal_histogram(shape_normal, shape_normal_mask)
                if (calculate_histogram_iou(gt_normal_hist, shape_hist) >= 0.2):
                    labels_shapes.append(1)
                else:
                    labels_shapes.append(0)
            
#             labels_shapes = torch.repeat_interleave(torch.Tensor([int(x) for x in shape['cat_id']]),1).to(torch.int)
            labels = torch.cat([labels_images, torch.tensor(labels_shapes)])
            #Concat embeddings to calculate loss
            embeddings = torch.cat([scan_embedding, shape_embedding], dim=0)
            
            loss = criterion(embeddings, labels)
            #This is not really epoch but step, but since taking the whole batch at once will fix it later
            print("Epoch:", epoch, " Loss:", loss.item())
            loss.backward()
            optimizer.step()

            history.append(loss.item())
            train_loss += loss.item()
            n_steps += 1

        avg_loss = train_loss / n_steps
        #Should check validation loss to save model
        if avg_loss < best_error:
            print("Saving model...")
            best_error = avg_loss
            torch.save(shape_model.state_dict(), f'src/runs/shape_model_{epoch}.ckpt')
            torch.save(scan_model.state_dict(), f'src/runs/scan_model_{epoch}.ckpt')
    #Torch summary writer
    #Add gradient scaling in the actual model

    return history