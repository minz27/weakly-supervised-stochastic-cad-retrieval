import torch
import numpy as np

from src.dataset import OverfitDatasetScannet, OverfitDatasetShapenet
from src.networks.basic_net import Encoder 
from src.util.losses import SupervisedContrastiveLoss
from src.data.prepare_data import retrieve_instances, transform_normal_map

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
    #Add training loop
    for epoch in range(config['epochs']):

        scan_model.train()
        shape_model.train()

        scannet_iterator = iter(scannetloader)
        
        for i, shape in enumerate(shapenetloader):
            try:
                scan = next(scannet_iterator)
            except StopIteration:
                scannet_iterator = iter(scannetloader)
                scan = next(scannet_iterator)

            rendered_views = shape["rendered_views"].squeeze(1).view(-1, 3, config['height'], config['height']).to(device)
            #Think about how to get this for multiple objects, maybe retrieve each label and stack them?
            #Also to reduce complexity, retrieve only the specific classes for which shapes exist in shapenet
            masked_img = retrieve_instances(scan["img"], scan["mask"], 6001).to(device)    
            #Compute embeddings, hardcoding instance label to retrieve here for overfitting
            scan_embedding = scan_model(masked_img)
            shape_embedding = shape_model(rendered_views)
            
            labels_images = torch.tensor([4256520])
            #6 should be configurable to number of rendered views
            # Currently doing this with a for loop and gt normal
            labels_images = torch.tensor([1])
            gt_normal_img = scan["normal"]
            gt_normal_instance = retrieve_instances(scan["img"], scan["mask"], 6001)
            #transform normal to world space
            gt_normal_instance = transform_normal_map(gt_normal_instance.squeeze(0).permute(1,2,0).detach().cpu().numpy(), 
                                                      scan["R"].squeeze(0).detach().cpu().numpy(), config['height'], config['height'])
            gt_normal_mask = ((gt_normal_instance[:,:,2] != 0) | (gt_normal_instance[:,:,1] != 0) | (gt_normal_instance[:,:,0] != 0))
            gt_normal_hist = self_similarity_normal_histogram(gt_normal_instance, gt_normal_mask)
           
            #Create labels now:
            labels_shapes = []
            for i in range(shape['normal_maps'].squeeze(1).squeeze(1).shape[0]):
                shape_normal = transform_normal_map(shape['normal_maps'].squeeze(1).squeeze(1)[i].permute(1,2,0).detach().cpu().numpy(), 
                                                    shape['R'].squeeze(1).squeeze(1)[1].detach().cpu().numpy(), 
                                                    config['height'], config['height'], inv=True)
                shape_normal_mask = ((shape_normal[:,:,2] != 0) | (shape_normal[:,:,1] != 0) | (shape_normal[:,:,0] != 0))
                shape_hist = self_similarity_normal_histogram(shape_normal, shape_normal_mask)
                if (calculate_histogram_iou(gt_normal_hist, shape_hist) >= 0.5):
                    labels_shapes.append(1)
                else:
                    labels_shapes.append(0)
            
#             labels_shapes = torch.repeat_interleave(torch.Tensor([int(x) for x in shape['cat_id']]),1).to(torch.int)
            labels = torch.cat([labels_images, labels_shapes])
            #Concat embeddings to calculate loss
            embeddings = torch.cat([scan_embedding, shape_embedding], dim=0)
            
            loss = criterion(embeddings, labels)
            #This is not really epoch but step, but since taking the whole batch at once will fix it later
            print("Epoch:", epoch, " Loss:", loss.item())
            loss.backward()
            optimizer.step()

            history.append(loss.item())

            #Should check validation loss to save model
            if loss.item() < best_error:
                print("Saving model...")
                best_error = loss.item() 
                torch.save(shape_model.state_dict(), f'src/runs/shape_model_{epoch}.ckpt')
                torch.save(scan_model.state_dict(), f'src/runs/scan_model_{epoch}.ckpt')
    #Torch summary writer
    #Add gradient scaling in the actual model

    return history