import torch
import numpy as np

from src.dataset import OverfitDatasetScannet, OverfitDatasetShapenet
from src.networks.basic_net import Encoder 
from src.util.losses import SupervisedContrastiveLoss
from src.data.prepare_data import retrieve_instances

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

            #Make 128x128 configurable
            rendered_views = shape["rendered_views"].squeeze(1).view(-1, 3, config['height'], config['height']).to(device)
            #Think about how to get this for multiple objects, maybe retrieve each label and stack them?
            #Also to reduce complexity, retrieve only the specific classes for which shapes exist in shapenet
            masked_img = retrieve_instances(scan["img"], scan["mask"], 6001).to(device)    
            #Compute embeddings, hardcoding instance label to retrieve here for overfitting
            scan_embedding = scan_model(masked_img)
            shape_embedding = shape_model(rendered_views)
            #Labels for Supervised learning, again for now just taking couch
            labels_images = torch.tensor([4256520])
            #6 should be configurable to number of rendered views, further should be calculated from normal similarity
            labels_shapes = torch.repeat_interleave(torch.Tensor([int(x) for x in shape['cat_id']]),1).to(torch.int)
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