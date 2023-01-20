import torch
import numpy as np

from src.dataset import OverfitDatasetScannet, OverfitDatasetShapenet
from src.networks.basic_net import Encoder 
from src.util.losses import SupervisedContrastiveLoss, VGGPerceptualLoss
from src.util.normal_similarity import self_similarity_normal_histogram, calculate_histogram_iou, calculate_histogram_similarity_matrix,generate_labels, calculate_perceptual_similarity_matrix
from src.data.prepare_data import retrieve_instances, transform_normal_map

#TODO: import from config file
valid_labels = [
    3,  4,    5,    6,    7,    10,    12,    14,    15,    35,    37,    39,    40
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
    transform = torch.nn.functional.interpolate

    vggloss = VGGPerceptualLoss().to(device)

    scan_model.to(device)
    shape_model.to(device)

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

            rendered_views = shape["rendered_views"].squeeze(1).view(-1, 3, config['width'], config['width']).to(device)
            shape_normals = shape['normal_maps'].view(-1, 3, config["width"], config["width"])
            
            #Mask out instances
            masked_imgs = retrieve_instances(scan["img"], scan["mask"], valid_labels, config['n_instances'], scan['frame'])[0].to(device) #2xNxCxWxH
            masked_imgs = masked_imgs.view(-1, 3, config["height"], config["width"]) #(2N)xCxWxH
            masked_normals = retrieve_instances(scan["normal"], scan["mask"], valid_labels, config['n_instances'],  scan['frame'])[0].view(-1, 3, config["height"], config["width"]).to(device)
            
            #Convert scan and shape to same dimensions by interpolation
            rendered_views = torch.nn.functional.interpolate(rendered_views, mode='bilinear', size=(224, 224), align_corners=False)
            masked_imgs = torch.nn.functional.interpolate(masked_imgs, mode='bilinear', size=(224, 224), align_corners=False)

            #Compute embeddings
            scan_embedding = scan_model(masked_imgs)
            shape_embedding = shape_model(rendered_views)
            
            # histograms = []

            # for i in range(shape['normal_maps'].view(-1, 3, config["width"], config["height"]).shape[0]):
            #     # shape_normal = transform_normal_map(shape['normal_maps'].squeeze(1).squeeze(1)[i].permute(1,2,0).detach().cpu().numpy(), 
            #                                         # shape['R'].squeeze(1).squeeze(1)[1].detach().cpu().numpy(), 
            #                                         # )
            #     # shape_normal = shape['normal_maps'].squeeze(1).squeeze(1)[i].permute(1,2,0).detach().cpu().numpy()
            #     shape_normal = shape['normal_maps'].view(-1, 3, config["width"], config["height"])[i].permute(1,2,0).detach().cpu().numpy()
            #     shape_normal_mask = ((shape_normal[:,:,2] != 0) | (shape_normal[:,:,1] != 0) | (shape_normal[:,:,0] != 0))
            #     shape_hist = self_similarity_normal_histogram(shape_normal, shape_normal_mask)
            #     histograms.append(shape_hist)

            # for i in range(masked_normals.shape[0]):
            #     instance_normal = (masked_normals[i].permute(1,2,0).detach().cpu().numpy())
            #     instance_normal_mask = ((instance_normal[:,:,2] != 0) | (instance_normal[:,:,1] != 0) | (instance_normal[:,:,0] != 0))
            #     histograms.append(self_similarity_normal_histogram(instance_normal, instance_normal_mask))

            # histograms = np.vstack(histograms)
            # similarity_matrix_histogram = calculate_histogram_similarity_matrix(histograms)

            
            view_tensor = torch.vstack([masked_imgs, rendered_views])

            shape_normals = torch.nn.functional.interpolate(shape_normals, mode='bilinear', size=(224, 224), align_corners=False)
            masked_normals = torch.nn.functional.interpolate(masked_normals, mode='bilinear', size=(224, 224), align_corners=False)
            normal_tensor = torch.vstack([masked_normals, shape_normals])

            similarity_matrix_perceptual = calculate_perceptual_similarity_matrix(view_tensor, normal_tensor, vggloss, beta = config['beta'], gamma=config['gamma'])

            similarity_matrix = similarity_matrix_perceptual
            # similarity_matrix = similarity_matrix_perceptual + config['alpha'] * similarity_matrix_histogram
            # similarity_matrix = calculate_histogram_similarity_matrix(histograms)
            #Generate labels based on similarity scores
            labels = generate_labels(similarity_matrix, n_frames = masked_normals.shape[0])
            print(labels)
            # embeddings = torch.cat([scan_embedding, shape_embedding], dim=0)
            embeddings = torch.cat([shape_embedding, scan_embedding], dim=0)
            
            loss = criterion(embeddings, labels)
        
            print("Epoch:", epoch, " Loss:", loss.item())
            loss.backward()
            optimizer.step()

            # history.append(loss.item())
            train_loss += loss.item()
            n_steps += 1

        avg_loss = train_loss / n_steps
        history.append(avg_loss)
        #Should check validation loss to save model
        if avg_loss < best_error:
            print("Saving model...")
            best_error = avg_loss
            torch.save(shape_model.state_dict(), f'src/runs/shape_model.ckpt')
            torch.save(scan_model.state_dict(), f'src/runs/scan_model.ckpt')
    #Torch summary writer
    #Add gradient scaling in the actual model

    return history