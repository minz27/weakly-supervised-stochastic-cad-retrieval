import torch
import numpy as np
import torchvision
from torch.optim.lr_scheduler import ExponentialLR

from src.dataset import OverfitDatasetScannet, OverfitDatasetShapenet
from src.networks.basic_net import Encoder 
from src.util.losses import SupervisedContrastiveLoss, VGGPerceptualLoss
from src.util.normal_similarity import self_similarity_normal_histogram, calculate_histogram_iou, calculate_histogram_similarity_matrix,generate_labels, calculate_perceptual_similarity_matrix
from src.data.prepare_data import retrieve_instances, transform_normal_map

#TODO refactor code to remove this
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
    #Add scheduler
    scheduler = ExponentialLR(optimizer, gamma=0.9)

    #Add criterion
    criterion = SupervisedContrastiveLoss(temperature=0.07)
    history = []
    best_error = 1e5
    transform = torch.nn.functional.interpolate

    vggloss = VGGPerceptualLoss().to(device)
    vgg16 = torchvision.models.vgg16(pretrained=True).features[:4].eval().to(device)

    scan_model.to(device)
    shape_model.to(device)
    # scannet_iterator = iter(scannetloader)
    shapenet_iterator = iter(shapenetloader)
    #Add training loop
    for epoch in range(config['epochs']):

        scan_model.train()
        shape_model.train()

        train_loss = 0
        n_steps = 0
        
        for i, scan in enumerate(scannetloader):
            try:
                shape = next(shapenet_iterator)
            except StopIteration:
                shapenet_iterator = iter(shapenetloader)
                shape = next(shapenet_iterator)        

            rendered_views = shape["rendered_views"].squeeze(1).view(-1, 3, config['width'], config['width']).to(device)
            shape_normals = shape['normal_maps'].view(-1, 3, config["width"], config["width"]).to(device)
            
            #Mask out instances
            masked_imgs, labels, indices = retrieve_instances(scan["img"], scan["mask"], valid_labels, config["n_instances"], scan['frame'], scan["R"], rotate=False ) #2xNxCxWxH
            masked_imgs = masked_imgs.view(-1, 3, config["height"], config["width"]).to(device) #(2N)xCxWxH
            masked_normals = retrieve_instances(scan["normal"], scan["mask"], valid_labels, config['n_instances'], scan['frame'], scan["R"])[0].view(-1, 3, config["height"], config["width"]).float().to(device)
            
            #Convert scan and shape to same dimensions by interpolation
            # rendered_views = torch.nn.functional.interpolate(rendered_views, mode='bilinear', size=(224, 224), align_corners=False)
            #masked_imgs = torch.nn.functional.interpolate(masked_imgs, mode='bilinear', size=(224, 224), align_corners=False)

            #Compute embeddings
            scan_embedding = scan_model(masked_imgs)
            shape_embedding = shape_model(rendered_views)
            # print(masked_normals.shape[0])
            # print(shape_normals.shape[0])
            # print(shape['normal_maps'].view(-1, 3, config["width"], config["height"]).shape[0])
            
            # histograms = []

            # for i in range(masked_normals.shape[0]):
            #     instance_normal = (masked_normals[i].permute(1,2,0).detach().cpu().numpy())
            #     instance_normal_mask = ((instance_normal[:,:,2] != 0) | (instance_normal[:,:,1] != 0) | (instance_normal[:,:,0] != 0))
            #     histograms.append(self_similarity_normal_histogram(instance_normal, instance_normal_mask))

            # for i in range(shape_normals.shape[0]):
            #     # shape_normal = transform_normal_map(shape['normal_maps'].squeeze(1).squeeze(1)[i].permute(1,2,0).detach().cpu().numpy(), 
            #                                         # shape['R'].squeeze(1).squeeze(1)[1].detach().cpu().numpy(), 
            #                                         # )
            #     # shape_normal = shape['normal_maps'].squeeze(1).squeeze(1)[i].permute(1,2,0).detach().cpu().numpy()
            #     shape_normal = shape['normal_maps'].view(-1, 3, config["width"], config["height"])[i].permute(1,2,0).detach().cpu().numpy()
            #     shape_normal_mask = ((shape_normal[:,:,2] != 0) | (shape_normal[:,:,1] != 0) | (shape_normal[:,:,0] != 0))
            #     shape_hist = self_similarity_normal_histogram(shape_normal, shape_normal_mask)
            #     histograms.append(shape_hist)


            # histograms = np.vstack(histograms)
            # similarity_matrix_histogram = 1 - calculate_histogram_similarity_matrix(histograms)

            
            view_tensor = torch.vstack([masked_imgs, rendered_views])

            # shape_normals = torch.nn.functional.interpolate(shape_normals, mode='bilinear', size=(224, 224), align_corners=False)
            # masked_normals = torch.nn.functional.interpolate(masked_normals, mode='bilinear', size=(224, 224), align_corners=False)
            normal_tensor = torch.vstack([abs(masked_normals), abs(shape_normals)])

            similarity_matrix_perceptual = calculate_perceptual_similarity_matrix(view_tensor, normal_tensor, vggloss, vgg16, beta = config['beta'], gamma=config['gamma'])

            similarity_matrix = similarity_matrix_perceptual
            # similarity_matrix = similarity_matrix_perceptual + config['alpha'] * similarity_matrix_histogram
            # similarity_matrix = calculate_histogram_similarity_matrix(histograms)
            #Generate labels based on similarity scores
            labels = generate_labels(similarity_matrix, n_frames = masked_normals.shape[0])
            print(labels)
            embeddings = torch.cat([scan_embedding, shape_embedding], dim=0)
            # embeddings = torch.cat([shape_embedding, scan_embedding], dim=0)
            embeddings, labels = embeddings[np.argwhere(labels>=0)[0]], labels[np.argwhere(labels>=0)[0]] 
            print(embeddings.shape)
            print(labels.shape)
            if (len(labels) > 0):
                loss = criterion(embeddings, labels)
        
                print("Epoch:", epoch, " Loss:", loss.item())
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                n_steps += 1

        if n_steps > 0:
            avg_loss = train_loss / n_steps
            history.append(avg_loss)
            #Should check validation loss to save model
            if (avg_loss < best_error) and (avg_loss != 0):
                print("Saving model...")
                best_error = avg_loss
                torch.save(shape_model.state_dict(), f'src/runs/shape_model.ckpt')
                torch.save(scan_model.state_dict(), f'src/runs/scan_model.ckpt')
        
            scheduler.step()    
    #Torch summary writer
    #Add gradient scaling in the actual model

    return history