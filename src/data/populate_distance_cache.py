import torch
import torchvision
from tqdm import tqdm
import pickle
import json
from src.dataset import OverfitDatasetScannet, OverfitDatasetShapenet
from src.data.prepare_data import retrieve_instances
import gc


vgg_block_0 = torchvision.models.vgg16(pretrained=True).features[:4].eval()
vgg_block_1 = torchvision.models.vgg16(pretrained=True).features[4:9].eval()
vgg_block_2 = torchvision.models.vgg16(pretrained=True).features[9:16].eval()
vgg_block_3 = torchvision.models.vgg16(pretrained=True).features[16:23].eval()

def get_gram_matrix(normal_map:torch.tensor)->list[torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
    x0 = vgg_block_0(abs(normal_map))
    x1 = vgg_block_1(x0)
    x2 = vgg_block_2(x1)
    x3 = vgg_block_3(x2)

    act_x0 = x0.reshape(x0.shape[0], x0.shape[1], -1)
    act_x1 = x1.reshape(x1.shape[0], x1.shape[1], -1)
    act_x2 = x2.reshape(x2.shape[0], x2.shape[1], -1)
    act_x3 = x3.reshape(x3.shape[0], x3.shape[1], -1)

    gram_x0 = act_x0 @ act_x0.permute(0, 2, 1)
    gram_x1 = act_x1 @ act_x1.permute(0, 2, 1)
    gram_x2 = act_x2 @ act_x2.permute(0, 2, 1)
    gram_x3 = act_x3 @ act_x3.permute(0, 2, 1)

    return [gram_x0.detach(), gram_x1.detach(), gram_x2.detach(), gram_x3.detach()]

def get_embedding(normal_map:torch.tensor)->torch.tensor:
    embedding = vgg_block_0(abs(normal_map))[:,9,:,:]
    return embedding.detach()

def batched_distance(n_shapes:int, n_scans:int, n_shape_batch:int, n_scan_batch:int, shape_chunk_loc:str, scan_chunk_loc:str, output_txt_loc:str, batch_size=16, continue_batch:int=0)->torch.tensor:
    # distance_matrix = torch.zeros((n_scans, n_shapes)).detach()
    print('Computing distance matrix...')
    scan_pointer = 0
    for i in tqdm(range(continue_batch, n_scan_batch)):
        tqdm.write(f"Batch {i}")
        with open(scan_chunk_loc + str(i) + '.pkl', 'rb') as file:
            scan_embeddings, scan_gram_matrices = pickle.load(file)
        if i in [97, 394]:
            with open(output_txt_loc, 'a') as file:
                distance_matrix = torch.ones((scan_embeddings.shape[0], n_shapes)).detach()
                matrix = distance_matrix.tolist()
                for line in matrix:
                    file.write(f"{line}\n")
                # pickle.dump(distance_matrix, file)
                del distance_matrix
                gc.collect()
                continue
        distance_matrix = torch.zeros((scan_embeddings.shape[0], n_shapes)).detach()
        
        shape_pointer = 0    
        for j in tqdm(range(n_shape_batch)):
            with open(shape_chunk_loc + str(j) + '.pkl', 'rb') as file:
                shape_embeddings, shape_gram_matrices = pickle.load(file)    
            rows = scan_embeddings[:, None, :]
            cols = shape_embeddings[None, :, :]
            content_loss = 0.1*abs(rows - cols).mean(dim=(2,3))
            del rows 
            del cols

            rows = scan_gram_matrices[0][:, None, :]
            cols = shape_gram_matrices[0][None, :, :]
            # style_loss_0 = abs(rows - cols).mean(dim=(2,3))
            style_loss = abs(rows - cols).mean(dim=(2,3))
            del shape_gram_matrices[0]
            del rows 
            del cols

            rows = scan_gram_matrices[1][:, None, :]
            cols = shape_gram_matrices[0][None, :, :]
            # style_loss_1 = abs(rows - cols).mean(dim=(2,3))
            style_loss +=  abs(rows - cols).mean(dim=(2,3))
            del shape_gram_matrices[0]
            del rows 
            del cols

            rows = scan_gram_matrices[2][:, None, :]
            cols = shape_gram_matrices[0][None, :, :]
            # style_loss_2 = abs(rows - cols).mean(dim=(2,3))
            style_loss += abs(rows - cols).mean(dim=(2,3))
            del shape_gram_matrices[0]
            del rows 
            del cols

            rows = scan_gram_matrices[3][:, None, :]
            cols = shape_gram_matrices[0][None, :, :]
            # style_loss_3 = abs(rows - cols).mean(dim=(2,3))
            style_loss += abs(rows - cols).mean(dim=(2,3))
            del shape_gram_matrices[0]
            del rows 
            del cols

            # style_loss = 0.08 * 1e-4 * (style_loss_0 + style_loss_1 + style_loss_2 + style_loss_3)
            style_loss = 0.08 * 1e-4 * style_loss
            # distance_matrix[scan_pointer : scan_pointer + scan_embeddings.shape[0], shape_pointer: shape_pointer + shape_embeddings.shape[0]] = (content_loss + style_loss)
            distance_matrix[:, shape_pointer: shape_pointer + shape_embeddings.shape[0]] = (content_loss + style_loss)
            shape_pointer += shape_embeddings.shape[0]
            del shape_embeddings
        scan_pointer += scan_embeddings.shape[0]    
        
        del scan_embeddings
        del scan_gram_matrices

        with open(output_txt_loc, 'a') as file:
            matrix = distance_matrix.tolist()
            for line in matrix:
                file.write(f"{line}\n")
            # pickle.dump(distance_matrix, file)
        del distance_matrix
        gc.collect()
    return distance_matrix

def cache_distances(batch_size:int = 4):
    config = {
    "width": 224,
    "height": 224,
    "device": 'cuda:0',
    "n_instances": 2,
    }
    output_dest='src/cached/cached_distances.pkl'
    output_text_file = 'src/cached/cached_distances.txt'
    scannet_mapping_dest = 'src/cached/scannet_mapping.pkl'
    scannet_chunks_dest = 'src/cached/scannet_chunks'
    shapenet_chunks_dest = 'src/cached/shapenet_chunks'

    #Load scan:
    scannet = OverfitDatasetScannet(config, split="src/splits/scannet_train.txt")
    scannetloader = torch.utils.data.DataLoader(scannet, batch_size=4, shuffle=False)
    scannet_mapping = []
    scan_count = 0
    scan_batch_count = 0
    print('Getting embeddings and gram matrices for scannet...')
    for i, scan in enumerate(tqdm(scannetloader)):
        masked_normals,labels, indices = retrieve_instances(scan["normal"], scan["mask"], [], config['n_instances'], scan['frame'], scan["R"])
        masked_normals = masked_normals.view(-1, 3, config["height"], config["width"]).float().cpu()
        if masked_normals.shape[0] == 0:
            continue
        scan_embedding = get_embedding(masked_normals)
        scan_gram_matrices = get_gram_matrix(masked_normals)
        scan_count += scan_embedding.shape[0]
        scan_batch_count += 1
        with open(scannet_chunks_dest + str(i) + '.pkl', 'wb') as file:
            pickle.dump((scan_embedding, scan_gram_matrices), file)

        del scan_embedding, scan_gram_matrices    

        for j in range(len(labels)):
            scannet_mapping.append(labels[j] + '_' + str(indices[j]))
    
    #Load shape:
    shapenet = OverfitDatasetShapenet(config, split="src/splits/shapenet_train.txt")
    shapenetloader = torch.utils.data.DataLoader(shapenet, batch_size=2, shuffle=False)
    shape_count = 0
    shape_batch_count = 0
    print('Getting embeddings and gram matrices for shapenet...')
    try:
        for i, shape in enumerate(tqdm(shapenetloader)):
            shape_normals = shape['normal_maps'].view(-1, 3, config["width"], config["width"]).cpu()
            shape_embeddings = get_embedding(shape_normals)
            shape_count += shape_embeddings.shape[0]
            shape_batch_count += 1
            shape_gram_matrices = get_gram_matrix(shape_normals)
            with open(shapenet_chunks_dest + str(i) + '.pkl', 'wb') as file:
                pickle.dump((shape_embeddings, shape_gram_matrices), file)

            del shape_embeddings
            del shape_gram_matrices    
    except:
        for i, shape in enumerate(tqdm(shapenetloader)):
            shape_normals = shape['normal_maps'].view(-1, 3, config["width"], config["width"]).cpu()
            shape_embeddings = get_embedding(shape_normals)
            shape_count += shape_embeddings.shape[0]
            shape_batch_count += 1
            shape_gram_matrices = get_gram_matrix(shape_normals)
            with open(shapenet_chunks_dest + str(i) + '.pkl', 'wb') as file:
                pickle.dump((shape_embeddings, shape_gram_matrices), file) 

            del shape_embeddings
            del shape_gram_matrices    
    print('Shape Count: %d Scan Count: %d Shape Batch Count: %d Scan Batch Count: %d'%(shape_count, scan_count, shape_batch_count, scan_batch_count))
    distance_matrix = batched_distance(shape_count,scan_count, shape_batch_count, scan_batch_count, shapenet_chunks_dest, scannet_chunks_dest, output_text_file, batch_size, continue_batch=0)
    print(distance_matrix)
    with open(output_dest, 'wb') as file:
        pickle.dump(distance_matrix, file)
    with open(scannet_mapping_dest, 'wb') as file:
        pickle.dump(scannet_mapping, file)

if __name__=='__main__':
    # with open('src/cached/cached_distances.txt', 'w') as file:
    #     pass
    # cache_distances()
    
    output_dest='src/cached/cached_distances.pkl'
    scannet_chunks_dest = 'src/cached/scannet_chunks'
    shapenet_chunks_dest = 'src/cached/shapenet_chunks'
    output_text_file = 'src/cached/cached_distances.txt'
    batch_size = 2
            
    distance_matrix = batched_distance(12276, 18860, 1023, 4006, shapenet_chunks_dest, scannet_chunks_dest, output_text_file, batch_size, continue_batch=394)
    # with open(output_dest, 'wb') as file:
    #     pickle.dump(distance_matrix, file)

    