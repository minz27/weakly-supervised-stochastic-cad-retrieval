from src.dataset import OverfitDatasetScannet
from src.data.prepare_data import retrieve_instances
import pickle
import torch
from tqdm import tqdm

def save_mapping()->None:
    config = {
    "width": 224,
    "height": 224,
    "device": 'cuda:0',
    "n_instances": 2,
    }
    scannet_mapping_dest = 'src/cached/scannet_mapping.pkl'
    scannet = OverfitDatasetScannet(config, split="src/splits/scannet_train.txt")
    scannetloader = torch.utils.data.DataLoader(scannet, batch_size=32, shuffle=False)
    scannet_mapping = []

    for i, scan in enumerate(tqdm(scannetloader)):
        masked_normals,labels, indices = retrieve_instances(scan["normal"], scan["mask"], [], config['n_instances'], scan['frame'], scan["R"])
        masked_normals = masked_normals.view(-1, 3, config["height"], config["width"]).float()
        if masked_normals.shape[0] == 0:
            continue   
        for j in range(len(labels)):
            scannet_mapping.append(labels[j] + '_' + str(indices[j]))
    
    with open(scannet_mapping_dest, 'wb') as file:
        pickle.dump(scannet_mapping, file)        

if __name__=='__main__':
    save_mapping()

