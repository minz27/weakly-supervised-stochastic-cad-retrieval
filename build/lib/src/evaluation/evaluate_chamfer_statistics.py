from pathlib import Path 
import random
from pytorch3d.loss import chamfer_distance
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.ops import sample_points_from_meshes
from tqdm import tqdm
import torch
import pickle 

def main():
    category_list = ['02818832','04256520','03001627','02747177','02933112','03211117','04379243','02871439']
    results_file = f'chamfer_analysis.csv'
    with open(results_file, 'w') as file:
        file.write('category,min,max,mean\n')

    with open('/mnt/raid/mdeka/stochastic_cad_retrieval/src/runs/per_category_obj_list.pkl', 'rb') as file:
        category_wise_objects = pickle.load(file)    

    per_category_minimum_loss = {}
    per_category_maximum_loss = {}
    per_category_mean_loss = {}

    for category in category_list:
        print('Evaluating %s'%category)
        
        #Initialise min/max
        per_category_minimum_loss[category] = 1e5
        per_category_maximum_loss[category] = 0.
        running_chamfer_loss = 0.
        counts = 0
        
        #Load pointclouds for each object
        pointclouds = []
        for model_path in category_wise_objects[category]:
            mesh = load_objs_as_meshes([str(model_path / 'models/model_normalized.obj')], device = torch.device('cpu'))
            pointcloud = sample_points_from_meshes(mesh, 5000)
            pointclouds.append(pointcloud)
        
        for i in tqdm(range(len(pointclouds))):
            for j in range(len(pointclouds)):
                if i != j:
                    chamfer_dist = chamfer_distance(pointclouds[i], pointclouds[j])[0].item()
                    if chamfer_dist > per_category_maximum_loss[category]:
                        per_category_maximum_loss[category] = chamfer_dist
                    if chamfer_dist < per_category_minimum_loss[category]:
                        per_category_minimum_loss[category] = chamfer_dist
                    running_chamfer_loss += chamfer_dist     
                    counts += 1      
        
        per_category_mean_loss[category] = running_chamfer_loss / counts          
        
        print('Minimum %.4f'%per_category_minimum_loss[category])
        print('Max %.4f'%per_category_maximum_loss[category])
        print('Mean %.4f'%per_category_mean_loss[category])

        with open(results_file, 'a') as file:
            file.write('%s,%s,%s,%s\n'%(category, per_category_minimum_loss[category], per_category_maximum_loss[category], per_category_mean_loss[category]))

if __name__ == '__main__':
    main()            