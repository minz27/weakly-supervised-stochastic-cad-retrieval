import pandas as pd
from pathlib import Path 
import random
import json
import os
import numpy as np
from PIL import Image

RENDERING_ROOT = '/mnt/raid/mdeka/Rendering'

def write_splits(scannet_dir: str, target_dir:str, scan2cad_path:str):
    category_list = ['02818832','04256520','03001627','02933112','04379243','02871439', "02747177"]

    data_dir = Path(scannet_dir)
    target_dir = Path(target_dir)
    files = []
    
    for scene in data_dir.iterdir():
        for frame in (scene / 'color').iterdir():
            files.append(frame)
    with open('/mnt/raid/mdeka/stochastic_cad_retrieval/src/splits/scannet_train.txt') as f: 
        selected_frames = f.readlines()

    selected_frames = [frame.strip() for frame in selected_frames]    
    # selected_frames = random.choices(files, k = 200)
    # selected_frames = [str(x).split('/')[5] + '/' +  str(x).split('/')[7].split('.')[0] for x in selected_frames]
    final_frame_list = []

    #Get objects from scan2cad dataset for each scannet frame
    scan2cad_img = {}

    with open(scan2cad_path) as f:
        scan2cad_img = json.load(f)

    object_set = set()
    print(target_dir)
    # with open(str(target_dir / "analyse_similarity_split_shapenet.txt"), "w") as f: 
    n_chosen_frames = 0
    for frame in selected_frames:
        print(frame)
        mask_path = os.path.join(RENDERING_ROOT, frame.split('/')[0], 'instance', frame.split('/')[1] + '.png')
        mask = Image.open(mask_path).resize(size=(224, 224), resample=Image.BILINEAR)
        mask = np.array(mask)
        labels,counts = np.unique(mask, return_counts=True)
        n_objects = 0
        for label in labels:
            if label != 0:
                try:
                    obj = scan2cad_img['alignments'][frame][label - 1]
                    # for obj in objects:
                        # print(obj)
                        # if n_objects == 5:
                        #     break
                    if obj['catid_cad'] in category_list:
                        # f.write(obj['catid_cad'] + '/' + obj['id_cad'] + '\n')
                        object_set.add(obj['catid_cad'] + '/' + obj['id_cad'] + '\n')
                        n_objects += 1
                                               
                except Exception as e:
                    print(e)
                    selected_frames.remove(frame)

        if n_objects == 0:
            selected_frames.remove(frame) 
        else: 
            final_frame_list.append(frame)
            n_chosen_frames += n_objects
            print("%d %d"%(n_chosen_frames, n_objects))
           
    print(final_frame_list)
    with open(str(target_dir / "scannet_train.txt"), "w") as f:
        for obj in object_set:
            f.write(obj) 

    with open(str(target_dir / "shapenet_train.txt"), "w") as f: 
        for frame in final_frame_list:
            f.write(frame + '\n')

if __name__ == '__main__':   
    scannet_dir = '/mnt/raid/mdeka/scannet_frames_25k'
    target_dir = '/mnt/raid/mdeka/stochastic_cad_retrieval/src/splits'
    scan2cad_path = '/mnt/raid/mdeka/scan2cad/scan2cad_image_alignments.json'
    write_splits(scannet_dir, target_dir, scan2cad_path)