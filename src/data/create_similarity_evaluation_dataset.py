import pandas as pd
from pathlib import Path 
import random
import json

def write_splits(scannet_dir: str, target_dir:str, scan2cad_path:str):
    data_dir = Path(scannet_dir)
    target_dir = Path(target_dir)
    files = []
    
    for scene in data_dir.iterdir():
        for frame in (scene / 'color').iterdir():
            files.append(frame)
    
    selected_frames = random.choices(files, k = 2000)
    selected_frames = [str(x).split('/')[5] + '/' +  str(x).split('/')[7].split('.')[0] for x in selected_frames]
    
    #Get objects from scan2cad dataset for each scannet frame
    scan2cad_img = {}

    with open(scan2cad_path) as f:
        scan2cad_img = json.load(f)

    object_set = set()
    with open(str(target_dir / "evaluate_shapenet_split.txt"), "w") as f: 
        for frame in selected_frames:
            try:
                objects = scan2cad_img['alignments'][frame]
                for obj in objects:
                    f.write(obj['catid_cad'] + '/' + obj['id_cad'] + '\n')
            except:
                selected_frames.remove(frame)

    with open(str(target_dir / "evaluate_scannet_split.txt"), "w") as f: 
        for frame in selected_frames:
            f.write(frame + '\n')

if __name__ == '__main__':   
    scannet_dir = '/mnt/raid/mdeka/scannet_frames_25k'
    target_dir = '/mnt/raid/mdeka/stochastic_cad_retrieval/src/splits'
    scan2cad_path = '/mnt/raid/mdeka/scan2cad/scan2cad_image_alignments.json'
    write_splits(scannet_dir, target_dir, scan2cad_path)