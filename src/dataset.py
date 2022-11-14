from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
import json
import os
from src.rendering.renderer import render_normalmap, render_view 
from src.util.normal_similarity import scale_tensor
import trimesh
from pytorch3d.io import load_objs_as_meshes

class OverfitDatasetScannet(torch.utils.data.Dataset):
    def __init__(self, 
                config,
                scannet_root = "../scannet_frames_25k/", 
                framenet_root = "../framenet_frames/scannet-frames/",
                split = "src/splits/overfit_scannet_split.txt",
                ):
        
        self.items = []
        self.scannet_root = scannet_root
        self.framenet_root = framenet_root
        self.config = config
        
        self.transforms = transforms.Compose([
            transforms.ToTensor()
        ])

        with open(split) as f: 
            self.items = f.readlines()

        self.items = [x.strip() for x in self.items]

    def __getitem__(self, index):
        
        items = self.items[index]

        image = items + '.jpg' 
        mask = items +  '.png' 
        pose = items + '.txt'

        img_path = os.path.join(self.scannet_root, image.split('/')[0], 'color', image.split('/')[1])

        img = Image.open(img_path).convert("RGB").resize(size=(self.config["width"], self.config["height"]), resample=Image.BILINEAR)
        img_tensor = self.transforms(img)

        img_large = Image.open(img_path).convert("RGB").resize(size=(480, 320), resample=Image.BILINEAR)
        img_tensor_large = self.transforms(img_large)
        
        #Ground truth instance masks
        mask_path = os.path.join(self.scannet_root, mask.split('/')[0], 'instance', mask.split('/')[1])

        mask = Image.open(mask_path).resize(size=(self.config["width"], self.config["height"]), resample=Image.BILINEAR)
        mask_tensor = self.transforms(mask)
        
        #Ground truth camera pose
        pose_path = os.path.join(self.scannet_root, pose.split('/')[0], 'pose', pose.split('/')[1])
        with open(pose_path) as f: 
            pose = f.readlines()

        pose = [x.strip() for x in pose]
        pose_matrix = [x.split(" ") for x in pose]

        R = np.array(pose_matrix[:3], dtype=np.float32)[:, :3]
        T = np.array(pose_matrix[:3], dtype=np.float32)[:, 3:]
        
        #Ground trurh normals
        normal_path = os.path.join(self.framenet_root, items.split('/')[0], "frame-" + items.split('/')[1] + "-normal.png")
        normal = Image.open(normal_path).resize(size=(self.config["width"], self.config["height"]), resample=Image.BILINEAR)
        normal_tensor = -self.transforms(normal) + 0.5

        #transform normal to world space here

        return {
            'img' : img_tensor,
            'mask': mask_tensor,
            'img_large': img_tensor_large,
            'R': torch.tensor(R),
            'T': torch.tensor(T),
            'normal': normal_tensor
        }
    def __len__(self):
        return len(self.items)    

class OverfitDatasetShapenet(torch.utils.data.Dataset):
    def __init__(self,
                config,
                shapenet_root = "../shapenet/ShapeNetCore.v2",
                split = "src/splits/overfit_shapenet_split.txt",
                ):

        self.shapenet_root = shapenet_root                 
        self.config = config
        self.transforms = transforms.Compose([
            transforms.ToTensor()
        ])

        with open(split) as f: 
            self.items = f.readlines()

        self.items = [x.strip() for x in self.items]


    def __getitem__(self, index):
        # Return rendered n canonical views and normal maps  
        # TODO: take these values from config
        # canonical_azimuth = [0, 60, 120, 180, 240, 300]
        canonical_azimuth = [240]
        dist = 1.0

        mesh_path = os.path.join(self.shapenet_root, self.items[index], "models/model_normalized.obj")
        # Load mesh for normal map rendering
        mesh_trimesh = trimesh.load(mesh_path, force='mesh')
        vertices = torch.tensor(mesh_trimesh.vertices).unsqueeze(dim = 0)
        faces = torch.tensor(mesh_trimesh.faces).unsqueeze(dim = 0)

        # Load mesh for rendering to given view
        mesh_pytorch3d = load_objs_as_meshes([mesh_path], device=torch.device(self.config["device"]))

        #For loop to render for each view
        normal_maps = []
        renders = []
        R = []
        T = []

        for azim in canonical_azimuth:
            normal_map, r, t = render_normalmap(vertices, faces, image_size = self.config["width"] ,device = torch.device(self.config["device"]), azim=azim, dist = dist)
            normal_maps.append(normal_map)
            R.append(r)
            T.append(t)
            renders.append(render_view(mesh_pytorch3d,  image_size = self.config["width"], device = torch.device(self.config["device"]), azim=azim, dist=dist))

        # normal_tensor = torch.stack(normal_maps).permute(1,0,2,3,4)
        # render_tensor = torch.stack(renders).permute(1,0,2,3,4) 
        normal_tensor = torch.stack(normal_maps).permute(1,0,4,2,3)
        render_tensor = torch.stack(renders).permute(1,0,4,2,3) 

        return {
            "normal_maps": normal_tensor,
            "rendered_views": render_tensor,
            "cat_id": self.items[index].split(sep="/")[0],
            "model_id": self.items[index].split(sep="/")[1],
            "R": torch.stack(R),
            "T": torch.stack(T)
        }

    def __len__(self):
        return len(self.items)    



