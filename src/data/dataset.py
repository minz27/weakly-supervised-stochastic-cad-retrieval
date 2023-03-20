import torch 
from pathlib import Path
import pickle
from src.data.prepare_data import retrieve_instances
import numpy as np
import torchvision.transforms as transforms
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.ops import sample_points_from_meshes
from src.rendering.renderer import render_view
from src.external import binvox_rw
from src.data.prepare_data import mesh_to_voxel, custom_load_obj
from PIL import Image
import json
import random

class RetrievalDataset(torch.utils.data.Dataset):
    def __init__(self,
                config,
                scannet_root=Path("/mnt/raid/mdeka/scannet_frames_25k"),
                shapenet_root=Path("/mnt/raid/mdeka/shapenet/ShapeNetCore.v2"),
                rendering_root = Path("/mnt/raid/mdeka/Rendering/"),
                cache_root = Path("src/cached"),
                scannet_split="src/splits/scannet_train.txt",
                shapenet_split="src/splits/shapenet_train.txt",
                threshold:float=0.0307):
        
        self.config = config
        self.scannet_root = scannet_root
        self.shapenet_root = shapenet_root
        self.rendering_root = rendering_root
        self.threshold = threshold
        self.DEFAULT_WIDTH, self.DEFAULT_HEIGHT = 480, 360

        scannet_mapping_path = cache_root /  "scannet_mapping.pkl" 
        with open(str(scannet_mapping_path), 'rb') as file:
            self.scannet_mapping = pickle.load(file)

        self.cache_distances_path = cache_root / "cached_distances_copy.txt"

        with open(scannet_split) as f: 
            self.items = f.readlines()

        # self.items = [x.strip() for x in self.items][:3668]
        self.items = [x.strip() for x in self.items][:10]

        with open(shapenet_split) as f: 
            self.shapenet_items = f.readlines()

        self.shapenet_items = [x.strip() for x in self.shapenet_items]    

        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4647, 0.4190, 0.3626],std=[0.2954, 0.2809, 0.2681])
        ])

        self.shapenet_normalize = transforms.Normalize(mean=[0.1149, 0.1149, 0.1149], std=[0.2395, 0.2395, 0.2395])


    def __getitem__(self, index):
        items = self.items[index]
        K=1
        image = items + '.jpg' 
        mask = items +  '.png' 

        img_path = self.scannet_root / image.split('/')[0] / 'color' /image.split('/')[1]
        img = Image.open(str(img_path)).convert("RGB").resize(size=(self.DEFAULT_WIDTH, self.DEFAULT_HEIGHT), resample=Image.BILINEAR)
        img_tensor = self.to_tensor(img)
        
        mask_path = self.rendering_root / mask.split('/')[0] /'instance' / mask.split('/')[1]
        mask = Image.open(mask_path).resize(size=(self.DEFAULT_WIDTH, self.DEFAULT_HEIGHT), resample=Image.BILINEAR)
        mask_tensor = torch.tensor(np.array(mask)).unsqueeze(0)

        masked_imgs, labels, indices, categories = retrieve_instances(img_tensor.unsqueeze(0), mask_tensor.unsqueeze(0), None, None, [self.items[index]], [0], rotate=False ) #2xNxCxWxH
        masked_imgs = masked_imgs.view(-1, 3, self.config["width"], self.config["height"]).to(torch.device(self.config["device"]))

        if masked_imgs.shape[0] == 0:
            return None
        
        init_frame_index = self.scannet_mapping.index(self.items[index] + '_' + str(indices[0]))    
        frame_indices = list(range(init_frame_index, init_frame_index + masked_imgs.shape[0])) 
        # print(frame_indices)
        # distances = self.get_distance_matrix([index])[0]
        distances = self.get_distance_matrix(frame_indices)
        scannet_frames = []
        shapenet_views = []
        canonical_azimuth = [0, 60, 120, 180, 240, 300]
        
        for i in range(len(masked_imgs)):
            if distances.size == 0:
                return None
            discovered_nodes = np.argwhere(distances[i] <= self.threshold)
            if len(discovered_nodes) > 0:
                discovered_nodes = np.argsort(distances[i])
                choices = []
                for node in discovered_nodes:
                    if self.shapenet_items[(node // 6)].split('/')[0] == categories[i]:
                        choices.append(node)
                if len(choices) == 0:
                    continue
                node = random.choice(choices)        
                # discovered_nodes = np.random.choice(discovered_nodes[0:2].ravel(), 1)
                # obj_id = self.shapenet_items[(discovered_nodes[0] // 6)]
                # view_azimuth = canonical_azimuth[(discovered_nodes[0] % 6)]
                obj_id = self.shapenet_items[(node // 6)]
                view_azimuth = canonical_azimuth[(node % 6)]
                # print("Node:%s Object id:%s View: %s"%(discovered_nodes[0], obj_id, view_azimuth))
                obj_path = self.shapenet_root / obj_id / "models/model_normalized.obj"
                mesh =  load_objs_as_meshes([str(obj_path)], device=torch.device(self.config["device"]))
                shapenet_views.append(render_view(mesh,  image_size = self.config["width"], device = torch.device(self.config["device"]), azim=view_azimuth, dist=1.))
                scannet_frames.append(masked_imgs[i])      
        if len(shapenet_views) == 0:
            return None
        
        shapenet_tensor = torch.stack(shapenet_views).permute(1,0,4,2,3) 
        scannet_tensor = torch.stack(scannet_frames)

        return {
            'scannet_frame': scannet_tensor.squeeze(0),
            'shapenet_tensor': shapenet_tensor.squeeze(0).squeeze(0),
            'original_image': img_tensor.expand(len(shapenet_views), 3, self.DEFAULT_HEIGHT, self.DEFAULT_WIDTH).unsqueeze(0)
        }


    def __len__(self):
        return len(self.items)        

    def get_distance_matrix(self, indices:list)->np.array:
        with open(str(self.cache_distances_path), 'r') as f:
            lines = []
            for i, line in enumerate(f):
                if i in indices:
                    line = line.replace('[', '')
                    line = line.replace(']', '')
                    lines.append(line.strip())  

        lines = [x.split(", ") for x in lines]
        return np.array(lines, dtype=np.float32)              

class ValidationDatasetScannet(torch.utils.data.Dataset):
    def __init__(self,
                config,
                scannet_root=Path("/mnt/raid/mdeka/scannet_frames_25k"),
                shapenet_root=Path("/mnt/raid/mdeka/shapenet/ShapeNetCore.v2"),
                rendering_root = Path("/mnt/raid/mdeka/Rendering/"),
                scan2cad_path=Path("/mnt/raid/mdeka/scan2cad/scan2cad_image_alignments.json"),
                scannet_split="src/splits/scannet_val.txt",
                use_train:bool=False):

        self.config = config
        self.scannet_root = scannet_root
        self.shapenet_root = shapenet_root
        self.rendering_root = rendering_root
        self.DEFAULT_WIDTH, self.DEFAULT_HEIGHT = 480, 360            

        with open(scannet_split) as f: 
            self.items = f.readlines()

        self.items = [x.strip() for x in self.items]
        if use_train:
            self.items = self.items[:10] 
        
        with open(scan2cad_path) as f:
            self.scan2cad_img = json.load(f) 

        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4647, 0.4190, 0.3626],std=[0.2954, 0.2809, 0.2681])
        ])    

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):        
        items = self.items[index]
        K=5
        image = items + '.jpg' 
        mask = items +  '.png' 

        img_path = self.scannet_root / image.split('/')[0] / 'color' /image.split('/')[1]
        img = Image.open(str(img_path)).convert("RGB").resize(size=(self.DEFAULT_WIDTH, self.DEFAULT_HEIGHT), resample=Image.BILINEAR)
        img_tensor = self.to_tensor(img)
        
        mask_path = self.rendering_root / mask.split('/')[0] /'instance' / mask.split('/')[1]
        mask = Image.open(mask_path).resize(size=(self.DEFAULT_WIDTH, self.DEFAULT_HEIGHT), resample=Image.BILINEAR)
        mask_tensor = torch.tensor(np.array(mask)).unsqueeze(0)

        masked_imgs, labels, indices, categories = retrieve_instances(img_tensor.unsqueeze(0), mask_tensor.unsqueeze(0), None, None, [self.items[index]], [0], rotate=False, retrieve=True ) #2xNxCxWxH
        masked_imgs = masked_imgs.view(-1, 3, self.config["width"], self.config["height"]).to(torch.device(self.config["device"]))

        if masked_imgs.shape[0] == 0:
            return None

        gt_objects = []
        pointclouds = []
        random_pointclouds = []
        categories = []
        voxels = []
        random_voxels = []

        for i in range(len(labels)):
            obj = self.scan2cad_img['alignments'][labels[i]][indices[i] - 1]
            gt_objects.append(obj['catid_cad'] + '/' + obj['id_cad'])    
            categories.append(obj['catid_cad'])

            mesh_path = self.shapenet_root / gt_objects[i] / "models/model_normalized.obj"
            mesh_pytorch3d = load_objs_as_meshes([str(mesh_path)], device=torch.device(self.config["device"]))

            pointclouds.append(sample_points_from_meshes(mesh_pytorch3d, 5000))  

            # voxel_path = self.shapenet_root / gt_objects[i] / "models/model_normalized.solid.binvox"
            # with open(voxel_path, 'rb') as f:
            #     voxel = binvox_rw.read_as_3d_array(f)
            vertices, faces = custom_load_obj(mesh_path)
            vertices = vertices - (vertices.max(0) + vertices.min(0)) / 2
            size = (36/32) * max(np.ptp(vertices, axis=0) / 2)
            voxel = torch.from_numpy(mesh_to_voxel(vertices, faces, 32, size)).unsqueeze(0)

            with open(mesh_path.with_suffix(".json")) as f:
                model = json.load(f) 
            shape_size = max(np.array(model["max"]) - np.array(model["min"])) / 2
            # voxel = torch.from_numpy(voxel.data).unsqueeze(0)
            # voxels.append(voxel)
            voxels.append(voxel * np.clip(shape_size, 0.01, 10))

            #Get a random voxel from same category
            cat_path = self.shapenet_root / obj['catid_cad']
            random_model = random.choice(list(cat_path.iterdir()))
            mesh_path = random_model /  "models/model_normalized.obj" 

            vertices, faces = custom_load_obj(mesh_path)
            vertices = vertices - (vertices.max(0) + vertices.min(0)) / 2
            size = (36/32) * max(np.ptp(vertices, axis=0) / 2)
            voxel = torch.from_numpy(mesh_to_voxel(vertices, faces, 32, size)).unsqueeze(0)

            with open(mesh_path.with_suffix(".json")) as f:
                model = json.load(f) 
            shape_size = max(np.array(model["max"]) - np.array(model["min"])) / 2
            random_voxels.append(voxel * np.clip(shape_size, 0.01, 10))

            mesh_pytorch3d = load_objs_as_meshes([str(mesh_path)], device=torch.device(self.config["device"]))
            random_pointclouds.append(sample_points_from_meshes(mesh_pytorch3d, 5000))

        pointclouds = torch.stack(pointclouds)
        random_pointclouds = torch.stack(random_pointclouds)
        voxels = torch.stack(voxels)
        random_voxels = torch.stack(random_voxels)
        frame_id = [self.items[index]]*len(labels)
        
        return {
            'scannet_frame': masked_imgs,
            'original_image': img_tensor.expand(len(masked_imgs), 3, self.DEFAULT_HEIGHT, self.DEFAULT_WIDTH).unsqueeze(0),
            'gt_objects': gt_objects,
            'categories': categories,
            'pointclouds': pointclouds.squeeze(1),
            'voxels': voxels,
            'frame_id': frame_id,
            'labels': indices,
            'random_voxels': random_voxels,
            'random_pointclouds': random_pointclouds
        }          

class ValidationDatasetShapenet(torch.utils.data.Dataset):
    def __init__(self,
                config,
                shapenet_root=Path("/mnt/raid/mdeka/shapenet/ShapeNetCore.v2"),
                shapenet_split="src/splits/shapenet_val.txt"):

        self.config = config
        self.shapenet_root = shapenet_root
                    
        with open(shapenet_split) as f: 
            self.items = f.readlines()

        self.items = [x.strip() for x in self.items]
 
        self.to_tensor = transforms.Compose([
            transforms.ToTensor()
        ])    

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        #Render, pointcloud, shapeid
        canonical_azimuth = [0, 60, 120, 180, 240, 300]
        dist = 1.        
        mesh_path = self.shapenet_root / self.items[index]/ "models/model_normalized.obj"
        mesh_pytorch3d = load_objs_as_meshes([str(mesh_path)], device=torch.device(self.config["device"]))
        pointcloud = sample_points_from_meshes(mesh_pytorch3d, 5000)

        # voxel_path = self.shapenet_root / self.items[index] / "models/model_normalized.solid.binvox"
        # with open(voxel_path, 'rb') as f:
        #     voxel = binvox_rw.read_as_3d_array(f)
        # voxel = torch.from_numpy(voxel.data).unsqueeze(0)
        vertices, faces = custom_load_obj(mesh_path)
        vertices = vertices - (vertices.max(0) + vertices.min(0)) / 2
        size = (36/32) * max(np.ptp(vertices, axis=0) / 2)
        voxel = torch.from_numpy(mesh_to_voxel(vertices, faces, 32, size)).unsqueeze(0)
        with open(mesh_path.with_suffix(".json")) as f:
            model = json.load(f) 
        shape_size = max(np.array(model["max"]) - np.array(model["min"])) / 2
        voxel = voxel * np.clip(shape_size, 0.01, 10)
        renders = []
        for azim in canonical_azimuth:
            renders.append(render_view(mesh_pytorch3d,  image_size = self.config["width"], device = torch.device(self.config["device"]), azim=azim, dist=dist))
        render_tensor = torch.stack(renders).permute(1,0,4,2,3)
        return {
            "rendered_views": render_tensor,
            "pointclouds": pointcloud.expand(len(canonical_azimuth), 5000, 3),
            "voxels": voxel.expand(len(canonical_azimuth), 1, 32, 32, 32),
            "shape_id": self.items[index]
        }    