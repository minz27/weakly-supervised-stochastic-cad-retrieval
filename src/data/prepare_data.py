'''
Functions to prepare and preprocess data from ScanNet and Shapenet
'''
import torch 
import torchvision
import torchvision.transforms.functional as F 
import torchvision.transforms as transforms
import numpy as np
import json
import pickle
import quaternion
import trimesh
from src.util.normal_similarity import scale_tensor

SCAN2CAD_TRAIN_PATH = '/mnt/raid/mdeka/scan2cad/scan2cad_instances_train.json'
SCAN2CAD_VAL_PATH = '/mnt/raid/mdeka/scan2cad/scan2cad_instances_val.json'
MAPPING_PATH = '/mnt/raid/mdeka/scan2cad/train_val_mapping.pkl'
SCAN2CAD_FULL_PATH = '/mnt/raid/mdeka/scan2cad/full_annotations.json'
SCAN2CAD_IMG_PATH = '/mnt/raid/mdeka/scan2cad/scan2cad_image_alignments.json'

with open(SCAN2CAD_IMG_PATH) as f:
    scan2cad_img = json.load(f)

with open(MAPPING_PATH, 'rb') as f:
    train_mapping, val_mapping = pickle.load(f)

with open(SCAN2CAD_TRAIN_PATH) as f:
    scan2cad_train = json.load(f)   
    
with open(SCAN2CAD_VAL_PATH) as f:
    scan2cad_val = json.load(f) 

with open(SCAN2CAD_FULL_PATH) as f:
    scan2cad_full = json.load(f)

class SquarePad:
    def __call__(self, image):

        w, h = image.shape[1], image.shape[2]
        max_wh = np.max([w, h])
        hp = int((max_wh - h) / 2)
        vp = int((max_wh - w) / 2)
        padding = (hp, vp, hp, vp)
        return F.pad(image, padding, 0, "constant") 

resize_bbox_transform = transforms.Compose([
    # transforms.ToPILImage(),
    SquarePad(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def make_M_from_tqs(t: list, q: list, s: list, center=None) -> np.ndarray:
    if not isinstance(q, np.quaternion):
        q = np.quaternion(q[0], q[1], q[2], q[3])
    T = np.eye(4)
    T[0:3, 3] = t
    R = np.eye(4)
    R[0:3, 0:3] = quaternion.as_rotation_matrix(q)
    S = np.eye(4)
    S[0:3, 0:3] = np.diag(s)

    C = np.eye(4)
    if center is not None:
        C[0:3, 3] = center

    M = T.dot(R).dot(S).dot(C)
    return M

#TODO: Save mapping before
def crop_bbox(image, frame, label, R, rotate, width, height):
    
    R = torch.tensor([[-0.9265, -0.1438,  0.3478],
        [-0.3761,  0.3229, -0.8685],
        [ 0.0126, -0.9354, -0.3533]])

    bbox = []
    try:
        indices = train_mapping[frame]
        for idx in indices:
            for i in scan2cad_train['annotations']:
                if i['image_id'] == idx and i['alignment_id'] == label:
                    bbox = i['bbox']
                    bbox = [int(x) for x in bbox]
                    category = i['model']['catid_cad']
                    break

    except KeyError:
        indices = val_mapping[frame]
        for idx in indices:
            for i in scan2cad_val['annotations']:
                if i['image_id'] == idx and i['alignment_id'] == label:
                    bbox = i['bbox']
                    bbox = [int(x) for x in bbox]
                    category = i['model']['catid_cad']
                    break

    if len(bbox) == 0:
        return torch.zeros(size=(1,0)), ''
    cropped_image = image[bbox[1]:bbox[1] + bbox[3],bbox[0]:bbox[0] + bbox[2],:]
    cropped_image = (SquarePad()(cropped_image.permute(2,0,1))).permute(1,2,0)
    if rotate:
        #Change canonical space
        for scan_info in scan2cad_full:
            if scan_info['id_scan'] == frame.split('/')[0]:
                trs = scan_info['trs']
        to_scannet = (make_M_from_tqs([0,0,0], trs['rotation'], [1,1,1]))
        
        tqs = scan2cad_img['alignments'][frame][label - 1]
        alignment = make_M_from_tqs([0,0,0], tqs['q'], [1,1,1])    
        
        cropped_image = torch.nn.functional.interpolate(cropped_image.permute(2,0,1).unsqueeze(0), mode='bilinear', size=(height, width), align_corners=False)
        # print(cropped_image[0].permute().shape)
        # cropped_image = cropped_image[0].permute(1,2,0) @ np.linalg.inv(alignment[:3,:3]) @ np.linalg.inv((R).double().cpu()) @ (to_scannet[:3, :3])
        cropped_image = scale_tensor(cropped_image[0].permute(1,2,0) @ np.linalg.inv(alignment[:3,:3]) @ (to_scannet[:3, :3]) @ ((R).double().cpu()))
        return ((cropped_image.permute(2,0,1)))
    # cropped_image = transforms.ToPILImage()
    # return resize_bbox_transform(cropped_image.permute(2,0,1))
    return torch.nn.functional.interpolate(cropped_image.permute(2,0,1).unsqueeze(0), mode='bilinear', size=(height, width), align_corners=False), category


def mask_instances(color_img, instance, label):
    '''
    Applies a mask corresponding to given label for scannet_frames_25k
    Args:
        color_img: RGB Image, array/tensor of size (W, H, 3)
        instance: mask array/tensor of size (W, H)
        label: integer
    Returns:
        Masked out image, array/tensor of size (W, H, 3)     
    '''
    # return color_img*(instance == label)[:, :, None]
    return color_img*(instance == label) 

def return_valid_instances(mask, label_dict, num_instances):
    unique, counts = np.unique(mask, return_counts=True)
    count_sort_ind = np.argsort(-counts)
    sorted_unique = unique[count_sort_ind]
    sorted_counts = counts[count_sort_ind]
    
    labels = []
    for i in range(len(sorted_unique)):
        #Hardcoded
        if sorted_counts[i] < 0.01 * 240 * 240:
          break
        if sorted_unique[i] // 1000 in label_dict:
            labels.append(sorted_unique[i])
        if len(labels) == num_instances:
            break
      
    return labels

def retrieve_instances(color_img, mask, label_dict, num_instances, frame, R, rotate=True, width=224, height=224, retrieve=False):
    #TODO convert it to torch, operate on tensors directly
    #call retrieve instances from here
 
    instances = []
    frame_names = []
    indices = []
    categories = []
    for i in range(mask.shape[0]):
        # labels = return_valid_instances(mask[i], label_dict, num_instances)
        labels,counts = np.unique(mask[i], return_counts=True)

        for label, count in zip(labels,counts):
            if (not retrieve) and (label != 0) and (count > 0.2 * 224 * 224):
                # masked_instance = mask_instances(color_img[i], mask[i], label)
                masked_instance = color_img[i] 
                cropped_instance, category = crop_bbox(masked_instance.permute(1,2,0), frame[i], label, R[i], rotate, width, height)
                if cropped_instance.nelement() != 0:
                    instances.append(cropped_instance)
                    frame_names.append(frame[i])
                    indices.append(label)
                    categories.append(category)
            elif (retrieve) and (label != 0) and (count > 0.1*224*224):        
                # masked_instance = mask_instances(color_img[i], mask[i], label) 
                masked_instance = color_img[i]
                cropped_instance, category = crop_bbox(masked_instance.permute(1,2,0), frame[i], label, R[i], rotate, width, height)
                if cropped_instance.nelement() != 0:
                    instances.append(cropped_instance)
                    frame_names.append(frame[i])
                    indices.append(label)
                    categories.append(category)
    if len(instances) != 0:
        instances = torch.stack(instances)
    else:
        instances = torch.empty(0, 0, width, height)
    return instances, frame_names, indices, categories

def transform_normal_map(normal_map, R):
    '''
    Multiply rotation matrix R with normal map
    Args:
        normal_map: array of size (W, H, 3)
        R: rotation matrix of size (3,3)
        width, height: dimensions of normal map
        inv: boolean, should we use R inverse
    Returns:
        transformed_normals: array of size (W, H, 3)    
    '''
    transformed_normals = np.apply_along_axis(np.linalg.inv(R).dot, 2, normal_map)
    return transformed_normals

def mesh_to_voxel(vertices, faces, resolution, size, fill=False, crop=False):
    """Voxelize a 0-centered mesh.
    All vertices must lie in a cube: [-size, size]^3
    Parameters
    ----------
    vertices : [type]
    faces : [type]
    resolution : [type]
    size : [type]
    Returns
    -------
    [type]
        [description]
    """
    if vertices.shape[0] == 0 or faces.shape[0] == 0:
        return np.zeros((resolution, resolution, resolution))

    max_edge = size / resolution
    xyzmin = -np.array([size, size, size])
    xyzmax = np.array([size, size, size])
    x_y_z = np.array([resolution, resolution, resolution])

    vertices = trimesh.Trimesh(vertices, faces).sample(300000)

    if crop:
        mask = np.all(xyzmax >= vertices, axis=1) * np.all(xyzmin <= vertices, axis=1)
        # print(f"Filtering to {np.mean(mask):.3f} of verts.")
        vertices = vertices[mask]

    all_in_box = np.all(xyzmax >= vertices.max(0)) and np.all(xyzmin <= vertices.min(0))
    assert all_in_box, f"Not in box {xyzmax} verts: {vertices.max(0)} {vertices.min(0)}"

    segments = []
    for i in range(3):
        # note the +1 in num
        segments.append(np.linspace(xyzmin[i], xyzmax[i], num=(x_y_z[i] + 1)))

    # find where each point lies in corresponding segmented axis
    # -1 so index are 0-based; clip for edge cases
    voxel_x = np.clip(np.searchsorted(segments[0], vertices[:, 0]) - 1, 0, x_y_z[0])
    voxel_y = np.clip(np.searchsorted(segments[1], vertices[:, 1]) - 1, 0, x_y_z[1])
    voxel_z = np.clip(np.searchsorted(segments[2], vertices[:, 2]) - 1, 0, x_y_z[2])

    voxel = np.zeros((resolution, resolution, resolution))
    voxel[voxel_x, voxel_y, voxel_z] = 1

    if fill:
        # Flood fill around the object by starting from the eight corners,
        # then invert the filled mask to get the object.
        stack = set(
            (
                (0, 0, 0),
                (0, 0, resolution - 1),
                (0, resolution - 1, 0),
                (0, resolution - 1, resolution - 1),
                (resolution - 1, 0, 0),
                (resolution - 1, 0, resolution - 1),
                (resolution - 1, resolution - 1, 0),
                (resolution - 1, resolution - 1, resolution - 1),
            )
        )
        while stack:
            x, y, z = stack.pop()

            if voxel[x, y, z] == 0:
                voxel[x, y, z] = -1
                if x > 0:
                    stack.add((x - 1, y, z))
                if x < (resolution - 1):
                    stack.add((x + 1, y, z))
                if y > 0:
                    stack.add((x, y - 1, z))
                if y < (resolution - 1):
                    stack.add((x, y + 1, z))
                if z > 0:
                    stack.add((x, y, z - 1))
                if z < (resolution - 1):
                    stack.add((x, y, z + 1))
        return voxel != -1

    return voxel
 
def custom_load_obj(filename_obj):
    obj_info = trimesh.load(filename_obj, file_type='obj', process=False)
    if type(obj_info) is trimesh.Scene:
        geo_keys = list(obj_info.geometry.keys())
        total_vert = []
        total_faces = []
        for gk in geo_keys:
            cur_geo = obj_info.geometry[gk]
            # fix_normals(cur_geo)
            cur_vert = cur_geo.vertices.tolist()
            cur_face = np.array(cur_geo.faces.tolist())+len(total_vert)
            total_vert += cur_vert
            total_faces += cur_face.tolist()
        return np.array(total_vert).astype("float32"), np.array(total_faces).astype("int32")
    else:
        return np.array(obj_info.vertices).astype("float32"), np.array(obj_info.faces).astype("int32")    
