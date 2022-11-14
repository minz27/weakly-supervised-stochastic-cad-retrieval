'''
Functions to prepare and preprocess data from ScanNet and Shapenet
'''
import torch 
import numpy as np

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

def retrieve_instances(color_img, mask, label_dict, num_instances):
    #TODO convert it to torch, operate on tensors directly
    #call retrieve instances from here
 
    instances = []
    for i in range(mask.shape[0]):
        labels = return_valid_instances(mask[i], label_dict, num_instances)
        for label in labels:
            instances.append(mask_instances(color_img[i], mask[i], label)) 

    instances = torch.stack(instances)
    return instances

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

 