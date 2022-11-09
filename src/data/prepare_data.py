'''
Functions to prepare and preprocess data from ScanNet and Shapenet
'''
import torch 
import numpy as np

def retrieve_instances(color_img, instance, label):
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

def return_masks():
    #TODO
    #call retrieve instances from here
    return None

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