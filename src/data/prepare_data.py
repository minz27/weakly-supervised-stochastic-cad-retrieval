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

def transform_normal_map(normal_map, R, width, height, inv = False):
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
    # compute indices:
    jj = np.tile(range(width), height)
    ii = np.repeat(range(height), width)
    length = height * width
    z = np.ones(length)
    pcd = np.dstack((jj, ii, z)).reshape((length, 3))

    if inv:
        cam_RGB = np.apply_along_axis(np.linalg.inv(R).dot, 1, pcd)
    else: 
        cam_RGB = np.apply_along_axis((R).dot, 1, pcd)
    
    inew = np.floor(cam_RGB[:, 0] - cam_RGB[:, 0].min()).astype(int)
    jnew = np.floor(cam_RGB[:, 1] - cam_RGB[:, 1].min()).astype(int)

    if inv:
        out = np.zeros((max(jnew.max() + 1, width),max(inew.max() + 1, height),3),  dtype=normal_map.dtype)
    else:    
        out = np.zeros((max(inew.max() + 1, width),max(jnew.max() + 1, height),3),  dtype=normal_map.dtype)

    inew = inew.reshape(normal_map.shape[:-1])
    jnew = jnew.reshape(normal_map.shape[:-1])

    if inv:
        out[jnew, inew, :] = normal_map
    else:    
        out[inew, jnew, :] = normal_map
    return out[:128, :128, :]    