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
    return color_img*(instance == label)[:, :, None] 

