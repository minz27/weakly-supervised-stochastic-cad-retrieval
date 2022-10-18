import numpy as np 
import math

def self_similarity_normal_histogram(nmap, mask,box=None):
    # TODO: Adapt for Pytorch tensor instead of converting to numpy  
    # nmap: a normal map of shape [h, w, 3]
    # mask: a binary mask of shape [h, w]
    # box: a box with shape [4,]
    # author: Weicheng Kuo (weicheng@google.com)

    num_bins = 18  # Bins for cosine value between [-1, 1].
    bin_vect = np.linspace(0, math.pi, num=num_bins + 1)
    if box is not None:
        ymin, xmin, ymax, xmax = box
        nmap = nmap[ymin:ymax, xmin:xmax]
        mask = mask[ymin:ymax, xmin:xmax]
    
    
    valid_map = nmap[mask > 0]
    valid_map /= np.linalg.norm(valid_map, axis=-1)[:, None]
    assert np.allclose(np.linalg.norm(valid_map, axis=-1), 1, rtol=1e-3)
    pairwise_product = np.matmul(valid_map, valid_map.transpose())
    pairwise_angle = np.arccos(pairwise_product)
    hist, _ = np.histogram(pairwise_angle, bin_vect)
    hist = hist.astype(np.float32) / np.sqrt(pairwise_product.size) # Divide by the square root.
    return hist 

def calculate_histogram_iou(hist1, hist2, eps = 1e-5):
    '''
    Args:
        hist1, hist2: self-similarity histograms, numpy.ndarray of shape (18, )
    Returns:
        IoU of hist1, hist2    
    '''
    intersection = np.minimum(hist1, hist2)
    union = np.maximum(hist1, hist2)
    
    return np.sum(intersection / (union + eps))    