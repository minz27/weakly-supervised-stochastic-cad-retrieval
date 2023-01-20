import numpy as np 
import math
import torch

def self_similarity_normal_histogram(nmap, mask,box=None):
    # TODO: Adapt for Pytorch tensor instead of converting to numpy  
    # nmap: a normal map of shape [h, w, 3]
    # mask: a binary mask of shape [h, w]
    # box: a box with shape [4,]
    # author: Weicheng Kuo (weicheng@google.com)

    num_bins = 5  # Bins for cosine value between [-1, 1].
    bin_vect = np.linspace(0, math.pi, num=num_bins + 1)
    if box is not None:
        ymin, xmin, ymax, xmax = box
        nmap = nmap[ymin:ymax, xmin:xmax]
        mask = mask[ymin:ymax, xmin:xmax]
    
    
    valid_map = nmap[mask > 0]
    valid_map /= np.linalg.norm(valid_map, axis=-1)[:, None]
    assert np.allclose(np.linalg.norm(valid_map, axis=-1), 1, rtol=1e-3)
    pairwise_product = np.matmul(valid_map, valid_map.transpose())
    pairwise_product = np.clip(pairwise_product, -1, 1)
    pairwise_angle = np.arccos(pairwise_product)
    hist, _ = np.histogram(pairwise_angle, bin_vect)
    hist = hist.astype(np.float32) / np.sqrt(pairwise_product.size) # Divide by the square root.

    hist += 1
    # Normalise to sum to 1
    hist = hist / np.sum(hist)
    #Clip to ensure similarity between scannet and shapenet hist
    # hist = np.clip(hist, 0, 0.2)
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

    return np.sum(intersection) / np.sum(union)
    # return np.sum(intersection)

def calculate_histogram_similarity_matrix(histograms, eps = 1e-5):
    '''
    Args:
        histograms: self-similarity histograms, numpy.ndarray of shape (N, 18)
    Returns:
        IoU Similarity Matrix: numpy.ndarray of shape (N,N)    
    '''
    rows = histograms[:, None, :]
    columns = histograms[None, :, :]

    intersection = np.minimum(rows, columns)
    union = np.maximum(rows, columns)

    similarity_matrix = np.sum(intersection, axis = 2) / np.sum(union + 1e-5, axis = 2)
    # return np.sum(intersection, axis=2)
    return similarity_matrix

def calculate_perceptual_similarity_matrix(image_tensor, normal_tensor, vggloss, beta:float, gamma:float):
    '''
    Brute force now, to be vectorised later    
    '''
    similarity_matrix = torch.zeros([normal_tensor.shape[0], normal_tensor.shape[0]])

    for i in range(normal_tensor.shape[0]):
        for j in range(normal_tensor.shape[0]):
            content_loss, style_loss = vggloss(normal_tensor[i].unsqueeze(0), normal_tensor[j].unsqueeze(0))
            similarity_matrix[i, j] = beta * (style_loss/10000) + gamma * content_loss

    return similarity_matrix.detach().cpu().numpy()

def scale_tensor(tensor, a = -1, b = 1):
    '''
    Args:
        Pytorch tensor
    Returns:
        Pytorch tensor with the values scaled to be in [a,b]    
    '''
    return (a + ((tensor - tensor.min())*(b - a)) / (tensor.max() - tensor.min()))    

# def generate_labels(similarity_matrix):
#     '''
#     Args:
#         similarity_matrix: numpy.ndarray of shape (N, N)
#     Returns:
#         labels: torch.tensor of shape (N)    
#     '''
#     label = 0
#     labels = [-1 for x in similarity_matrix[0]]

#     explored_nodes = []
#     for i in range(similarity_matrix.shape[0]):
#         #print(np.argwhere(similarity_matrix[i] > 0.6))
#         if i not in explored_nodes:
#             explored_nodes.append(i)
#             labels[i] = label
#             threshold = np.quantile(similarity_matrix[i], 0.3)
#             discovered_nodes = np.argwhere(similarity_matrix[i] <= threshold)

#             for node in discovered_nodes:
#                 if node[0] not in explored_nodes:
#                     labels[node[0]] = label
#                     explored_nodes.append(node[0])
#             label += 1
#         else:
#             threshold = np.quantile(similarity_matrix[i], 0.3)
#             discovered_nodes = np.argwhere(similarity_matrix[i] <= threshold)  
#             for node in discovered_nodes:
#                 if node[0] not in explored_nodes:
#                     labels[node[0]] = labels[i]
#                     explored_nodes.append(node[0])    
#     return torch.tensor(labels)    


#Force each frame to have a different label
# def generate_labels(similarity_matrix, n_frames:int):
#     '''
#     Args:
#         similarity_matrix: numpy.ndarray of shape (N, N)
#     Returns:
#         labels: torch.tensor of shape (N)    
#     '''
#     label = 0
#     labels = [-1 for x in similarity_matrix[0]]

#     explored_nodes = []
#     #TODO: Optimise and get rid of this loop
#     for i in range(n_frames):
#         labels[i] = i
#         explored_nodes.append(i)
#         threshold = np.quantile(similarity_matrix[i, n_frames:], 0.1)
#         discovered_nodes = np.argwhere(similarity_matrix[i, n_frames:] <= threshold)
#         for node in discovered_nodes:
#             if node[0] not in explored_nodes:
#                 labels[node[0]] = i
#                 explored_nodes.append(node[0])
    
#     label = n_frames
    
#     for i in range(n_frames,similarity_matrix.shape[0]):
#         #print(np.argwhere(similarity_matrix[i] > 0.6))
#         if i not in explored_nodes:
#             explored_nodes.append(i)
#             labels[i] = label
#             threshold = np.quantile(similarity_matrix[i, n_frames:], 0.1)
#             discovered_nodes = np.argwhere(similarity_matrix[i, n_frames:] <= threshold)

#             for node in discovered_nodes:
#                 if node[0] not in explored_nodes:
#                     labels[node[0]] = label
#                     explored_nodes.append(node[0])
#             label += 1
#         else:
#             threshold = np.quantile(similarity_matrix[i, n_frames:], 0.1)
#             discovered_nodes = np.argwhere(similarity_matrix[i, n_frames:] <= threshold)  
#             for node in discovered_nodes:
#                 if node[0] not in explored_nodes:
#                     labels[node[0]] = labels[i]
#                     explored_nodes.append(node[0])    
#     return torch.tensor(labels)  

def generate_labels(similarity_matrix, n_frames:int, K:int=3):
    '''
    Args:
        similarity_matrix: numpy.ndarray of shape (N, N)
    Returns:
        labels: torch.tensor of shape (N)    
    '''
    label = 0
    labels = [-1 for x in similarity_matrix[0]]

    explored_nodes = []
    #TODO: Optimise and get rid of this loop
    for i in range(n_frames):
        labels[i] = i
        explored_nodes.append(i)
        discovered_nodes = np.argsort(similarity_matrix[i])
        for j in range(K+1):
            if discovered_nodes[j].item() not in explored_nodes:
                labels[discovered_nodes[j].item()] = i
                explored_nodes.append(discovered_nodes[j].item())
            else:
                previous_label = labels[discovered_nodes[j].item()]
                previous_distance = similarity_matrix[previous_label][discovered_nodes[j].item()]
                if similarity_matrix[i, discovered_nodes[j].item()] < previous_distance:
                    labels[discovered_nodes[j].item()] = i
    
    label = n_frames
    
    for i in range(n_frames,similarity_matrix.shape[0]):
        if i not in explored_nodes:
            explored_nodes.append(i)
            labels[i] = label
            discovered_nodes = np.argsort(similarity_matrix[i])
            for j in range(K+1):
                if discovered_nodes[j].item() not in explored_nodes:
                    labels[discovered_nodes[j].item()] = label
                    explored_nodes.append(discovered_nodes[j].item())
            # label += 1
        # else: 
        #     discovered_nodes = np.argsort(similarity_matrix[i])
        #     for j in range(K):
        #         if discovered_nodes[j].item() not in explored_nodes:
        #             labels[discovered_nodes[j].item()] = labels[i]
        #             explored_nodes.append(discovered_nodes[j].item())    
    return torch.tensor(labels)            

def angular_loss(shape1, shape2):
    error = torch.cosine_similarity(shape1, shape2, dim=2, eps=1e-6)
    # optimize_loss = torch.sum(torch.acos(error))
    loss = torch.mean(torch.acos(error))
    return loss    