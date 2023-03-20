import torch
import numpy as np

class ResultsStorer():
    def __init__(self, K:int=1):
        self.nearest_neighbours_ids = [None]*K
        self.nearest_distances = [1e5]*K
        self.nearest_pcl = [None]*K
        self.K = K
        self.rendered_views = [None]*K
        self.voxels = [None]*K
    
    def update_results(self, neighbours:list, distances:list, pcl:list, views:list, voxels:list):
        #Note, make sure that they are sorted
        n_slots =  self.K
        for i in range(self.K):
            if n_slots == 0:
                break
            for j in range(n_slots):
                if self.nearest_distances[self.K - n_slots] > distances[i]:
                    self.nearest_distances.insert(self.K - n_slots, distances[i])
                    self.nearest_neighbours_ids.insert(self.K - n_slots, neighbours[i])
                    self.nearest_pcl.insert(self.K - n_slots, pcl[i])
                    self.rendered_views.insert(self.K - n_slots, views[i])
                    self.voxels.insert(self.K - n_slots, voxels[i])

                    self.nearest_distances = self.nearest_distances[:self.K]
                    self.nearest_neighbours_ids = self.nearest_neighbours_ids[:self.K]
                    self.nearest_pcl = self.nearest_pcl[:self.K]
                    self.rendered_views = self.rendered_views[:self.K]
                    self.voxels = self.voxels[:self.K]
                    n_slots -= 1
                    break
                n_slots -= 1
    
    def get_results(self):
        return self.nearest_distances, self.nearest_neighbours_ids, self.nearest_pcl, self.rendered_views
    
    def get_pcl(self):
        return self.nearest_pcl
    
    def get_voxel(self):
        return self.voxels

def get_neighbours(scan_embeddings:torch.tensor, shape_embeddings:torch.tensor):
    rows = shape_embeddings.detach().cpu().numpy()[:, None, :]
    columns = scan_embeddings.detach().cpu().numpy()[None, :, :]

    product_matrix = rows * columns

    dot_matrix = np.sum(product_matrix, axis = 2)

    row_norm = np.linalg.norm(rows, axis = 2)
    column_norm = np.linalg.norm(columns, axis = 2)
    norm_matrix = row_norm * column_norm

    similarity_matrix = np.arccos(dot_matrix / norm_matrix)

    neighbours = np.argsort((similarity_matrix), axis = 0)
    return similarity_matrix, neighbours

def iou(pred, target, mask=1):
    """Compute the (optionally masked) IoU metric between two occupancy grids.
    Parameters
    ----------
    pred : torch.tensor, shape=(B, 1, D, D, D) or (B, 1, 1, D, D, D)
        Predicted occupancy voxel grid.
    target : torch.tensor, shape=(B, 1, D, D, D) or (1, C, 1, 1, D, D, D)
        Target occupancy voxel grid.
    mask : torch.tensor, shape=(B, 1, D, D, D)
        Mask that is 1 where the IoU should be considered, and 0 otherwise.
    Returns
    -------
    iou : torch.tensor, shape=(B,) or (B, C)
        Per-element IoU.
    """
    intersection = (torch.logical_and(pred, target) * mask).sum((-1, -2, -3, -4))
    union = (torch.logical_or(pred, target) * mask).sum((-1, -2, -3, -4))
    return intersection / (union + 1e-10)