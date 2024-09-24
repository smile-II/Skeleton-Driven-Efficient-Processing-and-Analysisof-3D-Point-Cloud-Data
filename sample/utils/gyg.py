import os
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from .mesh_utils import knn_with_batch
from .provider import index_points, save_ply

def calculate_similarity(p1_normal, p2_normal):
    """
    :param p1: size[B,N,D]
    :param p2: size[B,M,D]
    :return similarity: size[B,N,M]
    """
    B, N, _ = p1_normal.shape
    _, M, _ = p2_normal.shape
    p1_normal = torch.nn.functional.normalize(p1_normal, dim=2)
    p2_normal = torch.nn.functional.normalize(p2_normal, dim=2)
    cos_dis = torch.matmul(p1_normal, p2_normal.transpose(2, 1))
    similarity = 1 - cos_dis
    min_similarity = similarity.min(dim =2, keepdim=True)[0]
    max_similarity = similarity.max(dim =2, keepdim=True)[0]
    normalized_similarity = (similarity - min_similarity) / (max_similarity - min_similarity)
    return similarity, normalized_similarity

def calculate_distance(p1_xyz, p2_xyz):
    """
    :param p1: size[B,N,D]
    :param p2: size[B,M,D]
    :return distance: size[B,N,M]
    :return normalized_distance: size[B,N,M]
    """
    B, N, _ = p1_xyz.shape
    _, M, _ = p2_xyz.shape
    p1_xyz = p1_xyz.unsqueeze(2)  # [B,N,1,3]
    p2_xyz = p2_xyz.unsqueeze(1)  # [B,1,M,3]
    distance = torch.norm(p1_xyz - p2_xyz, dim=3)
    min_distance = distance.min(dim =2, keepdim=True)[0]
    max_distance = distance.max(dim =2, keepdim=True)[0]
    normalized_distance = (distance - min_distance) / (max_distance - min_distance)
    return distance, normalized_distance

def knn_gyg(p1_xyz, p2_xyz, p1_normal, p2_normal, k):
    """
    :param p1: size[B,N,D]
    :param p2: size[B,M,D]
    :param k: k nearest neighbors
    :return: for each point in p1, returns the indices of the k nearest points in p2; size[B,N,k]
    """
    assert p1_xyz.size(0) == p2_xyz.size(0) and p1_xyz.size(2) == p2_xyz.size(2)
    assert p1_normal.size(0) == p2_normal.size(0) and p1_normal.size(2) == p2_normal.size(2)
    B, N, _ = p1_xyz.shape
    _, M, _ = p2_xyz.shape
    device = p1_xyz.device
    similarity, normalized_similarity = calculate_similarity(p1_normal, p2_normal)  # [B,N,M]
    distance, normalized_distance = calculate_distance(p1_xyz, p2_xyz)  # [B,N,M], [B,N,M]
    measure = normalized_similarity - normalized_distance
    radius_point_idx = torch.argmax(measure, dim=-1)
    query_ball_radius = distance[torch.arange(B).unsqueeze(1), torch.arange(N), radius_point_idx]  # [B,N]
    group_idx = torch.arange(M, dtype=torch.long).to(device).view(1, 1, M).repeat([B, N, 1])  # [B, N, M]
    radius = query_ball_radius.unsqueeze(2)  # [B,N,1]
    group_idx[distance > radius] = M
    group_idx = group_idx.sort(dim=-1)[0][:, :, :k]
    group_first = group_idx[:, :, 0].view(B, N, 1).repeat([1, 1, k])
    mask = group_idx == M
    group_idx[mask] = group_first[mask]
    return group_idx
    
def calculate_deviation(points, skel_xyz, k=20):
    """
    :param points: size[B,N,6]
    :param skel_xyz: size[B,S,3]
    :param skel_r: size[B,S,1]
    :return: size[B,N]
    """
    B, N, C = points.shape
    assert C == 6
    _, S, _ = skel_xyz.shape
    points_xyz = points[:, :, :3]  # [B,N,3]
    points_normal = points[:, :, 3:]  # [B,N,3]
    knn_idx = knn_with_batch(skel_xyz, points_xyz, k=k)  # [B,S,k]
    skel_normal = index_points(points_normal, knn_idx)  # [B,S,k,3]
    mean_normal = torch.mean(skel_normal, dim=2)  # [B,S,3]
    mean_normal = mean_normal.unsqueeze(2).repeat(1, 1, k, 1)  # [B,S,k,3]
    dot_products = torch.sum(skel_normal * mean_normal, dim=-1)  # [B,S,k]
    flipped_idx = dot_products < 0  # [B,S,k]
    dot_products[flipped_idx] = -dot_products[flipped_idx]  # [B,S,k]
    variance = torch.var(dot_products, dim=-1)  # [B,S]
    distance, _ = calculate_distance(points_xyz, skel_xyz)  # [B,N,S]
    distance_weight = F.softmax(-distance, dim=-1)  # [B,N,S]
    deviation = torch.matmul(distance_weight, variance.unsqueeze(2)).squeeze()  # [B,N]
    return deviation

def save_ply_batch_gyg(points_batch, file_path, names=None, step=0, patch=False):
    batch_size = len(points_batch)

    # if type(file_path) != list:
    #     basename = os.path.splitext(file_path)[0]
    #     ext = '.ply'

    # if patch:
    #     colors = []
    #     for i in range(points_batch.shape[1]):
    #         color_i = np.repeat([color_palette[i]], points_batch.shape[2], axis=0)
    #         colors.append(color_i)
    #     colors = np.array(colors) / 255.0

    for batch_idx in range(batch_size):
        if patch:
            if names is None:
                save_name = os.path.join(file_path[batch_idx] if type(file_path) == list else file_path,
                                         '%04d_patches.ply' % (step * batch_size + batch_idx))
            else:
                save_name = os.path.join(file_path[batch_idx] if type(file_path) == list else file_path,
                                         names[batch_idx] + '.ply')

            # save_ply_patches(points_batch[batch_idx], save_name, colors=colors)
        else:
            if names is None:
                save_name = os.path.join(file_path[batch_idx] if type(file_path) == list else file_path,
                                         '%04d.ply' % (step * batch_size + batch_idx))
            else:
                save_name = os.path.join(file_path[batch_idx] if type(file_path) == list else file_path,
                                         names[batch_idx] + '.ply')

            save_ply(points_batch[batch_idx][:, :3], save_name, points_batch[batch_idx][:, 3:])

        # if type(file_path) == list:
        #     save_ply(points_batch[batch_idx], os.path.join(file_path[batch_idx], '%04d.ply' % (step * batch_size + batch_idx)))
        # else:
        #     save_ply(points_batch[batch_idx], os.path.join(file_path, '%04d.ply' % (step * batch_size + batch_idx)))