import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import pairwise_distances


# releated paper:(AAAI2020) Measuring and Relieving the Over-smoothing Problem for Graph Neural Networks from the Topological View.
# https://aaai.org/ojs/index.php/AAAI/article/view/5747

# the numpy version for mad (Be able to compute quickly)
# in_arr:[node_num * hidden_dim], the node feature matrix;
# mask_arr: [node_num * node_num], the mask matrix of the target raltion;
# target_idx = [1,2,3...n], the nodes idx for which we calculate the mad value;
def mad_value(in_arr, mask_arr=None, distance_metric='cosine', digt_num=4, target_idx=None):
    # Convert numpy arrays to torch tensors
    # in_tensor = torch.tensor(in_arr, dtype=torch.float32)

    # Calculate pairwise distances
    if distance_metric == 'cosine':
        dist_tensor = 1 - F.cosine_similarity(in_arr.unsqueeze(1), in_arr.unsqueeze(0), dim=2)
    elif distance_metric == 'dist':
        pdist = torch.nn.PairwiseDistance(p=2)
        dist_tensor = pdist(in_arr.unsqueeze(1), in_arr.unsqueeze(0))
    else:
        raise ValueError("Unsupported distance metric. Only 'cosine' is supported.")

    # Create a mask tensor if not provided
    if mask_arr is None:
        mask_tensor = torch.ones_like(dist_tensor)
    else:
        mask_tensor = torch.tensor(mask_arr, dtype=torch.float32)


    # Apply the mask
    mask_dist_tensor = dist_tensor * mask_tensor

    # Calculate the divide array
    divide_tensor = (mask_dist_tensor != 0).sum(dim=1).float() + 1e-8

    # Calculate the node distance
    node_dist_tensor = mask_dist_tensor.sum(dim=1) / divide_tensor

    # If target_idx is provided, apply it
    if target_idx is not None:
        target_tensor = torch.tensor(target_idx, dtype=torch.float32)
        node_dist_tensor = node_dist_tensor * target_tensor
        mad = node_dist_tensor.sum() / ((node_dist_tensor != 0).sum().float() + 1e-8)
    else:
        mad = torch.mean(node_dist_tensor)

    # Round the result
    mad = round(mad.item(), digt_num)

    return mad


# the tensor version for mad_gap (Be able to transfer gradients)
# intensor: [node_num * hidden_dim], the node feature matrix;
# neb_mask,rmt_mask:[node_num * node_num], the mask matrices of the neighbor and remote raltion;
# target_idx = [1,2,3...n], the nodes idx for which we calculate the mad_gap value;
def mad_gap_regularizer(intensor, neb_mask, rmt_mask, target_idx):
    node_num, feat_num = intensor.size()

    input1 = intensor.expand(node_num, node_num, feat_num)
    input2 = input1.transpose(0, 1)

    input1 = input1.contiguous().view(-1, feat_num)
    input2 = input2.contiguous().view(-1, feat_num)

    simi_tensor = F.cosine_similarity(input1, input2, dim=1, eps=1e-8).view(node_num, node_num)
    dist_tensor = 1 - simi_tensor

    neb_dist = torch.mul(dist_tensor, neb_mask)
    rmt_dist = torch.mul(dist_tensor, rmt_mask)

    divide_neb = (neb_dist != 0).sum(1).type(torch.FloatTensor).cuda() + 1e-8
    divide_rmt = (rmt_dist != 0).sum(1).type(torch.FloatTensor).cuda() + 1e-8

    neb_mean_list = neb_dist.sum(1) / divide_neb
    rmt_mean_list = rmt_dist.sum(1) / divide_rmt

    neb_mad = torch.mean(neb_mean_list[target_idx])
    rmt_mad = torch.mean(rmt_mean_list[target_idx])

    mad_gap = rmt_mad - neb_mad

    return mad_gap


if __name__ == '__main__':

    in_arr = np.random.rand(5, 3)
    mask_arr = np.random.randint(1, 2, (5, 5))
    neb_mask = torch.tensor(mask_arr, dtype=torch.float32)
    rmt_mask = torch.tensor(1 - mask_arr, dtype=torch.float32)
    # target_idx = [1, 0, 1, 0, 1]

    mad_value_result = mad_value(in_arr, mask_arr)
    print(f'MAD value: {mad_value_result}')
