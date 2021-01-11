import torch


def sum_tensors(*tensors):
    # given torch tensors tensor_1, tensor_2, ...
    # calculate tensor_1 + tensor_2 + ...
    return torch.sum(torch.stack(tensors, 0), 0)
