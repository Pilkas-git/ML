import torch
from torch import nn
import torch.nn.functional as F
import math
from config.train_config import cfg

"""
    PyTorch Implementation for DR Loss
    Reference
    CVPR'20: "DR Loss: Improving Object Detection by Distributional Ranking"
    Copyright@Alibaba Group Holding Limited

https://github.com/facebookresearch/maskrcnn-benchmark/blob/57eec25b75144d9fb1a6857f32553e1574177daf/maskrcnn_benchmark/modeling/rpn/retinanet/loss.py#L99
labels = guesses, regression targets = regression targets
"""

'''def drLoss(logits, targets):
    margin = 0.5
    pos_lambda = 1
    neg_lambda = 0.1/math.log(3.5)
    L = 6.0
    tau = 4.0

    num_classes = logits.shape[1]
    dtype = targets.dtype
    device = targets.device
    class_range = torch.arange(1, num_classes + 1, dtype=dtype, device=device).unsqueeze(0)
    t = targets.unsqueeze(1)
    torch.set_printoptions(threshold=10_000)
        
    pos_ind = (t == class_range)
    neg_ind = (t != class_range) * (t >= 0)
    pos_prob = logits[pos_ind].sigmoid()
    neg_prob = logits[neg_ind].sigmoid()
    neg_q = F.softmax(neg_prob/neg_lambda, dim=0)
    neg_dist = torch.sum(neg_q * neg_prob)
    if pos_prob.numel() > 0:
        pos_q = F.softmax(-pos_prob/pos_lambda, dim=0)
        pos_dist = torch.sum(pos_q * pos_prob)
        loss = tau*torch.log(1.+torch.exp(L*(neg_dist - pos_dist+margin)))/L
    else:
        loss = tau*torch.log(1.+torch.exp(L*(neg_dist - 1. + margin)))/L
    return loss'''

def drLoss(logits, targets):
    margin = 0.5
    pos_lambda = 1
    neg_lambda = 0.1/math.log(3.5)
    L = 6.0 # Controlls the smootheness for loss function
    tau = 4.0 # Weight between classification and regression
    
    num_classes = logits.shape[1]
    N = logits.shape[0]
    dtype = targets.dtype
    device = targets.device
    class_range = torch.arange(0, num_classes, dtype=dtype, device=device).unsqueeze(0)
    t = targets.unsqueeze(1)
    torch.set_printoptions(threshold=10_000)

    pos_ind = (t == class_range)
    neg_ind = (t != class_range)

    a = neg_ind.size()
    b = pos_ind.size()
    c = logits[pos_ind]
    d = logits[neg_ind]

    pos_prob = logits[pos_ind].sigmoid()
    neg_prob = logits[neg_ind].sigmoid()

    posG = pos_prob.size()
    negG = neg_prob.size()

    neg_q = F.softmax(neg_prob/neg_lambda, dim=0)
    neg_dist = torch.sum(neg_q * neg_prob)
    if pos_prob.numel() > 0:
        pos_q = F.softmax(-pos_prob/pos_lambda, dim=0)
        pos_dist = torch.sum(pos_q * pos_prob)
        loss = tau*torch.log(1.+torch.exp(L*(neg_dist - pos_dist+margin)))/L
    else:
        loss = tau*torch.log(1.+torch.exp(L*(neg_dist - 1. + margin)))/L
    return loss