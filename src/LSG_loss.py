import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def gradul_consistency_loss(adj_graph_list,train_labels,device='cpu'):
    loss = torch.tensor(0.).to(device)
    for i in range(1,len(adj_graph_list)):
       label_consistency = F.l1_loss(train_labels[i,:],train_labels[i-1,:],reduction='mean')
       adj_consistency   = F.l1_loss(adj_graph_list[i].flatten(),adj_graph_list[i-1].flatten(),reduction='sum')
       loss = loss + torch.mul((1-label_consistency),adj_consistency)
    return loss/(len(adj_graph_list)-1 + 0.1)

def regularization_loss(adj_graph_list,mode='L2',device='cpu'):
    loss = torch.tensor(0.).to(device)
    for adj in adj_graph_list:
        if mode=='L1':
            loss = loss + torch.sum(torch.abs(adj))
        elif mode == 'L2':
            loss = loss + torch.norm(adj,2)
    return loss/(len(adj_graph_list))




