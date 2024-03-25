import sys 
import math
import torch
import time
import random
import torch.nn as nn
from collections import OrderedDict
from itertools import combinations 
import torch.nn.functional as F
from torch.autograd import Variable
from models.OTAM import SoftDTW
from models.ta2n import TA2N
from utils import split_first_dim_linear
import torchvision.models as models

NUM_SAMPLES=1

def euclidean_dist(x, y, timewise=False):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    # return: N x M
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)

def cos_dist(x, y, timewise=False):
    if timewise:
        similarity_list = []
        for i in range(x.shape[1]):
            normalized_x = x[:,i,:] / x[:,i,:].norm(dim=-1, keepdim=True)
            normalized_y = y[:,i,:] / y[:,i,:].norm(dim=-1, keepdim=True)
            cos_similarity = normalized_x.mm(normalized_y.t())
            similarity_list.append(cos_similarity)
        similarity = torch.stack(similarity_list).sum(dim=0)
        return 1- similarity
    else:
        normalized_x = x / x.norm(dim=-1, keepdim=True)
        normalized_y = y / y.norm(dim=-1, keepdim=True)
        cos_similarity = normalized_x.mm(normalized_y.t())
        return 1 - cos_similarity


def all_timewise_cos(x,y):
    '''
    Input: x: (N, M, C, T, H, W)  y: (N, M, C, T, H, W)
    Return: dist: (N, M)
    '''
    n,m,C,T=x.shape[:4]
    x=x.transpose(2,3)# C<->T
    y=y.transpose(2,3)# C<->T
    x=F.normalize(x.reshape(n,m,T,-1),dim=-1,p=2)
    y=F.normalize(y.reshape(n,m,T,-1),dim=-1,p=2)
    dist=(1-(x*y).sum(-1)).sum(-1)
    return dist              

def extract_class_indices(labels, which_class):
    """
    Helper method to extract the indices of elements which have the specified label.
    :param labels: (torch.tensor) Labels of the context set.
    :param which_class: Label for which indices are extracted.
    :return: (torch.tensor) Indices in the form of a mask that indicate the locations of the specified label.
    """
    class_mask = torch.eq(labels, which_class)  # binary mask of labels equal to which_class
    class_mask_indices = torch.nonzero(class_mask)  # indices of labels equal to which class
    return torch.reshape(class_mask_indices, (-1,))  # reshape to be a 1D vector


class ProtypicalNet(nn.Module):
    def __init__(self, args):
        super(ProtypicalNet, self).__init__()

        self.args = args
        if args.metric == 'L2':
            self.metric = euclidean_dist
        elif args.metric == 'cos':
            self.metric = cos_dist
        elif args.metric == 'otam':
            self.metric = SoftDTW(use_cuda=True, gamma=0.1)
        self.timewise = self.args.timewise
        self.norm_layer = nn.LayerNorm(self.args.way)
    
    def forward(self, support_set, support_labels, queries):
        # queries: way, n_queries, dim, T, 1, 1
        # support_set: way, n_queries, dim, T, 1, 1
        n_queries = queries.shape[1]        
        all_distances_tensor = -all_timewise_cos(queries, support_set) # way, n_queries
        all_distances_tensor = all_distances_tensor.transpose(0,1) # n_queries, way
        if self.args.dist_norm:
            all_distances_tensor = self.norm_layer(all_distances_tensor)
        all_logits_tensor = all_distances_tensor.unsqueeze(0) # 1, n_queries, way
        
        return_dict = {'logits': all_logits_tensor}
            
        return return_dict

        

class CNN(nn.Module):
    """
    Standard Resnet connected to a Protypical Network.
    
    """
    def __init__(self, args):
        super(CNN, self).__init__()

        self.train()
        self.args = args

        if self.args.backbone == "resnet18":
            self.resnet = models.resnet18(pretrained=True)
            self.dim = 512
        elif self.args.backbone == "resnet34":
            self.resnet = models.resnet34(pretrained=True)
            self.dim = 512
        elif self.args.backbone == "resnet50":
            self.resnet = models.resnet50(pretrained=True)
            self.dim = 2048

        last_layer_idx = -1
        #self.resnet = nn.Sequential(*list(resnet.children())[:last_layer_idx])
        self.resnet.fc = nn.Identity()
        self.resnet.avgpool = nn.Identity()
        self.protypicalNet = ProtypicalNet(args)
        self.align = TA2N(T=args.seq_len, shot=args.shot, dim=(self.dim,self.dim), first_stage=True, second_stage=True)
        self.dropout = nn.Dropout(p=self.args.dropout)

    def forward(self, context_images, context_labels, target_images):
        # breakpoint()
        context_features = self.resnet(context_images)
        context_features = self.dropout(context_features)
        context_features = context_features.reshape(-1, self.args.seq_len, self.dim, 7, 7).transpose(1,2)
        target_features = self.resnet(target_images)
        target_features = self.dropout(target_features)
        target_features = target_features.reshape(-1, self.args.seq_len, self.dim, 7, 7).transpose(1,2)
        dim = int(context_features.shape[1])
        
        aligned_pair, offset = self.align(context_features, target_features)
        context_features = aligned_pair[:,:,:self.dim,...] # N/k, M, dim, T, 1, 1 
        target_features = aligned_pair[:,:,self.dim:,...] # N/K, M, dim, T, 1, 1
        #breakpoint()

        # context_features = context_features.reshape(-1, self.args.seq_len, dim)
        # target_features = target_features.reshape(-1, self.args.seq_len, dim)
        logits = self.protypicalNet(context_features, context_labels, target_features)
        
        return logits

    def distribute_model(self):
        """
        Distributes the CNNs over multiple GPUs.
        :return: Nothing
        """
        if self.args.num_gpus > 1:
            self.resnet = torch.nn.DataParallel(self.resnet, device_ids=[i for i in range(0, self.args.num_gpus)])
            self.resnet = self.resnet.cuda()
            self.protypicalNet = self.protypicalNet.cuda()
            self.align = self.align.cuda()
            




if __name__ == "__main__":
    class ArgsObject(object):
        def __init__(self):
            self.trans_linear_in_dim = 512
            self.trans_linear_out_dim = 128

            self.way = 5
            self.shot = 1
            self.query_per_class = 1
            self.query_per_class_test = 1
            self.trans_dropout = 0.1
            self.seq_len = 8 
            self.img_size = 224
            self.backbone = "resnet50"
            self.num_gpus = 1
            self.temp_set = [2]
            self.metric = 'cos'
            self.timewise = True
            self.dropout = 0.5
    args = ArgsObject()
    torch.manual_seed(0)
    
    device = 'cuda:0'
    model = CNN(args).to(device).eval()
    
    support_imgs = torch.rand(args.way*args.shot*args.seq_len, 3, args.img_size, args.img_size).to(device)
    target_imgs = torch.rand(args.way*args.query_per_class*args.seq_len, 3, args.img_size, args.img_size).to(device)
    #support_labels = torch.tensor([0,1,2,3,4]).to(device)
    support_labels = torch.tensor([0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4,0,1,2,3,4]).to(device)

    # breakpoint()
    # s = list(zip(support_imgs, support_labels))
    # random.shuffle(s)
    # support_imgs, support_labels = zip(*s)
    # support_imgs = torch.cat(support_imgs)
    # support_labels = torch.FloatTensor(support_labels)

    print("Support images input shape: {}".format(support_imgs.shape))
    print("Target images input shape: {}".format(target_imgs.shape))
    print("Support labels input shape: {}".format(support_labels.shape))

    # start = torch.cuda.Event(enable_timing=True)
    # end = torch.cuda.Event(enable_timing=True)
    # start.record()
    out = model(support_imgs, support_labels, target_imgs)
    # end.record()
    # torch.cuda.synchronize()
    # print('duration',start.elapsed_time(end))

    print("TRX returns the distances from each query to each class prototype.  Use these as logits.  Shape: {}".format(out['logits'].shape))

    #print("logits", out['logits'].sum(dim=1))




