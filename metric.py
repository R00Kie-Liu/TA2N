import torch.nn.functional as F

def timewise_L2(x,y):
    x=x.mean([-1,-2])
    y=y.mean([-1,-2])
    x=F.normalize(x,dim=-2,p=2) # normed in C
    y=F.normalize(y,dim=-2,p=2) # n,m,C,T
    dist=((x-y)**2).sum(dim=(-1,-2))
    return dist

def timewise_cos(x,y):
    n,m,C,T=x.shape[:4]
    x=x.transpose(2,3)# C<->T
    y=y.transpose(2,3)# C<->T
    x=F.normalize(x.reshape(n,m,T,-1),dim=-1,p=2)
    y=F.normalize(y.reshape(n,m,T,-1),dim=-1,p=2)
    dist=(1-(x*y).sum(-1)).sum(-1)
    return dist