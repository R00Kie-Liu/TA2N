import torch

def theta_add1(x):
    a,b=x[:,0],x[:,1]
    return torch.stack([a+1,b],-1)


def gen_mask(offsets,resolution):
    '''
    g in (-1,1) for 7 pix
    '''
    with torch.no_grad():
        L=resolution
        shape=offsets.shape[:-1]
    offsets=offsets.reshape(-1,1,2)
    with torch.no_grad():
        N=offsets.size(0)
        grid=torch.arange(0,L).cuda()/(L-1)*2-1
        grid=grid.reshape(1,L,1).expand(N,-1,2)
    grid=grid+offsets
    full  = (-1<grid)*(grid<1)*1.0
    margin= (1-full)*(1-(grid.abs()-1)*((L-1)/2))
    margin=torch.nn.functional.relu(margin)
    mask_xy=full+margin
    mask_x,mask_y=mask_xy[...,0],mask_xy[...,1]
    mask=mask_y.unsqueeze(-1)*mask_x.unsqueeze(-2)
    mask=mask.reshape(*shape,resolution,resolution)
    return mask

class Squeeze(nn.Module):
    def __init__(self,*dim):
        super().__init__()
        if all(v>=0 for v in dim):
            self.dim=sorted(dim,reverse=True)
        elif all(v<0 for v in dim):
            self.dim=sorted(dim)

    def forward(self,x):
        for d in self.dim:
            x=torch.squeeze(x,dim=d)
        return x