import torch
import torch.nn as nn
import torch.nn.functional as F

def theta_add1(x):
    a,b=x[:,0],x[:,1]
    return torch.stack([a+1,b],-1)

def timewise_cos(x,y):
    n,m,C,T=x.shape[:4]
    x=x.transpose(2,3)# C<->T
    y=y.transpose(2,3)# C<->T
    x=F.normalize(x.reshape(n,m,T,-1),dim=-1,p=2)
    y=F.normalize(y.reshape(n,m,T,-1),dim=-1,p=2)
    dist=(1-(x*y).sum(-1)).sum(-1)
    return dist

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

class TTM(nn.Module):
    def __init__(self,T,shot=1,dim=(64,64)):
        super().__init__()
        self.T=T
        self.dim=dim
        self.shot=shot
        self.locnet=torch.nn.Sequential(
            nn.Conv3d(dim[0],64,3,padding=1),
            nn.BatchNorm3d(64),
            nn.MaxPool3d(2),
            nn.ReLU(),#5,4,4
            nn.Conv3d(64,128,3,padding=1),
            nn.BatchNorm3d(128),
            nn.MaxPool3d(2),
            nn.ReLU(),#3,2,2
            nn.AdaptiveMaxPool3d((1,1,1)),
            nn.Flatten(),#128
            nn.Linear(128,32),
            nn.ReLU(),
            nn.Linear(32,2),
            nn.Tanh(),
        )#[B,2] 2:=(a,b)=>ax+b
        self.locnet[-2].weight.data.zero_()
        self.locnet[-2].bias.data.copy_(torch.tensor([2.,0]))

    def align(self,feature,vis=False):
        n,C,T,H,W=feature.shape
        theta=self.locnet(feature)
        device=theta.device
        grid_t=torch.linspace(start=-1.0,end=1.0,steps=self.T).to(device).unsqueeze(0).expand(n,-1)
        grid_t=grid_t.reshape(n,1,T,1) #source uniform coord
        grid_t=torch.einsum('bc,bhtc->bht',theta,torch.cat([grid_t,torch.ones_like(grid_t)],-1)).unsqueeze(-1)
        grid=torch.cat([grid_t,torch.zeros_like(grid_t)-1.0],-1) # N*1*T*2 -> (t,-1.0)
        # grid=torch.min(torch.max(grid,-1*torch.ones_like(grid)),torch.ones_like(grid))
        #use gird to wrap support
        feature=feature.transpose(-3,-4).reshape(n,T,-1)
        feature=feature.transpose(-1,-2).unsqueeze(-2) # N*C*1*T
        feature_aligned=F.grid_sample(feature,grid,align_corners=True) #N*C*1*T
        #                               N*C*T             N*T*C
        feature_aligned=feature_aligned.squeeze(-2).transpose(-1,-2)\
                                    .reshape(n,T,-1,H,W).transpose(-3,-4)
        #                                  |S|*T*C*H*W
        if not vis:
            return feature_aligned
        else:
            return feature_aligned,theta.detach()

    def forward(self,support,query,vis=False):
        '''
        inputs must have shape of N*C*T*H*W
        return S*Q*T
        '''
        # support=support.mean([-1,-2])
        # query=query.mean([-1,-2])
        n,C,T,H,W=support.shape
        m=query.size(0)
        theta_support=self.locnet(support)
        theta_query=self.locnet(query)

        grid_t=torch.linspace(start=-1.0,end=1.0,steps=self.T).cuda().unsqueeze(0).expand(n,-1)
        grid_t=grid_t.reshape(n,1,T,1) #source uniform coord
        grid_t=torch.einsum('bc,bhtc->bht',theta_support,torch.cat([grid_t,torch.ones_like(grid_t)],-1)).unsqueeze(-1)
        grid=torch.cat([grid_t,torch.zeros_like(grid_t)-1.0],-1) # N*1*T*2 -> (t,-1.0)
        # grid=torch.min(torch.max(grid,-1*torch.ones_like(grid)),torch.ones_like(grid))
        grid_support=grid
        #use gird to wrap support
        support=support.transpose(-3,-4).reshape(n,T,-1)
        support=support.transpose(-1,-2).unsqueeze(-2) # N*C*1*T
        support_aligned=F.grid_sample(support,grid,align_corners=True) #N*C*1*T
        #                               N*C*T             N*T*C
        support_aligned=support_aligned.squeeze(-2).transpose(-1,-2)\
                                    .reshape(n,T,-1,H,W).transpose(-3,-4)
        #                                  |S|*T*C*H*W
        grid_t=torch.linspace(start=-1.0,end=1.0,steps=self.T).cuda().unsqueeze(0).expand(m,-1)
        grid_t=grid_t.reshape(m,1,T,1) #source uniform coord
        grid_t=torch.einsum('bc,bhtc->bht',theta_query,torch.cat([grid_t,torch.ones_like(grid_t)],-1)).unsqueeze(-1)

        grid=torch.cat([grid_t,torch.zeros_like(grid_t)-1.0],-1) # N*1*T*2 -> (t,-1.0)
        # grid=torch.min(torch.max(grid,-1*torch.ones_like(grid)),torch.ones_like(grid))
        grid_query=grid
        #use gird to wrap query
        query=query.transpose(-3,-4).reshape(m,T,-1)
        query=query.transpose(-1,-2).unsqueeze(-2) # N*C*1*T
        query_aligned=F.grid_sample(query,grid,align_corners=True) #N*C*1*T
        #                               N*C*T             N*T*C
        query_aligned=query_aligned.squeeze(-2).transpose(-1,-2)\
                                    .reshape(m,T,-1,H,W).transpose(-3,-4)
        #                                |Q|*T*C*H*W
        support_aligned=support_aligned #.unsqueeze(1).expand(n,m,C,T,H,W)
        query_aligned=query_aligned #.unsqueeze(0).expand(n,m,C,T,H,W)
        if vis:
            vis_dict={
                'grid_support':grid_support.clone().detach(),
                'grid_query':grid_query.clone().detach(),
                'theta_support':theta_support.clone().detach(),
                'theta_query':theta_query.clone().detach(),
            }
            return support_aligned, query_aligned, vis_dict
        else:
            return support_aligned, query_aligned

class ACM(nn.Module):
    def __init__(self,T,shot=1,dim=(2048,2048)):
        super().__init__()
        self.T=T
        self.shot=shot
        self.keynet=nn.Conv1d(*dim,kernel_size=1,bias=False)
        self.querynet=nn.Conv1d(*dim,kernel_size=1,bias=False)
        self.valuenet=nn.Conv1d(dim[0],dim[0],kernel_size=1,bias=False)
        self.dim=dim

        self.mvnet=nn.Sequential(
            nn.Conv3d(dim[0]*2,128,3,padding=1),
            nn.BatchNorm3d(128),
            nn.MaxPool3d((1,2,2)),
            nn.ReLU(),#8,4,4
            nn.Conv3d(128,128,3,padding=1),
            nn.BatchNorm3d(128),
            nn.MaxPool3d((1,2,2)),
            nn.ReLU(),#8,2,2
            nn.AdaptiveMaxPool3d((None,1,1)),
            Squeeze(-2,-1),#B,128,8
            nn.Conv1d(128,64,1),
            nn.ReLU(),
            nn.Conv1d(64,2,1),#B,(x,y)
            nn.Tanh()
        )
        self.mvnet[-2].weight.data.zero_()
        self.mvnet[-2].bias.data.zero_()

        with torch.no_grad():
            delta=0.2
            self.perturb=torch.tensor([[0,0],[0,1],[1,0],[0,-1],[-1,0],[1,1],[-1,-1],[1,-1],[-1,1]]).float().cuda()*delta
            self.perturb=self.perturb.reshape(1,1,9,2)

    def spatial_parameters(self):
        yield from self.mvnet.parameters()

    def temporal_coordinate(self,support,query):
        ''''
        The temporal coordinate (TC) module
        '''
        with torch.no_grad():
            n,m=support.size(0),query.size(0)
            _,C,T,H,W=support.shape
            vis_results={}
        rawsupport,rawquery=support, query
        support=rawsupport.mean((-2,-1))
        query  =rawquery  .mean((-2,-1))
        keys  =self.keynet(support)#NCT #.reshape(n,m,-1,T) # NMCT
        querys=self.querynet(query)#MCT #.reshape(n,m,-1,T) # NMCT
        attentions=torch.einsum('ncx,mcy->nmxy',keys,querys)/(self.dim[1]**0.5) #NMT(N)T(M)
        attentions=attentions.softmax(-1)
        # breakpoint()
        values=self.valuenet(query)# M,C,T
        query_aligned =rawquery+torch.einsum('nmxy,mcy->nmcx',attentions,values).unsqueeze(-1).unsqueeze(-1) #N, M, C, T, H, W 
        support_projed=rawsupport+self.valuenet(support).unsqueeze(-1).unsqueeze(-1)# N,C,T,hw
        #Temporal Done
        return support_projed, query_aligned

    def forward(self,support,query,vis=False):
        support_projed,query_aligned = self.temporal_coordinate(support,query)
        n,m=query_aligned.shape[:2]
        support_projed=support_projed.unsqueeze(1).expand(-1,m,-1,-1,-1,-1) # N, M, C, T, H, W
        pairs=torch.cat([support_projed,query_aligned],-4)#N, M, (C*2), T, H, W

        #C,T,H,W=self.dim[0],self.T,7,7
        C, T, H, W = support_projed.shape[2:]
        B=n*m

        pairs=pairs.reshape(-1,C*2,T,H,W) # N*M=B, 2C, T, H, W
        offsets=self.mvnet(pairs).transpose(1,2)#B,2,T -> B,T,2
        offsets=offsets*0.75
        raw_offsets=offsets

        if True or self.training:
            # Add perturb on offsets
            S=9
            offsets=offsets.unsqueeze(2)+self.perturb # B,T,1,2 + 1,1,S,2 => B,T,S,2
        else:
            pass
            S=1
            offsets=offsets.unsqueeze(2) # B,T,1,2

        mask=gen_mask(offsets, H) # B,T,S,H,W
        area=mask.sum([-1,-2],keepdim=True) # B,T,S
        mask=(mask/area).mean(2) # B,T,H,W
        mask=mask.reshape(n,m,1,T,H,W)
        #n,m,C,T,H,W
        support_projed=(mask*support_projed).sum([-1,-2]) # n,m,C,T

        mask=gen_mask(-offsets, H) # B,T,S,H,W
        area=mask.sum([-1,-2],keepdim=True)
        mask=(mask/area).mean(2) # B(NM),T,H,W
        mask=mask.reshape(n,m,1,T,H,W)
        #n,m,C,T,H,W
        query_aligned=(mask*query_aligned).sum([-1,-2]) # n,m,C,T
        #breakpoint()

        pairs=torch.cat([support_projed,query_aligned],2).unsqueeze(-1).unsqueeze(-1) #NM(C*2)THW

        return pairs, raw_offsets

    def decay(self):
        with torch.no_grad():
            self.perturb*=0.5

class TA2N(nn.Module):
    def __init__(self,T,shot=1,dim=(64,64),first_stage=True,second_stage=True,fusion='insupport', vis=False):
        super().__init__()
        if first_stage:
            self.firststage=TTM(T,shot,dim)
        else:
            print('Remove the 1st stage--TTM')
            self.firststage=None
        if second_stage==True:
            self.secondstage=ACM(T,shot,dim)
        else:
            self.secondstage=None
        self.fusion=fusion
        self.dim=dim
        self.shot=shot

        self.keynet_multi = nn.Conv1d(*dim,kernel_size=1,bias=False)
        self.querynet_multi = nn.Conv1d(*dim,kernel_size=1,bias=False)
        self.valuenet_multi = nn.Conv1d(dim[0],dim[0],kernel_size=1,bias=False)

        self.vis = vis

    def spatial_parameters(self):
        yield from self.secondstage.mvnet.parameters()

    def insupport_align(self,support_aligned):
        with torch.no_grad():
            k=self.shot
            n,C,T,H,W=support_aligned.shape
            n//=k
        support_input=support_aligned.mean([-1,-2]) # nk,C,T
        reference=support_input.reshape(n,k,C,T)[:,0,:,:]# n,C,T
        keys=self.keynet_multi(support_input).reshape(n,k,-1,T) # n,k,C,T
        querys=self.querynet_multi(reference)# n,C,T
        attention=torch.einsum('nkcx,ncy->nkxy',keys,querys)/(self.dim[1]**0.5) # n,kT,T
        attention=attention.softmax(dim=2)
        del keys,querys

        values=self.valuenet_multi(support_aligned.reshape(n*k,C,T*H*W)).reshape(n,k,C,T,H,W) # n,k,C,T,H,W
        support_recon=torch.einsum('nkxy,nkcxhw->nkcyhw',attention,values) #n,k,C,T
        prototype=support_recon.mean(1)

        return prototype


    def forward(self,support,query,sum_mask=None):
        '''
        inputs must have shape of N*C*T*H*W
        return S*Q*T
        '''
        vis = self.vis
        with torch.no_grad():
            n,m=support.size(0)//self.shot,query.size(0)
            k=self.shot
            C,T,H,W=support.shape[1:]
        # breakpoint()
        # original_query = query
        original_query = query.unsqueeze(0).expand(n, -1, -1, -1, -1, -1)
        if self.firststage:
            if not vis:
                support_aligned=self.firststage.align(support)
                query_aligned=self.firststage.align(query)
            else:
                support_aligned,theta_support=self.firststage.align(support,vis=True)
                query_aligned,theta_query=self.firststage.align(query,vis=True)
                vis_dict={
                    'first':{
                        'theta_support':theta_support,
                        'theta_query':theta_query,
                    }
                }
        else:
            support_aligned=support#.unsqueeze(1).expand(n,m,-1,-1,-1,-1)
            query_aligned=query#.unsqueeze(0).expand(n,m,-1,-1,-1,-1)
        # multi shot fusion
        if self.shot>1:
            prototype=self.insupport_align(support_aligned)
            # m,C,T,H,W -> m,C,T*H*W
            query_aligned = self.valuenet_multi(query_aligned.reshape(m,C,T*H*W)).reshape(m,C,T,H,W)
        else:
            prototype=support_aligned
            query_aligned=query_aligned
        # multi shot fusion end
        offsets=None
        #breakpoint()
        if self.secondstage:
            ##  ============= Conduct Second Stage ================
            pairs, offsets=self.secondstage(prototype,query_aligned,vis=vis)
            if vis:
                vis_dict['second'] = {'offset': offsets}
        else:
            pairs=torch.cat([prototype.unsqueeze(1).expand(n,m,-1,-1,-1,-1),query_aligned.unsqueeze(0).expand(n,m,-1,-1,-1,-1)],-4)#NM(C*2)THW
            rawsupport=None
        if not vis:
            return pairs, original_query
        
        else:
            return pairs, offsets, vis_dict

    def decay(self):
        if 'decay' in dir(self.secondstage):
            self.secondstage.decay()



if __name__ == '__main__':
    # stage1 = TTM(T=8,shot=1,dim=(2048,2048)).cuda()
    # stage2 = ACM(T=8, shot=1, dim=(2048,2048)).cuda()
    # def count_param(model):
    #     param_count = 0
    #     for param in model.parameters():
    #         param_count += param.view(-1).size()[0]
    #     return param_count
    import torch
    from thop import profile    
    support = torch.rand(5,2048,8,7,7).cuda() # N, C, T, H, W
    query = torch.rand(1,2048,8,7,7).cuda()
    # support, query = stage1(support, query)
    # pairs, offset = stage2(support, query)
    #breakpoint()

    ta2n = TA2N(T=8,shot=1, dim=(2048,2048),first_stage=True, second_stage=True, vis=True).cuda()
    # flops, params = profile(ta2n, inputs=(support, query))
    # breakpoint()
    with torch.autograd.profiler.profile(enabled=True, use_cuda=True) as prof:
        pairs, offsets, vis_dict = ta2n(support, query) # pairs: (N, M, C+C, T, 1, 1) offsets: (N*M, T, 2)
    print(prof.table())
    aligned_support, aligned_query = pairs[:,:,:2048,...],pairs[:,:,2048:,...]
    #breakpoint()
    dist = timewise_cos(aligned_support, aligned_query)
