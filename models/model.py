import torch
import torch.nn as nn
from models.utils import fps_subsample
from models.utils import MLP, MLP_CONV
import pointnet2_ops.pointnet2_utils as pn2


def symmetric_sample(points, num):
    p1_idx = pn2.furthest_point_sample(points, num)
    input_fps = pn2.gather_operation(points.transpose(1, 2).contiguous(), p1_idx).transpose(1, 2).contiguous()
    x = torch.unsqueeze(input_fps[:, :, 0], dim=2)
    y = torch.unsqueeze(input_fps[:, :, 1], dim=2)
    z = torch.unsqueeze(-input_fps[:, :, 2], dim=2)
    input_fps_flip = torch.cat([x, y, z], dim=2)
    input_fps = torch.cat([input_fps, input_fps_flip], dim=1)
    return input_fps


def knn(x, k):
    '''
    Args:
        x: (B, C, N)
    Returns:
        idx: (B, N, k)
    '''
    inner = -2*torch.matmul(x.transpose(2, 1), x)                              # (B, N, N)     
    xx = torch.sum(x**2, dim=1, keepdim=True)                                  # (B, 1, N)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)                       # (B, N, N)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]                               # (B, N, k)
    return idx.int()


def get_graph_feature(x, k=20, idx=None):
    '''
    Args:
        x: (B, C, N)
        idx: (B, N, k)
    Returns:
        feature: (B, C*2, N, k)
    '''
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)                                                      # (B, N, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size,
                            device=device).view(-1, 1, 1)*num_points           # (B, 1, 1)

    idx = idx + idx_base                                                       

    idx = idx.view(-1)                                                         # (B*N*k)

    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size*num_points, -1)[idx, :]                        # (B*N*k, C)
    feature = feature.view(batch_size, num_points, k, num_dims)                # (B, N, k, C)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)         # (B, N, k, C)
   
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()# (B, C*2, N, k)
    return feature
    

class point_shuffler(nn.Module):
    """
    Input:
        x: point cloud, [B, C1, N]
    Return:
        x: point cloud, [B, C1, scale*N]
    """   
    def __init__(self, scale=2):
        super(point_shuffler, self).__init__()
        self.scale = scale
    def forward(self, inputs):
        if self.scale == 1:
            ou = inputs
        else:
            B, C, N = inputs.shape
            x = inputs.permute([0,2,1])
            ou = x.reshape([B, N, self.scale, C//self.scale])
            ou = ou.reshape([B, N*self.scale, C//self.scale]).permute([0,2,1])
        return ou
    
    
class cross_attentinon(nn.Module):
    def __init__(self, d_model=256, d_model_out=256, nhead=4, dim_feedforward=1024, dropout=0.0):
        super(cross_attentinon, self).__init__()
        self.multihead_attn1 = nn.MultiheadAttention(d_model_out, nhead, dropout=dropout)
        self.linear11 = nn.Linear(d_model_out, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.linear12 = nn.Linear(dim_feedforward, d_model_out)
        self.norm12 = nn.LayerNorm(d_model_out)
        self.norm13 = nn.LayerNorm(d_model_out)
        self.dropout12 = nn.Dropout(dropout)
        self.dropout13 = nn.Dropout(dropout)
        self.activation1 = torch.nn.GELU()
        self.input_proj = nn.Conv1d(d_model, d_model_out, kernel_size=1)

    def forward(self, src1, src2):
        src1 = self.input_proj(src1)
        src2 = self.input_proj(src2)
        b, c, _ = src1.shape
        src1 = src1.reshape(b, c, -1).permute(2, 0, 1)
        src2 = src2.reshape(b, c, -1).permute(2, 0, 1)
        src1 = self.norm13(src1)
        src2 = self.norm13(src2)
        src12 = self.multihead_attn1(query=src1,
                                     key=src2,
                                     value=src2)[0]
        src1 = src1 + self.dropout12(src12)
        src1 = self.norm12(src1)
        src12 = self.linear12(self.dropout1(self.activation1(self.linear11(src1))))
        src1 = src1 + self.dropout13(src12)
        src1 = src1.permute(1, 2, 0)
        return src1


class multi_scale_attention_encoder(nn.Module):
    def __init__(self, num_coarse=512, u1=4, u2=8):
        super(multi_scale_attention_encoder, self).__init__()
        self.u1 = u1
        self.u2 = u2
        self.num_coarse = num_coarse
        self.mlp_0 = MLP_CONV(3, [32,64])
        self.mlp_1 = MLP_CONV(3, [32,64])
        self.mlp_2 = MLP_CONV(3, [32,64])
        self.atten1 = cross_attentinon(64, 64)
        self.atten2 = cross_attentinon(64, 64)
        # self.mlp_up = MLP_CONV(64, [128, 512])
        # self.mlp_coarse = MLP(512, [512, num_coarse*3])
        
    def forward(self, xyz):
        B, N, _ = xyz.shape
        level0 = xyz
        level1 = fps_subsample(level0, int(2048/self.u1))
        level2 = fps_subsample(level1, int(2048/self.u2))
        
        level0 = level0.permute(0, 2, 1).contiguous() # B 3 2048
        level1 = level1.permute(0, 2, 1).contiguous() # B 3 512
        level2 = level2.permute(0, 2, 1).contiguous() # B 3 256
        
        level0 = self.mlp_0(level0) # B 64 2048
        level1 = self.mlp_1(level1) # B 64 512
        level2 = self.mlp_2(level2) # B 64 256
        
        level1 = self.atten1(level1, level0) # B 64 512
        level2 = self.atten2(level2, level1) # B 64 256
        # x_256 = self.mlp_up(x_256) # B 512 256
        global_feature = torch.max(level2, 2)[0] # B 64
        # coarse = self.mlp_coarse(global_feature).reshape(B, -1, self.num_coarse) # B 3 512
        return global_feature


class geometric_detail_extractor(nn.Module):
    def __init__(self, in_dim=64, out_dim=64, k=20, up_factor=2):
        super(geometric_detail_extractor, self).__init__()
        self.k = k
        self.conv = nn.Sequential(nn.Conv2d(in_dim*2, out_dim // 2, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(out_dim // 2),
                                   nn.LeakyReLU(negative_slope=0.2),
                                   nn.Conv2d(out_dim // 2, out_dim // 2, kernel_size=1, bias=False),
                                   nn.BatchNorm2d(out_dim // 2),
                                   nn.LeakyReLU(negative_slope=0.2),
                                   nn.Conv2d(out_dim // 2, out_dim, kernel_size=1, bias=False),
                                   )
        self.PF = point_shuffler(scale=up_factor)

    def forward(self, xyz, xyz_f):
        B, _, _ = xyz.shape
        idx = knn(xyz, self.k)
        f = get_graph_feature(xyz_f, self.k, idx)                              # B 2*in_dim N k
        f = self.conv(f)
        f = torch.max(f, dim=-1)[0]                                            # B out_dim N
        f = self.PF(f)
        return f                                                               # B out_dim/up_factor N*up_factor
    
    
class structural_refinement_module(nn.Module):
    def __init__(self, i=1, up_factor=2, k=20, dim1=64, dim2=128, dim3=64):
        super(structural_refinement_module, self).__init__()
        self.up_factor = up_factor
        self.up = nn.Upsample(scale_factor=up_factor)
        self.mlp_up = MLP(64, [128, 512+i*256])
        self.mlp_coarse = MLP_CONV(3 + 512+i*256, [dim1])
        self.gde = geometric_detail_extractor(dim1, dim2, k, up_factor)
        self.delta = MLP_CONV(3 + dim1 + int(dim2/up_factor) + dim3, [256, 256, 3])
        self.mlp_g = MLP_CONV(dim1, [64, 32])
        self.deconv = nn.ConvTranspose1d(32, dim3, up_factor, up_factor, bias=False)
        self.PF = point_shuffler(scale=up_factor)
        
    def forward(self, coarse, g_f):
        B, _, N = coarse.size()
        g_f = self.mlp_up(g_f)
        # project feature
        input_cat = torch.cat([coarse, g_f.unsqueeze(2).repeat(1, 1, N)], dim=1)
        coarse_f = self.mlp_coarse(input_cat) # B dim1 N
        # Detail/Local Extractor
        detail_f = self.gde(coarse, coarse_f) # B dim2/up_factor N*up_factor
        # Global feature
        global_f = self.deconv(self.mlp_g(coarse_f)) # B dim3 N*up_factor
        # output
        delta_cat = torch.cat([self.up(coarse), self.up(coarse_f), detail_f, global_f], dim=1)
        fine = self.up(coarse) + self.delta(delta_cat) # B 3 N*up_factor
        return fine
    

class FPCN(nn.Module):
    def __init__(self, num_coarse=512, up_factor=[2]):
        super(FPCN, self).__init__()
        self.num_coarse = num_coarse
        self.encoder = multi_scale_attention_encoder(num_coarse)
        self.mlp_up = MLP(64, [128, num_coarse])
        self.mlp_coarse = MLP(num_coarse, [num_coarse, num_coarse*3])
        SR = []
        for i, factor in enumerate(up_factor):
            SR.append(structural_refinement_module(i+1, factor))
        self.SR = nn.ModuleList(SR)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        g_f = self.encoder(xyz)
        # fc coarse decoder
        coarse = self.mlp_coarse(self.mlp_up(g_f)).reshape(B, -1, self.num_coarse) # B 3 num_coarse
        # symmetric partial input prior
        input_fps = symmetric_sample(xyz, int(self.num_coarse/2))  # B 512 3
        input_fps = input_fps.transpose(2, 1).contiguous()  # B 3 512
        new_x = torch.cat([input_fps, coarse], 2) # B 3 1024
        # lifting module
        coarse_input = new_x
        out = []
        out.append(coarse.transpose(1, 2).contiguous())
        # out.append(coarse_input.transpose(1, 2).contiguous())
        for layer in self.SR:
            coarse_input = layer(coarse_input, g_f)
            out.append(coarse_input.transpose(1, 2).contiguous())
        return out # 512 2048
