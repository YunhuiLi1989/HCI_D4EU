import torch.nn as nn
from einops import rearrange
from torch import einsum
import torch
import torch.nn.functional as F
import torch_dct as dct


########################################################################################################################
class BlockSpaAttn(nn.Module):
    def __init__(self, dim, window_size=(8, 8), dim_head=28, heads=1, shift=False):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.scale = dim ** -0.5
        self.window_size = window_size
        self.shift = shift
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(dim, dim, 1, bias=True)
        self.apply(self.init_weight)
    def init_weight(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02, a=-2.0, b=2.0)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        w_size = self.window_size
        if self.shift:
            x = x.roll(shifts=w_size[0] // 2, dims=2).roll(shifts=w_size[1] // 2, dims=3)

        bs, nC, H, W = x.shape
        ph, pw = H // self.window_size[0], W // self.window_size[1]
        q, k, v = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b c (ph h) (pw w) -> b (ph pw) (h w) c', ph=ph, pw=pw), (q, k, v))
        q = q * self.scale
        sim = einsum('b p i d, b p j d -> b p i j', q, k)
        attn = sim.softmax(dim=-1)
        out = einsum('b p i j, b p j d -> b p i d', attn, v)
        out = rearrange(out, 'b (ph pw) (h w) c -> b c (ph h) (pw w)', ph=ph, h=H // ph)
        out = self.to_out(out)

        if self.shift:
            out = out.roll(shifts=-1 * w_size[1] // 2, dims=3).roll(shifts=-1 * w_size[0] // 2, dims=2)

        return out

########################################################################################################################
def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

def get_topk_closest_indice(q_windows, k_windows, topk=1):
    # get pair-wise relative position index for each token inside the window
    coords_h_q = torch.arange(q_windows[0])
    coords_w_q = torch.arange(q_windows[1])

    if q_windows[0] != k_windows[0]:
        factor = k_windows[0] // q_windows[0]
        coords_h_q = coords_h_q * factor + factor // 2
        coords_w_q = coords_w_q * factor + factor // 2
    else:
        factor = 1

    coords_q = torch.stack(torch.meshgrid([coords_h_q, coords_w_q]))  # 2, Wh_q, Ww_q

    coords_h_k = torch.arange(k_windows[0])  # --20241209--[0...H]
    coords_w_k = torch.arange(k_windows[1])  # --20241209--[0...W]
    coords_k = torch.stack(torch.meshgrid([coords_h_k, coords_w_k]))  # 2, Wh, Ww

    coords_flatten_q = torch.flatten(coords_q, 1)  # 2, Wh_q*Ww_q
    coords_flatten_k = torch.flatten(coords_k, 1)  # 2, Wh_k*Ww_k

    relative_coords = coords_flatten_q[:, :, None] - coords_flatten_k[:, None, :]  # 2, Wh_q*Ww_q, Wh_k*Ww_k

    relative_position_dists = torch.sqrt(relative_coords[0].float() ** 2 + relative_coords[1].float() ** 2)

    topk = min(topk, relative_position_dists.shape[1])
    topk_score_k, topk_index_k = torch.topk(-relative_position_dists, topk, dim=1)  # B, topK, nHeads
    indice_topk = topk_index_k
    relative_coord_topk = torch.gather(relative_coords, 2, indice_topk.unsqueeze(0).repeat(2, 1, 1))
    return indice_topk, relative_coord_topk.permute(1, 2, 0).contiguous().float(), topk


class WindowExdAttention(nn.Module):
    def __init__(self, dim, window_size=(4, 4), focal_level=1, num_heads=1, topK=100):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.input_resolution = 256 // (dim // 28)  # NWh, NWw
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.focal_level = focal_level
        self.nWh, self.nWw = (256 // (dim // 28)) // self.window_size[0], (256 // (dim // 28)) // self.window_size[1]

        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.proj = nn.Conv2d(dim, dim, 1, bias=True)

        self.topK = topK
        coords_h_window = torch.arange(self.window_size[0]) - self.window_size[0] // 2
        coords_w_window = torch.arange(self.window_size[1]) - self.window_size[1] // 2
        coords_window = torch.stack(torch.meshgrid([coords_h_window, coords_w_window]), dim=-1)  # 2, Wh_q, Ww_q
        self.register_buffer("window_coords", coords_window)

        self.topks = []
        for k in range(self.focal_level):
            if k == 0:
                range_h = self.input_resolution  # --20241209--H
                range_w = self.input_resolution  # --20241209--W
            else:
                range_h = self.nWh
                range_w = self.nWw

            # build relative position range
            topk_closest_indice, topk_closest_coord, topK_updated = get_topk_closest_indice(
                (self.nWh, self.nWw), (range_h, range_w), self.topK)
            self.topks.append(topK_updated)

            if k > 0:
                # scaling the coordinates for pooled windows
                topk_closest_coord = topk_closest_coord * self.window_size[0]
            topk_closest_coord_window = topk_closest_coord.unsqueeze(1) + coords_window.view(-1, 2)[None, :, None, :]

            self.register_buffer("topk_cloest_indice_{}".format(k), topk_closest_indice)
            self.register_buffer("topk_cloest_coords_{}".format(k), topk_closest_coord_window)

        self.apply(self.init_weight)

    def init_weight(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02, a=-2.0, b=2.0)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        B, C, nH, nW = x.shape
        qkv = self.qkv(x).reshape(B, 3, C, nH, nW).permute(1, 0, 3, 4, 2).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]  # B, nH, nW, C

        # partition q map  # --20241209--(bs,H,W,nC)->(bs*ph*pw,nh,hw,nC/nh)
        q_windows = window_partition(q, self.window_size[0]).view(-1, self.window_size[0] * self.window_size[0], self.num_heads, C // self.num_heads).transpose(1, 2)

        k_all = []
        v_all = []
        topKs = []
        for l_k in range(self.focal_level):
            topk_closest_indice = getattr(self, "topk_cloest_indice_{}".format(l_k))
            topk_indice_k = topk_closest_indice.view(1, -1).repeat(B, 1)

            if l_k == 0:
                k_k = k.view(B, -1, self.num_heads, C // self.num_heads)
                v_k = v.view(B, -1, self.num_heads, C // self.num_heads)
            else:
                x_k = x
                qkv_k = self.qkv(x_k).view(B, -1, 3, self.num_heads, C // self.num_heads)
                k_k, v_k = qkv_k[:, :, 1], qkv_k[:, :, 2]

            k_k_selected = torch.gather(k_k, 1, topk_indice_k.view(B, -1, 1).unsqueeze(-1).repeat(1, 1, self.num_heads, C // self.num_heads))
            v_k_selected = torch.gather(v_k, 1, topk_indice_k.view(B, -1, 1).unsqueeze(-1).repeat(1, 1, self.num_heads, C // self.num_heads))

            k_k_selected = k_k_selected.view((B,) + topk_closest_indice.shape + (self.num_heads, C // self.num_heads,)).transpose(2, 3)
            v_k_selected = v_k_selected.view((B,) + topk_closest_indice.shape + (self.num_heads, C // self.num_heads,)).transpose(2, 3)

            k_all.append(k_k_selected.view(-1, self.num_heads, topk_closest_indice.shape[1], C // self.num_heads))
            v_all.append(v_k_selected.view(-1, self.num_heads, topk_closest_indice.shape[1], C // self.num_heads))
            topKs.append(topk_closest_indice.shape[1])

        k_all = torch.cat(k_all, 2)
        v_all = torch.cat(v_all, 2)

        q_windows = q_windows * self.scale
        sim = (q_windows @ k_all.transpose(-2, -1))  # B*nW, nHead, window_size*window_size, focal_window_size*focal_window_size
        attn = sim.softmax(dim=-1)

        x = (attn @ v_all).transpose(1, 2).flatten(2)
        x = window_reverse(x, self.window_size[0], nH, nW)

        out = self.proj(x.permute(0, 3, 1, 2))
        return out

########################################################################################################################


class BlockSpeAttn(nn.Module):
    def __init__(self, dim, window_size=(4, 4), dim_head=28, heads=1, shift=False):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.scale = (window_size[0] * window_size[1]) ** -0.5
        self.window_size = window_size
        self.shift = shift
        self.to_qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(dim, dim, 1, bias=True)
        self.apply(self.init_weight)
    def init_weight(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02, a=-2.0, b=2.0)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        w_size = self.window_size
        if self.shift:
            x = x.roll(shifts=w_size[0] // 2, dims=2).roll(shifts=w_size[1] // 2, dims=3)

        bs, nC, H, W = x.shape
        ph, pw = H // self.window_size[0], W // self.window_size[1]
        q, k, v = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b c (ph h) (pw w) -> b (ph pw) c (h w) ', ph=ph, pw=pw), (q, k, v))
        q = q * self.scale
        sim = einsum('b p i d, b p j d -> b p i j', q, k)
        attn = sim.softmax(dim=-1)
        out = einsum('b p i j, b p j d -> b p i d', attn, v)
        out = rearrange(out, 'b (ph pw) c (h w) -> b c (ph h) (pw w)', ph=ph, h=H // ph)
        out = self.to_out(out)

        if self.shift:
            out = out.roll(shifts=-1 * w_size[1] // 2, dims=3).roll(shifts=-1 * w_size[0] // 2, dims=2)

        return out


class SpaFFT(nn.Module):
    def __init__(self, dim):
        super().__init__()
        if dim == 28:
            self.Hf = 256
            self.Wf = 129
            self.complex_weight = nn.Parameter(torch.randn(self.Hf, self.Wf, dim, 2, dtype=torch.float32) * 0.02)
        if dim == 56:
            self.Hf = 128
            self.Wf = 65
            self.complex_weight = nn.Parameter(torch.randn(self.Hf, self.Wf, dim, 2, dtype=torch.float32) * 0.02)
        if dim == 112:
            self.Hf = 64
            self.Wf = 33
            self.complex_weight = nn.Parameter(torch.randn(self.Hf, self.Wf, dim, 2, dtype=torch.float32) * 0.02)

    def forward(self, x):
        bs, nC, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        x = x.to(torch.float32)

        xf = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        xf = xf * weight
        out = torch.fft.irfft2(xf, s=(H, W), dim=(1, 2), norm='ortho')

        out = out.permute(0, 3, 1, 2)
        return out



class SpeDCT(nn.Module):
    def __init__(self, dim):
        super().__init__()
        if dim == 28:
            self.weight = nn.Parameter(torch.randn(256, 256, dim, dtype=torch.float32) * 0.02)
        if dim == 56:
            self.weight = nn.Parameter(torch.randn(128, 128, dim, dtype=torch.float32) * 0.02)
        if dim == 112:
            self.weight = nn.Parameter(torch.randn(64, 64, dim, dtype=torch.float32) * 0.02)


    def forward(self, x):
        B, nC, H, W = x.shape
        x = x.permute(0, 2, 3, 1)
        x = x.to(torch.float32)

        xd = dct.dct(x)
        xd = xd * self.weight
        out = dct.idct(xd)

        out = out.permute(0, 3, 1, 2)
        return out



class FFN(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
                                 nn.GELU(),
                                 nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
                                 nn.GELU(),
                                 nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
                                 )

    def forward(self, x):
        out = self.net(x)
        return out


class PCFN(nn.Module):
    def __init__(self, dim, growth_rate=2.0, p_rate=0.25):
        super().__init__()
        hidden_dim = int(dim * growth_rate)
        p_dim = int(hidden_dim * p_rate)
        self.conv_0 = nn.Conv2d(dim, hidden_dim, 1, 1, 0)
        self.conv_1 = nn.Conv2d(p_dim, p_dim, 3, 1, 1)

        self.act = nn.GELU()
        self.conv_2 = nn.Conv2d(hidden_dim, dim, 1, 1, 0)

        self.p_dim = p_dim
        self.hidden_dim = hidden_dim

    def forward(self, x):
        x = self.act(self.conv_0(x))
        x1, x2 = torch.split(x, [self.p_dim, self.hidden_dim-self.p_dim], dim=1)
        x1 = self.act(self.conv_1(x1))
        x = self.conv_2(torch.cat([x1, x2], dim=1))
        return x


class AttnFilterBlock1(nn.Module):
    def __init__(self, dim, window_size=(8, 8), dim_head=28, heads=1):
        super().__init__()
        self.LN1 = nn.LayerNorm(dim)
        self.BlockSpaAttn = WindowExdAttention(dim=dim)
        self.LN2 = nn.LayerNorm(dim)
        self.BlockSpeAttn = BlockSpeAttn(dim=dim, window_size=window_size, dim_head=dim_head, heads=heads, shift=False)
        self.project = nn.Conv2d(2 * dim, dim, 1, 1, 0, bias=True)
        self.GN = nn.LayerNorm(dim)
        self.FFN = FFN(dim=dim)

    def forward(self, x):
        xa1 = self.LN1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        xb1 = self.BlockSpaAttn(xa1)
        xa2 = self.LN2(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        xb2 = self.BlockSpeAttn(xa2)

        xb = torch.cat([xb1, xb2], dim=1)
        xc = self.project(xb) + x

        # xd = self.GN(xc)
        xd = self.GN(xc.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        out = self.FFN(xd) + xc
        return out



class AttnFilterBlock2(nn.Module):
    def __init__(self, dim, window_size=(8, 8), dim_head=28, heads=1):
        super().__init__()
        self.LN3 = nn.LayerNorm(dim)
        self.SpeDCT = SpeDCT(dim=dim)
        self.LN4 = nn.LayerNorm(dim)
        self.SpaFFT = SpaFFT(dim=dim)
        self.project = nn.Conv2d(2 * dim, dim, 1, 1, 0, bias=True)
        self.GN = nn.LayerNorm(dim)
        self.FFN = FFN(dim=dim)

    def forward(self, x):
        xa3 = self.LN3(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        xb3 = self.SpeDCT(xa3)
        xa4 = self.LN4(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        xb4 = self.SpaFFT(xa4)

        xb = torch.cat([xb3, xb4], dim=1)
        xc = self.project(xb) + x

        xd = self.GN(xc.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        out = self.FFN(xd) + xc
        return out

########################################################################################################################
def kaiming_init(module,
                 a=0,
                 mode='fan_out',
                 nonlinearity='relu',
                 bias=0,
                 distribution='normal'):
    assert distribution in ['uniform', 'normal']
    if distribution == 'uniform':
        nn.init.kaiming_uniform_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    else:
        nn.init.kaiming_normal_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

class PSA_p(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=1, stride=1):
        super(PSA_p, self).__init__()

        self.inplanes = inplanes
        self.inter_planes = planes // 2
        self.planes = planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (kernel_size-1)//2

        self.conv_q_right = nn.Conv2d(self.inplanes, 1, kernel_size=1, stride=stride, padding=0, bias=False)
        self.conv_v_right = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0, bias=False)
        self.conv_up = nn.Conv2d(self.inter_planes, self.planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.softmax_right = nn.Softmax(dim=2)

        self.sigmoid = nn.Sigmoid()

        self.conv_q_left = nn.Conv2d(self.inplanes, self.inter_planes * 4, kernel_size=1, stride=stride, padding=0, bias=False)   #g
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_v_left = nn.Conv2d(self.inplanes, self.inter_planes, kernel_size=1, stride=stride, padding=0, bias=False)   #theta
        self.softmax_left = nn.Softmax(dim=2)

        self.reset_parameters()

    def reset_parameters(self):

        kaiming_init(self.conv_q_right, mode='fan_in')
        kaiming_init(self.conv_v_right, mode='fan_in')

        kaiming_init(self.conv_q_left, mode='fan_in')
        kaiming_init(self.conv_v_left, mode='fan_in')

        self.conv_q_right.inited = True
        self.conv_v_right.inited = True

        self.conv_q_left.inited = True
        self.conv_v_left.inited = True


    def spatial_pool(self, x):
        input_x = self.conv_v_right(x)
        batch, channel, height, width = input_x.size()

        regionsize = 8   #--20241221--区域精细化size

        input_x = rearrange(input_x, 'b c (ph h) (pw w) -> (b ph pw) c (h w) ', h=regionsize, w=regionsize)
        # [N, IC, H*W]
        # input_x = input_x.view(batch, channel, height * width)
        # [N, 1, H, W]
        context_mask = self.conv_q_right(x)
        context_mask = rearrange(context_mask, 'b c (ph h) (pw w) -> (b ph pw) c (h w) ', h=regionsize, w=regionsize)
        # [N, 1, H*W]
        # context_mask = context_mask.view(batch, 1, height * width)
        # [N, 1, H*W]
        context_mask = self.softmax_right(context_mask)
        # [N, IC, 1]
        # context = torch.einsum('ndw,new->nde', input_x, context_mask)
        context = torch.matmul(input_x, context_mask.transpose(1, 2))
        # [N, IC, 1, 1]
        context = context.unsqueeze(-1)
        # [N, OC, 1, 1]
        context = self.conv_up(context)
        context = context.squeeze(2, 3).view(batch, height//regionsize, width//regionsize, self.inplanes)
        context = torch.repeat_interleave(context, repeats=regionsize, dim=1)
        context = torch.repeat_interleave(context, repeats=regionsize, dim=2)
        context = context.permute(0, 3, 1, 2)
        # [N, OC, 1, 1]
        mask_ch = self.sigmoid(context)
        out = x * mask_ch
        return out


    def channel_pool(self, x):
        # [N, IC, H, W]
        g_x = self.conv_q_left(x)
        batch, channel, height, width = g_x.size()
        # [N, IC, 1, 1]
        avg_x = self.avg_pool(g_x)
        batch, channel, avg_x_h, avg_x_w = avg_x.size()
        # [N, 1, IC]
        avg_x = avg_x.view(batch, channel, avg_x_h * avg_x_w).permute(0, 2, 1)
        avg_x = (avg_x.squeeze(1)).view(batch, 4, -1)
        # [N, IC, H*W]
        theta_x = self.conv_v_left(x).view(batch, self.inter_planes, height * width)
        # [N, 1, H*W]
        # context = torch.einsum('nde,new->ndw', avg_x, theta_x)
        context = torch.matmul(avg_x, theta_x)
        # [N, 1, H*W]
        context = self.softmax_left(context)
        # [N, 1, H, W]
        context = context.view(batch, 4, height, width)
        context = torch.repeat_interleave(context, repeats=self.inplanes // 4, dim=1)
        # [N, 1, H, W]
        mask_sp = self.sigmoid(context)
        out = x * mask_sp
        return out

    def forward(self, x):
        # [N, C, H, W]
        context_channel = self.spatial_pool(x)
        # [N, C, H, W]
        context_spatial = self.channel_pool(x)
        # [N, C, H, W]
        out = context_spatial + context_channel
        return out + x


########################################################################################################################
class SSRB(nn.Module):
    def __init__(self, dim, window_size=(8, 8), dim_head=28, heads=1):
        super().__init__()
        self.PosEmb = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim)
        self.AttnFilterBlock1 = AttnFilterBlock1(dim=dim, window_size=window_size, dim_head=dim_head, heads=heads)
        self.AttnFilterBlock2 = AttnFilterBlock2(dim=dim, window_size=window_size, dim_head=dim_head, heads=heads)
        self.PSA_p1 = PSA_p(inplanes=dim, planes=dim)

    def forward(self, x):
        xa = self.PosEmb(x) + x
        xb = self.AttnFilterBlock1(xa)
        xc = self.AttnFilterBlock2(xb)
        xd = self.PSA_p1(xc)

        return xd


class MaskEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Conv2d(28, 28, 3, 1, 1, bias=False)
        self.mask = nn.Conv2d(28, 28, 3, 1, 1, bias=False)
        self.apply(self.init_weight)

    def init_weight(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02, a=-2.0, b=2.0)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, mask):   #--20241120--(bs,nC,H,W),(bs,nC,H,W)
        out = self.cnn(x) * (1 + self.mask(mask))
        return out


class SSRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.mask_embedding = MaskEmbedding()
        self.down1 = SSRB(dim=28, window_size=(8, 8), dim_head=28, heads=1)
        self.downsample1 = nn.Conv2d(28, 56, 4, 2, 1, bias=False)
        self.down2 = SSRB(dim=56, window_size=(8, 8), dim_head=28, heads=2)
        self.downsample2 = nn.Conv2d(56, 112, 4, 2, 1, bias=False)
        self.bottleneck = SSRB(dim=112, window_size=(8, 8), dim_head=28, heads=4)
        self.upsample2 = nn.ConvTranspose2d(112, 56, 2, 2)
        self.fusion2 = nn.Conv2d(112, 56, 1, 1, 0, bias=False)
        self.up2 = SSRB(dim=56, window_size=(8, 8), dim_head=28, heads=2)
        self.upsample1 = nn.ConvTranspose2d(56, 28, 2, 2)
        self.fusion1 = nn.Conv2d(56, 28, 1, 1, 0, bias=False)
        self.up1 = SSRB(dim=28, window_size=(8, 8), dim_head=28, heads=1)
        self.out = nn.Conv2d(28, 28, 3, 1, 1, bias=False)
        self.apply(self.init_weight)

    def init_weight(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02, a=-2.0, b=2.0)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, mask):
        xin = self.mask_embedding(x, mask)
        x1 = self.down1(xin)
        xin = self.downsample1(x1)
        x2 = self.down2(xin)
        xin = self.downsample2(x2)
        xin = self.bottleneck(xin)
        xin = self.upsample2(xin)
        xin = self.fusion2(torch.cat([xin, x2], dim=1))
        xin = self.up2(xin)
        xin = self.upsample1(xin)
        xin = self.fusion1(torch.cat([xin, x1], dim=1))
        xin = self.up1(xin)
        out = self.out(xin) + x

        return out

########################################################################################################################

class DeltaEst(nn.Module):
    def __init__(self, in_dim=28, out_dim=1, med_dim=32):
        super().__init__()
        self.fusion = nn.Conv2d(in_dim, med_dim, 1, 1, 0, bias=True)
        self.bias = nn.Parameter(torch.FloatTensor([1.]))
        self.avpool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(nn.Conv2d(med_dim, med_dim, 1, padding=0, bias=True),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(med_dim, med_dim, 1, padding=0, bias=True),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(med_dim, out_dim, 1, padding=0, bias=False)
                                 )
        self.relu = nn.ReLU(inplace=True)
        self.apply(self.init_weight)

    def init_weight(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02, a=-2.0, b=2.0)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.relu(self.fusion(x))
        x = self.avpool(x)
        x = self.mlp(x) + self.bias
        return x


class D4EU(torch.nn.Module):
    def __init__(self, opt):
        super(D4EU, self).__init__()
        self.stage = opt.stage

        self.initial = nn.Conv2d(56, 28, 1, 1, 0, bias=True)

        NetLayer = []
        for i in range(opt.stage):
            NetLayer.append(SSRU())
            NetLayer.append(nn.Identity())
        self.net_stage = nn.ModuleList(NetLayer)

        DeltaLayer = []
        for i in range(opt.stage):
            DeltaLayer.append(DeltaEst())
        self.rhos = nn.ModuleList(DeltaLayer)


    def shift_back(self, x, len_shift=2):
        for i in range(28):
            x[:, i, :, :] = torch.roll(x[:, i, :, :], shifts=(-1) * len_shift * i, dims=2)
        return x[:, :, :, :256]

    def shift(self, x, len_shift=2):
        x = F.pad(x, [0, 28 * 2 - 2, 0, 0], mode='constant', value=0)
        for i in range(28):
            x[:, i, :, :] = torch.roll(x[:, i, :, :], shifts=len_shift * i, dims=2)
        return x

    def mul_Phif(self, Phi_shift, x):
        x_shift = self.shift(x)
        Phix = Phi_shift * x_shift
        Phix = torch.sum(Phix, 1)
        return Phix.unsqueeze(1)

    def mul_PhiTg(self, Phi_shift, x):
        temp_1 = x.repeat(1, Phi_shift.shape[1], 1, 1).cuda()
        PhiTx = temp_1 * Phi_shift
        PhiTx = self.shift_back(PhiTx)
        return PhiTx


    def forward(self, g, input_mask):

        Phi, PhiPhiT = input_mask
        Phi_shift = self.shift(Phi, len_shift=2)

        meas_r = g.repeat(1, 28, 1, 1)
        meas_s = self.shift_back(meas_r)

        xin = self.initial(torch.cat([meas_s, Phi], dim=1))

        out = []
        for i in range(self.stage):

            rho = self.rhos[i](xin)
            Phi_x = self.mul_Phif(Phi_shift, xin)
            xa = xin + rho * self.mul_PhiTg(Phi_shift, torch.div(g - Phi_x, PhiPhiT))

            xb = self.net_stage[2 * i](xa, Phi)

            xc = self.net_stage[2 * i + 1](xb)

            out.append(xc)
            xin = xc

        return out













