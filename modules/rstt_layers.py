import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias.data, 0.0)


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out

def window_partition(x, window_size):
    """Partition the input video sequences into several windows along spatial 
    dimensions.

    Args:
        x (torch.Tensor): (B, D, H, W, C)
        window_size (tuple[int]): Window size

    Returns: 
        windows: (B*nW, D, Wh, Ww, C)
    """
    B, D, H, W, C = x.shape
    # B, D, num_Hwin, Wh, num_Wwin, Ww, C
    x = x.view(B, D, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C) 
    windows = x.permute(0, 2, 4, 1, 3, 5, 6).contiguous().view(-1, D, window_size[0], window_size[1], C)
    return windows

def window_reverse(windows, window_size, B, D, H, W):
    """Reverse window partition.

    Args:
        windows (torch.Tensor): (B*nW, D, Wh, Ww, C)
        window_size (tuple[int]): Window size
        B (int): Number of batches
        D (int): Number of frames
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, D, H, W, C)
    """
    x = windows.view(B, H // window_size[0], W // window_size[1], D, window_size[0], window_size[1], -1)
    x = x.permute(0, 3, 1, 4, 2, 5, 6).contiguous().view(B, D, H, W, -1)
    return x

def get_window_size(x_size, window_size, shift_size=None):
    """Adjust window size and shift size based on the size of the input.

    Args:
        x_size (tuple[int]): The shape of x.
        window_size (tuple[int]): Window size.
        shift_size (tuple[int], optional): Shift size. Defaults to None.

    Returns:
        use_window_size: Window size for use.
        use_shift_size: Shift size for use.
    """
    use_window_size = list(window_size)
    if shift_size is not None:
        use_shift_size = list(shift_size)
    for i in range(len(x_size)):
        if x_size[i] <= window_size[i]:
            use_window_size[i] = x_size[i]
            if shift_size is not None:
                use_shift_size[i] = 0

    if shift_size is None:
        return tuple(use_window_size)
    else:
        return tuple(use_window_size), tuple(use_shift_size)

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class WindowAttention3D(nn.Module):
    """Window based multi-head self/cross attention (W-MSA/W-MCA) module with relative 
    position bias. 
    It supports both of shifted and non-shifted window.
    """
    def __init__(self, dim, num_frames_q, num_frames_kv, window_size, num_heads, 
                 qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        """Initialization function.

        Args:
            dim (int): Number of input channels.
            num_frames (int): Number of input frames.
            window_size (tuple[int]): The size of the window.
            num_heads (int): Number of attention heads.
            qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Defaults to True.
            qk_scale (float, optional): Override default qk scale of head_dim ** -0.5 if set. Defaults to None.
            attn_drop (float, optional): Dropout ratio of attention weight. Defaults to 0.0
            proj_drop (float, optional): Dropout ratio of output. Defaults to 0.0
        """
        super().__init__()
        self.dim = dim
        self.num_frames_q = num_frames_q # D1
        self.num_frames_kv = num_frames_kv # D2
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads # nH
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * num_frames_q - 1) * (2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )  # 2*D-1 * 2*Wh-1 * 2*Ww-1, nH

        # Get pair-wise relative position index for each token inside the window
        coords_d_q = torch.arange(self.num_frames_q)
        coords_d_kv = torch.arange(0, self.num_frames_q, int((self.num_frames_q + 1) // self.num_frames_kv))
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords_q = torch.stack(torch.meshgrid([coords_d_q, coords_h, coords_w]))  # 3, D1, Wh, Ww
        coords_kv = torch.stack(torch.meshgrid([coords_d_kv, coords_h, coords_w]))  # 3, D2, Wh, Ww
        coords_q_flatten = torch.flatten(coords_q, 1)  # 3, D1*Wh*Ww
        coords_kv_flatten = torch.flatten(coords_kv, 1)  # 3, D2*Wh*Ww
        relative_coords = coords_q_flatten[:, :, None] - coords_kv_flatten[:, None, :]  # 3, D1*Wh*Ww, D2*Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # D1*Wh*Ww, D2*Wh*Ww, 3
        relative_coords[:, :, 0] += self.num_frames_q - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[0] - 1
        relative_coords[:, :, 2] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= (2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1)
        relative_coords[:, :, 1] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # D1*Wh*Ww, D2*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, kv=None, mask=None):
        """Forward function.

        Args:
            q (torch.Tensor): (B*nW, D1*Wh*Ww, C)
            kv (torch.Tensor): (B*nW, D2*Wh*Ww, C). Defaults to None.
            mask (torch.Tensor, optional): Mask for shifted window attention (nW, D1*Wh*Ww, D2*Wh*Ww). Defaults to None.

        Returns:
            torch.Tensor: (B*nW, D1*Wh*Ww, C)
        """
        kv = q if kv is None else kv
        B_, N1, C = q.shape # N1 = D1*Wh*Ww, B_ = B*nW
        B_, N2, C = kv.shape # N2 = D2*Wh*Ww, B_ = B*nW

        q = self.q(q).reshape(B_, N1, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        kv = self.kv(kv).reshape(B_, N2, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = q[0], kv[0], kv[1] # B_, nH, N1(2), C
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1)) # B_, nH, N1, N2

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            N1, N2, -1)  # D1*Wh*Ww, D2*Wh*Ww, nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, D1*Wh*Ww, D2*Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0) # B_, nH, D1*Wh*Ww, D2*Wh*Ww

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N1, N2) + mask.unsqueeze(1).unsqueeze(0) # B, nW, nH, D1*Wh*Ww, D2*Wh*Ww
            attn = attn.view(-1, self.num_heads, N1, N2)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

class VSTSREncoderTransformerBlock(nn.Module):
    """Video spatial-temporal super-resolution encoder transformer block.
    """
    def __init__(self, dim, num_heads, num_frames=4, window_size=(8, 8), 
                 shift_size=(0, 0), mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm):
        """Initialization function.

        Args:
            dim (int): Number of input channels. 
            num_heads (int): Number of attention heads.
            num_frames (int): Number of input frames.
            window_size (tuple[int], optional): Window size. Defaults to 8.
            shift_size (tuple[int], optional): Shift size. Defaults to 0.
            mlp_ratio (int, optional): Ratio of mlp hidden dim to embedding dim. Defaults to 4.
            qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Defaults to True.
            qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set. Defaults to None.
            drop (float, optional): Dropout rate. Defaults to 0.
            attn_drop (float, optional): Attention dropout rate. Defaults to 0.
            drop_path (float, optional):  Stochastic depth rate. Defaults to 0.
            act_layer (nn.Module, optional): Activation layer. Defaults to nn.GELU.
            norm_layer (nn.Module, optional): Normalization layer. Defaults to nn.LayerNorm.
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_frames = num_frames
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-win_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-win_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention3D(
            dim, num_frames_q=self.num_frames, num_frames_kv=self.num_frames,
            window_size=self.window_size, num_heads=num_heads, 
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, 
            proj_drop=drop
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
       

    def forward(self, x, mask_matrix):
        """Forward function.

        Args:
            x (torch.Tensor): (B, D, H, W, C)
            mask_matrix (torch.Tensor): (nW*B, D*Wh*Ww, D*Wh*Ww)

        Returns:
            torch.Tensor: (B, D, H, W, C)
        """
        B, D, H, W, C = x.shape
        window_size, shift_size = get_window_size((H, W), self.window_size, self.shift_size)

        shortcut = x
        x = self.norm1(x)

        # Padding
        pad_b = (window_size[0] - H % window_size[0]) % window_size[0]
        pad_r = (window_size[1] - W % window_size[1]) % window_size[1]
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b, 0, 0))
        _, _, Hp, Wp, _ = x.shape

        # cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(2, 3))
            attn_mask = mask_matrix
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, window_size)  # nW*B, D, window_size, window_size, C
        x_windows = x_windows.view(-1, D * window_size[0] * window_size[1], C)  # nW*B, D*window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)[0]  # nW*B, D*window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, D, window_size[0], window_size[1], C)
        shifted_x = window_reverse(attn_windows, window_size, B, D, Hp, Wp)  # B, D, H, W, C

        # reverse cyclic shift
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1]), dims=(2, 3))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :, :H, :W, :].contiguous()

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class VSTSRDecoderTransformerBlock(nn.Module):
    """Video spatial-temporal super-resolution decoder transformer block.
    """
    def __init__(self, dim, num_heads, num_frames=4, window_size=(8, 8), 
                 shift_size=(0, 0), mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm):
        """Initialization function.

        Args:
            dim (int): Number of input channels. 
            num_heads (int): Number of attention heads.
            num_frames (int): Number of input frames.
            window_size (tuple[int], optional): Window size. Defaults to 8.
            shift_size (tuple[int], optional): Shift size. Defaults to 0.
            mlp_ratio (int, optional): Ratio of mlp hidden dim to embedding dim. Defaults to 4.
            qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Defaults to True.
            qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set. Defaults to None.
            drop (float, optional): Dropout rate. Defaults to 0.
            attn_drop (float, optional): Attention dropout rate. Defaults to 0.
            drop_path (float, optional):  Stochastic depth rate. Defaults to 0.
            act_layer (nn.Module, optional): Activation layer. Defaults to nn.GELU.
            norm_layer (nn.Module, optional): Normalization layer. Defaults to nn.LayerNorm.
        """
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_frames = num_frames
        self.num_out_frames = num_frames
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size[0] < self.window_size[0], "shift_size must in 0-win_size"
        assert 0 <= self.shift_size[1] < self.window_size[1], "shift_size must in 0-win_size"

        self.norm1 = norm_layer(dim)
        self.attn1 = WindowAttention3D(
            dim, num_frames_q=self.num_out_frames, 
            num_frames_kv=self.num_out_frames, window_size=self.window_size, 
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, 
            attn_drop=attn_drop, proj_drop=drop
        )
        self.attn2 = WindowAttention3D(
            dim, num_frames_q=self.num_out_frames, 
            num_frames_kv=self.num_frames, window_size=self.window_size, 
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, 
            attn_drop=attn_drop, proj_drop=drop
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.norm_kv = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
       

    def forward(self, x, attn_kv, mask_matrix_q, mask_matrix_qkv):
        """Forward function.

        Args:
            x (torch.Tensor): (B, D1, H, W, C)
            attn_kv (torch.Tensor): (B, D2, H, W, C)
            mask_matrix_q (torch.Tensor): (nW*B, D1*Wh*Ww, D1*Wh*Ww)
            mask_matrix_qkv (torch.Tensor): (nW*B, D1*Wh*Ww, D2*Wh*Ww)

        Returns:
            torch.Tensor: (B, D1, H, W, C)
        """
        B, D1, H, W, C = x.shape
        B, D2, H, W, C = attn_kv.shape
        window_size, shift_size = get_window_size((H, W), self.window_size, self.shift_size)

        shortcut = x
        x = self.norm1(x)

        # Padding
        pad_b = (window_size[0] - H % window_size[0]) % window_size[0]
        pad_r = (window_size[1] - W % window_size[1]) % window_size[1]
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b, 0, 0))
        _, _, Hp, Wp, _ = x.shape

        # cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(2, 3))
            attn_mask_q = mask_matrix_q
            attn_mask_qkv = mask_matrix_qkv
        else:
            shifted_x = x
            attn_mask_q = None
            attn_mask_qkv = None

        # partition windows
        x_windows = window_partition(shifted_x, window_size)  # nW*B, D1, window_size, window_size, C
        x_windows = x_windows.view(-1, D1 * window_size[0] * window_size[1], C)  # nW*B, D1*window_size*window_size, C

        # W-MSA/SW-MSA for query
        attn_windows = self.attn1(x_windows, mask=attn_mask_q)[0] # nW*B, D1*window_size*window_size, C
        attn_windows = attn_windows.view(-1, D1, window_size[0], window_size[1], C)
        shifted_x = window_reverse(attn_windows, window_size, B, D1, Hp, Wp) # B, D1, Hp, Wp, C

        # reverse cyclic shift
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1]), dims=(2, 3))
        else:
            x = shifted_x
        
        if pad_r > 0 or pad_b > 0:
            x = x[:, :, :H, :W, :].contiguous()

        x = shortcut + self.drop_path(x)

        shortcut = x
        x = self.norm2(x)
        attn_kv = self.norm_kv(attn_kv)
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b, 0, 0))
        attn_kv = F.pad(attn_kv, (0, 0, 0, pad_r, 0, pad_b, 0, 0))

        # cyclic shift
        if any(i > 0 for i in shift_size):
            shifted_x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(2, 3))
            shifted_attn_kv = torch.roll(attn_kv, shifts=(-shift_size[0], -shift_size[1]), dims=(2, 3))
            attn_mask_q = mask_matrix_q
            attn_mask_qkv = mask_matrix_qkv
        else:
            shifted_x = x
            shifted_attn_kv = attn_kv
            attn_mask_q = None
            attn_mask_qkv = None

        # partition windows
        x_windows = window_partition(shifted_x, window_size)  # nW*B, D1, window_size, window_size, C
        attn_kv_windows = window_partition(shifted_attn_kv, window_size)  # nW*B, D2, window_size, window_size, C
        x_windows = x_windows.view(-1, D1 * window_size[0] * window_size[1], C)  # nW*B, D1*window_size*window_size, C
        attn_kv_windows = attn_kv_windows.view(-1, D2 * window_size[0] * window_size[1], C)  # nW*B, D2*window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn2(x_windows, attn_kv_windows, mask=attn_mask_qkv)[0]  # nW*B, D1*window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, D1, window_size[0], window_size[1], C)
        shifted_x = window_reverse(attn_windows, window_size, B, D1, Hp, Wp)  # B, D1, H, W, C

        # reverse cyclic shift
        if any(i > 0 for i in shift_size):
            x = torch.roll(shifted_x, shifts=(shift_size[0], shift_size[1]), dims=(2, 3))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :, :H, :W, :].contiguous()

        x = shortcut + self.drop_path(x)

        # FFN
        x = x + self.drop_path(self.mlp(self.norm3(x)))

        return x

class EncoderLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, num_frames, window_size=(8, 8), 
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., 
                 drop_path=0., norm_layer=nn.LayerNorm):
        """Encoder layer

        Args:
            dim (int): Number of feature channels
            depth (int): Depths of this stage.
            num_heads (int): Number of attention head.
            num_frames (int]): Number of input frames.
            window_size (tuple[int], optional): Window size. Defaults to (8, 8).
            mlp_ratio (int, optional): Ratio of mlp hidden dim to embedding dim. Defaults to 4.
            qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Defaults to True.
            qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set. Defaults to None.
            drop (float, optional): Dropout rate. Defaults to 0.
            attn_drop (float, optional): Attention dropout rate. Defaults to 0.
            drop_path (float, optional): Stochastic depth rate. Defaults to 0.
            norm_layer (nn.Module, optional): Normalization layer. Defaults to nn.LayerNorm.
        """
        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.depth = depth

        # Build blocks
        self.blocks = nn.ModuleList([
            VSTSREncoderTransformerBlock(dim=dim, num_heads=num_heads,
            num_frames=num_frames,window_size=window_size, 
            shift_size=(0, 0) if (i % 2 == 0) else self.shift_size,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop, attn_drop=attn_drop,
            drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
            norm_layer=norm_layer)
        for i in range(depth)])

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): (B, D, C, H, W)

        Returns:
            torch.Tensor: (B, D, C, H, W)
        """
        B, D, C, H, W = x.shape
        x = x.permute(0, 1, 3, 4, 2) # B, D, H, W, C

        window_size, shift_size = get_window_size((H, W), self.window_size, self.shift_size)

        Hp = int(np.ceil(H / window_size[0])) * window_size[0]
        Wp = int(np.ceil(W / window_size[1])) * window_size[1]

        img_mask = torch.zeros((1, D, Hp, Wp, 1), device=x.device) # 1, D, H, W, 1
        h_slices = (slice(0, -window_size[0]),
                    slice(-window_size[0], -shift_size[0]),
                    slice(-shift_size[0], None))
        w_slices = (slice(0, -window_size[1]),
                    slice(-window_size[1], -shift_size[1]),
                    slice(-shift_size[1], None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, :, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, window_size) # nW, D, Wh, Ww, 1
        mask_windows = mask_windows.view(-1, D * window_size[0] * window_size[1]) # nW, D*Wh*Ww
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2) # nW, D*Wh*Ww, D*Wh*Ww
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        for blk in self.blocks:
            x = blk(x, attn_mask)

        x = x.permute(0, 1, 4, 2, 3) # B, D, C, H, W

        return x

class DecoderLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, num_frames, window_size=(8, 8), 
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., 
                 drop_path=0., norm_layer=nn.LayerNorm):
        """Decoder layer

        Args:
            dim (int): Number of feature channels
            depth (int): Depths of this stage.
            num_heads (int): Number of attention head.
            num_frames (int]): Number of input frames.
            window_size (tuple[int], optional): Window size. Defaults to (8, 8).
            mlp_ratio (int, optional): Ratio of mlp hidden dim to embedding dim. Defaults to 4.
            qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Defaults to True.
            qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set. Defaults to None.
            drop (float, optional): Dropout rate. Defaults to 0.
            attn_drop (float, optional): Attention dropout rate. Defaults to 0.
            drop_path (float, optional): Stochastic depth rate. Defaults to 0.
            norm_layer (nn.Module, optional): Normalization layer. Defaults to nn.LayerNorm.
        """
        super().__init__()
        self.window_size = window_size
        self.shift_size = tuple(i // 2 for i in window_size)
        self.depth = depth

        # Build blocks
        self.blocks = nn.ModuleList([
            VSTSRDecoderTransformerBlock(dim=dim, num_heads=num_heads,
            num_frames=num_frames,window_size=window_size, 
            shift_size=(0, 0) if (i % 2 == 0) else self.shift_size,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop, attn_drop=attn_drop,
            drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
            norm_layer=norm_layer)
        for i in range(depth)])

    def forward(self, x, attn_kv):
        """Forward function.

        Args:
            x (torch.Tensor): (B, D1, C, H, W)
            attn_kv (torch.Tensor): (B, D2, C, H, W)

        Returns:
            torch.Tensor: (B, D1, C, H, W)
        """
        B, D1, C, H, W = x. shape
        _, D2, C, _, _ = attn_kv.shape
        x = x.permute(0, 1, 3, 4, 2) # B, D1, H, W, C
        attn_kv = attn_kv.permute(0, 1, 3, 4, 2) # B, D2, H, W, C

        window_size, shift_size = get_window_size((H, W), self.window_size, self.shift_size)

        Hp = int(np.ceil(H / window_size[0])) * window_size[0]
        Wp = int(np.ceil(W / window_size[1])) * window_size[1]

        img_mask_q = torch.zeros((1, D1, Hp, Wp, 1), device=x.device) # 1, D1, H, W, 1
        img_mask_kv = torch.zeros((1, D2, Hp, Wp, 1), device=x.device) # 1, D2, H, W, 1
        h_slices = (slice(0, -window_size[0]),
                    slice(-window_size[0], -shift_size[0]),
                    slice(-shift_size[0], None))
        w_slices = (slice(0, -window_size[1]),
                    slice(-window_size[1], -shift_size[1]),
                    slice(-shift_size[1], None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask_q[:, :, h, w, :] = cnt
                img_mask_kv[:, :, h, w, :] = cnt
                cnt += 1

        mask_windows_q = window_partition(img_mask_q, window_size) # nW, D1, Wh, Ww, 1
        mask_windows_kv = window_partition(img_mask_kv, window_size) # nW, D2, Wh, Ww, 1
        mask_windows_q = mask_windows_q.view(-1, D1 * window_size[0] * window_size[1]) # nW, D1*Wh*Ww
        mask_windows_kv = mask_windows_kv.view(-1, D2 * window_size[0] * window_size[1]) # nW, D2*Wh*Ww
        attn_mask_q = mask_windows_q.unsqueeze(1) - mask_windows_q.unsqueeze(2) # nW, D1*Wh*Ww, D1*Wh*Ww
        attn_mask_qkv = mask_windows_kv.unsqueeze(1) - mask_windows_q.unsqueeze(2) # nW, D1*Wh*Ww, D2*Wh*Ww
        attn_mask_q = attn_mask_q.masked_fill(attn_mask_q != 0, float(-100.0)).masked_fill(attn_mask_q == 0, float(0.0))
        attn_mask_qkv = attn_mask_qkv.masked_fill(attn_mask_qkv != 0, float(-100.0)).masked_fill(attn_mask_qkv == 0, float(0.0))

        for blk in self.blocks:
            x = blk(x, attn_kv, attn_mask_q, attn_mask_qkv)

        x = x.permute(0, 1, 4, 2, 3) # B, D, C, H, W

        return x


class InputProj(nn.Module):
    """Video input projection

    Args:
        in_channels (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of output channels. Default: 32.
        kernel_size (int): Size of the convolution kernel. Default: 3
        stride (int): Stride of the convolution. Default: 1
        norm_layer (nn.Module, optional): Normalization layer. Default: None.
        act_layer (nn.Module): Activation layer. Default: nn.LeakyReLU.
    """
    def __init__(self, in_channels=3, embed_dim=32, kernel_size=3, stride=1, 
                 norm_layer=None, act_layer=nn.LeakyReLU):
        super().__init__()

        self.embed_dim = embed_dim

        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=kernel_size, 
                      stride=stride, padding=kernel_size//2),
            act_layer(inplace=True)
        )

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """

        Args:
            x (torch.Tensor): (B, D, C, H, W)

        Returns:
            torch.Tensor: (B, D, C, H, W)
        """
        B, D, C, H, W = x.shape
        x = x.view(-1, C, H, W) # B*D, C, H, W
        x = self.proj(x).view(B, D, -1, H, W) # B, D, C, H, W
        if self.norm is not None:
            x = x.permute(0, 1, 3, 4, 2) # B, D, H, W, C
            x = self.norm(x)
            x = x.permute(0, 1, 4, 2, 3) # B, D, C, H, W
        return x

class Downsample(nn.Module):
    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): (B, D, C, H, W)

        Returns:
            torch.Tensor: (B, D, C, H, W)
        """
        B, D, C, H, W = x.shape
        x = x.view(-1, C, H, W) # B*D, C, H, W
        out = self.conv(x).view(B, D, -1, H // 2, W // 2)  # B, D, C, H, W
        return out

class Upsample(nn.Module):
    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(in_chans, out_chans, kernel_size=2, stride=2),
        )

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): (B, D, C, H, W)

        Returns:
            torch.Tensor: (B, D, C, H, W)
        """
        B, D, C, H, W = x. shape
        x = x.view(-1, C, H, W) # B*D, C, H, W
        out = self.deconv(x).view(B, D, -1, H * 2, W * 2) # B, D, C, H, W
        return out


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
def nonlinearity(x):
    # swish
    return F.silu(x, inplace=True)  # x*torch.sigmoid(x)


from torch.utils.checkpoint import checkpoint

class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.checkpointing = False

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout, inplace=True)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def _forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h

    def forward(self, x, temb):
        if self.checkpointing and self.training:
            out = checkpoint(self._forward, x, temb)
        else:
            out = self._forward(x, temb)
        return out




class TDResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.checkpointing = False

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout, inplace=True)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def _forward(self, x, temb=None):
        ts = False
        if len(x.shape)==5:
            ts = True
            B,D,C,H,W = x.shape
            input = x.view(B*D,C,H,W)
        else:
            input = x
        h = input
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(input)
            else:
                x = self.nin_shortcut(input)
        _,CO,_,_ = h.shape
        if ts:
            h = h.view(B,D,CO,H,W)
        return x+h

    def forward(self, x, temb):
        if self.checkpointing and self.training:
            out = checkpoint(self._forward, x, temb)
        else:
            out = self._forward(x, temb)
        return out



class SResBlock(nn.Module):
    def __init__(self,num_res_blocks,in_channels, out_channels=None, conv_shortcut=False,
                 dropout=0, temb_channels=512):
        super().__init__()
        block = []
        for i in range(num_res_blocks):
            block.append(ResnetBlock(in_channels=in_channels,
                                        out_channels=out_channels,
                                        temb_channels=0,
                                        dropout=dropout))
            in_channels = out_channels
        self.mid = nn.Sequential(block)
        
        
    def forward(self, input):
        B, D, C, H, W = input. shape
        x = input.view(B*D,C,H,W)
        
        out = self.mid(x)
        _,CO,H,W = out.shape
        
        out = out.view(B, D, CO, H, W)
        return out
        