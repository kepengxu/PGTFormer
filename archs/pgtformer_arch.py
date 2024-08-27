import math
import numpy as np
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Optional, List
import sys
import os
current_working_dir = os.getcwd()
sys.path.append(current_working_dir+'/')

from archs.codeformer_arch import TransformerSALayer,adaptive_instance_normalization
from archs.rqvae_arch import RQVAE, nonlinearity
from archs.tdcrqvae3_arch import TDCRQVAE3,TDResnetBlock
from basicsr.utils import get_root_logger
from basicsr.utils.registry import ARCH_REGISTRY



import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as modelzoo


import torchvision.transforms as transforms
import torchvision

# from modules.bn import InPlaceABNSync as BatchNorm2d

resnet18_url = 'https://download.pytorch.org/models/resnet18-5c106cde.pth'


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, in_chan, out_chan, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_chan, out_chan, stride)
        self.bn1 = nn.BatchNorm2d(out_chan)
        self.conv2 = conv3x3(out_chan, out_chan)
        self.bn2 = nn.BatchNorm2d(out_chan)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if in_chan != out_chan or stride != 1:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_chan, out_chan,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_chan),
                )

    def forward(self, x):
        residual = self.conv1(x)
        residual = F.relu(self.bn1(residual))
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        shortcut = x
        if self.downsample is not None:
            shortcut = self.downsample(x)

        out = shortcut + residual
        out = self.relu(out)
        return out


def create_layer_basic(in_chan, out_chan, bnum, stride=1):
    layers = [BasicBlock(in_chan, out_chan, stride=stride)]
    for i in range(bnum-1):
        layers.append(BasicBlock(out_chan, out_chan, stride=1))
    return nn.Sequential(*layers)


class Resnet18(nn.Module):
    def __init__(self):
        super(Resnet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = create_layer_basic(64, 64, bnum=2, stride=1)
        self.layer2 = create_layer_basic(64, 128, bnum=2, stride=2)
        self.layer3 = create_layer_basic(128, 256, bnum=2, stride=2)
        self.layer4 = create_layer_basic(256, 512, bnum=2, stride=2)
        # self.init_weight()

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.maxpool(x)

        x = self.layer1(x)
        feat8 = self.layer2(x) # 1/8
        feat16 = self.layer3(feat8) # 1/16
        feat32 = self.layer4(feat16) # 1/32
        return feat8, feat16, feat32

    def init_weight(self):
        state_dict = modelzoo.load_url(resnet18_url)
        self_state_dict = self.state_dict()
        for k, v in state_dict.items():
            if 'fc' in k: continue
            self_state_dict.update({k: v})
        self.load_state_dict(self_state_dict)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module,  nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


# if __name__ == "__main__":
#     net = Resnet18()
#     x = torch.randn(16, 3, 224, 224)
#     out = net(x)
#     print(out[0].size())
#     print(out[1].size())
#     print(out[2].size())
#     net.get_params()




# from resnet import Resnet18
# from modules.bn import InPlaceABNSync as BatchNorm2d


class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                out_chan,
                kernel_size = ks,
                stride = stride,
                padding = padding,
                bias = False)
        self.bn = nn.BatchNorm2d(out_chan)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(self.bn(x))
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

class BiSeNetOutput(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes, *args, **kwargs):
        super(BiSeNetOutput, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(AttentionRefinementModule, self).__init__()
        self.conv = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size= 1, bias=False)
        self.bn_atten = nn.BatchNorm2d(out_chan)
        self.sigmoid_atten = nn.Sigmoid()
        self.init_weight()

    def forward(self, x):
        feat = self.conv(x)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        out = torch.mul(feat, atten)
        return out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class ContextPath(nn.Module):
    def __init__(self, *args, **kwargs):
        super(ContextPath, self).__init__()
        self.resnet = Resnet18()
        self.arm16 = AttentionRefinementModule(256, 128)
        self.arm32 = AttentionRefinementModule(512, 128)
        self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_avg = ConvBNReLU(512, 128, ks=1, stride=1, padding=0)

        self.init_weight()

    def forward(self, x):
        H0, W0 = x.size()[2:]
        feat8, feat16, feat32 = self.resnet(x)
        H8, W8 = feat8.size()[2:]
        H16, W16 = feat16.size()[2:]
        H32, W32 = feat32.size()[2:]

        avg = F.avg_pool2d(feat32, feat32.size()[2:])
        avg = self.conv_avg(avg)
        avg_up = F.interpolate(avg, (H32, W32), mode='nearest')

        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm + avg_up
        feat32_up = F.interpolate(feat32_sum, (H16, W16), mode='nearest')
        feat32_up = self.conv_head32(feat32_up)

        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = F.interpolate(feat16_sum, (H8, W8), mode='nearest')
        feat16_up = self.conv_head16(feat16_up)

        return feat8, feat16_up, feat32_up  # x8, x8, x16

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


### This is not used, since I replace this with the resnet feature with the same size
class SpatialPath(nn.Module):
    def __init__(self, *args, **kwargs):
        super(SpatialPath, self).__init__()
        self.conv1 = ConvBNReLU(3, 64, ks=7, stride=2, padding=3)
        self.conv2 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1)
        self.conv3 = ConvBNReLU(64, 64, ks=3, stride=2, padding=1)
        self.conv_out = ConvBNReLU(64, 128, ks=1, stride=1, padding=0)
        self.init_weight()

    def forward(self, x):
        feat = self.conv1(x)
        feat = self.conv2(feat)
        feat = self.conv3(feat)
        feat = self.conv_out(feat)
        return feat

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(out_chan,
                out_chan//4,
                kernel_size = 1,
                stride = 1,
                padding = 0,
                bias = False)
        self.conv2 = nn.Conv2d(out_chan//4,
                out_chan,
                kernel_size = 1,
                stride = 1,
                padding = 0,
                bias = False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.init_weight()

    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv1(atten)
        atten = self.relu(atten)
        atten = self.conv2(atten)
        atten = self.sigmoid(atten)
        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class BiSeNet(nn.Module):
    def __init__(self, n_classes, *args, **kwargs):
        super(BiSeNet, self).__init__()
        self.cp = ContextPath()
        ## here self.sp is deleted
        self.ffm = FeatureFusionModule(256, 256)
        self.conv_out = BiSeNetOutput(256, 256, n_classes)
        self.conv_out16 = BiSeNetOutput(128, 64, n_classes)
        self.conv_out32 = BiSeNetOutput(128, 64, n_classes)
        self.init_weight()

    def forward(self, x):
        H, W = x.size()[2:]
        feat_res8, feat_cp8, feat_cp16 = self.cp(x)  # here return res3b1 feature
        feat_sp = feat_res8  # use res3b1 feature to replace spatial path feature
        feat_fuse = self.ffm(feat_sp, feat_cp8)

        feat_out = self.conv_out(feat_fuse)
        feat_out16 = self.conv_out16(feat_cp8)
        feat_out32 = self.conv_out32(feat_cp16) 

        feat_out = F.interpolate(feat_out, (32, 32), mode='bilinear', align_corners=True)
        feat_out16 = F.interpolate(feat_out16, (32, 32), mode='bilinear', align_corners=True)
        # feat_out32 = F.interpolate(feat_out32, (32, 32), mode='bilinear', align_corners=True)
        outf = torch.cat([feat_out,feat_out16,feat_out32],1)
        return outf

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = [], [], [], []
        for name, child in self.named_children():
            child_wd_params, child_nowd_params = child.get_params()
            if isinstance(child, FeatureFusionModule) or isinstance(child, BiSeNetOutput):
                lr_mul_wd_params += child_wd_params
                lr_mul_nowd_params += child_nowd_params
            else:
                wd_params += child_wd_params
                nowd_params += child_nowd_params
        return wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params




@torch.jit.script
def swish(x):
    return x*torch.sigmoid(x)

def normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super(ResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.norm1 = normalize(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = normalize(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            self.conv_out = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x_in):
        x = x_in
        x = self.norm1(x)
        x = swish(x)
        x = self.conv1(x)
        x = self.norm2(x)
        x = swish(x)
        x = self.conv2(x)
        if self.in_channels != self.out_channels:
            x_in = self.conv_out(x_in)
        out = x + x_in
        return out


class Fuse_sft_block(nn.Module):
    def __init__(self, in_ch, out_ch,t=3):
        super().__init__()
        self.tcc = 32
        self.encode_enc = ResBlock(2*in_ch+ self.tcc, out_ch)


        self.scale = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1))

        self.shift = nn.Sequential(
                    nn.Conv2d(in_ch , out_ch, kernel_size=3, padding=1),
                    nn.LeakyReLU(0.2, True),
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1))
        self.t = t
        # last_zero_init(self.scale)
        # last_zero_init(self.shift)
        self.tconvenc = nn.Conv2d(in_ch,self.tcc,1)
        self.tconvdec = nn.Conv2d(in_ch,self.tcc,1)
        
        self.tfusion0 = nn.Conv2d(2*t*self.tcc,self.tcc*self.t,1)
        self.tfusion1 = nn.Conv2d(self.tcc,self.tcc,1)

    def forward(self, enc_feat, dec_feat,temb=None, w=1):
        
        
        b,d,c,h,wf = enc_feat.shape
        enc_feat = enc_feat.view(b*d,c,h,wf)
        dec_feat = dec_feat.view(b*d,c,h,wf)
        
        enct = self.tconvenc(enc_feat).view(b,d,self.tcc,h,wf).contiguous().view(b,d*self.tcc,h,wf)
        dect = self.tconvdec(dec_feat).view(b,d,self.tcc,h,wf).contiguous().view(b,d*self.tcc,h,wf)
        
        fut = torch.cat([enct,dect],1)
        fut = self.tfusion0(fut).view(b,d,self.tcc,h,wf).contiguous().view(b*d,self.tcc,h,wf)
        fut = self.tfusion1(fut)
        
        
        enc_feat = self.encode_enc(torch.cat([enc_feat, dec_feat, fut], dim=1))
        scale = self.scale(enc_feat)
        shift = self.shift(enc_feat) 
        residual = w * (dec_feat * scale + shift)
        out = dec_feat + residual
        # out = dec_feat +w * dec_feat * scale + w * shift)
        # out = (w*scale +1 ) * dec_feat + w * shift
        
        _,cc,_,_ = out.shape
        return out.view(b,d,cc,h,wf)




# @ARCH_REGISTRY.register()
class PGTFormer(TDCRQVAE3):
    def __init__(self, ddconfig, dim_embd=512, n_head=8, n_layers=9, 
                connect_list=['32', '64', '128', '256'],
                fix_modules=['quantizer','decoder','conditionnet'],
                w=0, detach_16=True, adain=False,tf =3,droprate=0.0, **kwargs
                ):
        super(PGTFormer, self).__init__(ddconfig=ddconfig, **kwargs)
        self.fix_modules = fix_modules
        self.t = tf
        self.w = w
        self.detach_16 = detach_16
        self.adain = adain

        self.connect_list = connect_list
        self.n_layers = n_layers
        self.dim_embd = dim_embd
        self.dim_mlp = dim_embd*2

        # self.position_emb = nn.Parameter(torch.zeros(1024, self.dim_embd))
        
        
        
        self.conditionnet = BiSeNet(19)
        # kepeng xu TODO
        # state = torch.load('weights/facelib/faceparse/weights/79999.pth',map_location='cpu')
        # self.conditionnet.load_state_dict(state,strict=False)
        # self.conditionnet.eval()
        
        
        
        self.convpos = nn.Conv2d(57,512,1)

        self.feat_emb = nn.Linear(512, self.dim_embd)

        # transformer
        self.ft_layers = nn.Sequential(*[TransformerSALayer(embed_dim=dim_embd, nhead=n_head, dim_mlp=self.dim_mlp, dropout=droprate) 
                                    for _ in range(self.n_layers)])

        # logits_predict head
        self.codebook_size = self.quantizer.n_embed[-1]
        self.quantizer_depth = self.quantizer.code_shape[-1]
        self.idx_pred_layer = nn.Sequential(
            nn.LayerNorm(dim_embd),
            nn.Linear(dim_embd, self.quantizer_depth*self.codebook_size, bias=False))

        self.channels = {
            '16': 512,
            '32': 512,
            '64': 256,
            '128': 256,
            '256': 128,
            '512': 64,
        }

        self.fuse_encoder_indices = {'512':0, '256':1, '128':2, '64':3, '32':4, '16':5}

        # fuse_convs_dict
        self.fuse_convs_dict = nn.ModuleDict()
        for f_size in self.connect_list:
            in_ch = self.channels[f_size]
            self.fuse_convs_dict[f_size] = Fuse_sft_block(in_ch, in_ch)

        self.extra_encoder_modules = [self.ft_layers, self.feat_emb, self.quant_conv, self.idx_pred_layer]
        self._freeze_modules(self.fix_modules)
        self.to_tensor = transforms.Compose([
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def _freeze_modules(self, fix_modules):
        if fix_modules is not None:
            for module in fix_modules:
                for param in getattr(self, module).parameters():
                    param.requires_grad = False
                getattr(self, module).eval()

            if 'decoder' in self.fix_modules:
                for param in self.post_quant_conv.parameters():
                    param.requires_grad = False
                self.post_quant_conv.eval()

            if 'encoder' in self.fix_modules:
                for module in self.extra_encoder_modules:
                    for param in module.parameters():
                        param.requires_grad = False
                    module.eval()
                # self.position_emb.requires_grad = False # make pytorch happy

    def train(self, mode=True):
        """Convert the model into training mode while keep some modules
        freezed (fixed)."""
        super(PGTFormer, self).train(mode)
        self._freeze_modules(self.fix_modules)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def get_last_layer(self):
        if 'decoder' not in self.fix_modules:
            return self.decoder.conv_out.weight
        else:
            return self.fuse_convs_dict[self.connect_list[-1]].encode_enc.conv2.weight

    def forward(self, x, w=None, detach_16=True, code_only=None, adain=None):
        if w is None:
            w = self.w
        if detach_16 is None:
            detach_16 = self.detach_16
        if adain is None:
            adain = self.adain
            
        nx = self.to_tensor(x)
        conditionfeature = self.conditionnet(nx)
        # print(conditionfeature.shape)
        conditionfeature = self.convpos(conditionfeature)
        tb,tc,th,tw = conditionfeature.shape
        b = tb//self.t
        # conditionfeature = conditionfeature.view(b,self.t,tc,th*tw).per   #.permute(2,0,1)

        conditionfeature = conditionfeature.view(b,self.t,tc,th,tw).permute(0,2,1,3,4).contiguous().view(b,tc,self.t*th*tw).permute(2,0,1)

        bt,c,h,wf = x.shape

        b = bt//self.t
        x = x.view(b,self.t,c,h,wf)
        
        # ################### Encoder #####################



        enc_feat_dict = {}
        x, multi_res_feats = self.encoder(x, return_multi_res_feats=True)
        for f_size in self.connect_list:
            idx = self.fuse_encoder_indices[f_size]
            feat = multi_res_feats[idx]
            enc_feat_dict[str(feat.shape[-1])] = feat.clone()
        x = self.quant_conv(x)

        lq_feat = x
        # ################# Transformer ###################
        # quant_feat, codebook_loss, quant_stats = self.quantize(lq_feat)
        # pos_emb = self.position_emb.unsqueeze(1).repeat(1,x.shape[0],1)
        # BCHW -> BC(HW) -> (HW)BC
        feat_emb = self.feat_emb(lq_feat.flatten(2).permute(2,0,1))
        _,_,cc = feat_emb.shape
        query_emb = feat_emb.view(th*tw,b,self.t,cc).permute(2,0,1,3).contiguous().view(self.t*th*tw,b,cc)
        # Transformer encoder
        for layer in self.ft_layers:
            query_emb = layer(query_emb, query_pos=conditionfeature)

        # output logits
        logits = self.idx_pred_layer(query_emb.view(self.t,th*tw,b,cc).permute(1,2,0,3).contiguous().view(th*tw,b*self.t,cc)) # [3072, 1, 512]
        batch_size = logits.shape[1]
        # (hw)b(dn) -> bhwdn
        logits = logits.transpose(0,1).reshape(batch_size, *self.quantizer.code_shape, self.codebook_size)
        # tttt = lq_feat.permute(0,2,3,1)
        if code_only: # for training stage II
          # logits doesn't need softmax before cross_entropy loss
            return logits, lq_feat.permute(0,2,3,1)

        # ################# Quantization ###################
        # if self.training:
        #     quant_feat = torch.einsum('btn,nc->btc', [soft_one_hot, self.quantize.embedding.weight])
        #     # b(hw)c -> bc(hw) -> bchw
        #     quant_feat = quant_feat.permute(0,2,1).view(lq_feat.shape)
        # ------------
        # soft_one_hot = F.softmax(logits, dim=2)
        # _, top_idx = torch.topk(soft_one_hot, 1, dim=2)
        codes = logits.argmax(-1)  # [2, 16, 16, 1] 
        quant_feat = self.quantizer.embed_code(codes).permute(0, 3, 1, 2).contiguous()
        # preserve gradients
        # quant_feat = lq_feat + (quant_feat - lq_feat).detach()

        if detach_16:
            quant_feat = quant_feat.detach() # for training stage III
        if adain:
            quant_feat = adaptive_instance_normalization(quant_feat, lq_feat)

        # ################## decoder ####################
        # z = quant_feat.permute(0, 3, 1, 2).contiguous()
        z = self.post_quant_conv(quant_feat)
        
        # btq,cq,hq,wq = z.shape
        # z = z.view(btq//self.t,self.t,cq,hq,wq)
        
        self.decoder.last_z_shape = z.shape
        # timestep embedding
        temb = None
        # z to block_in
        h = self.decoder.conv_in(z)
        # middle
        h = self.decoder.mid.block_1(h, temb)
        BDF,CF,HF,WF = h.shape
        h = h.view(BDF//self.t,self.t,CF,HF,WF)
        h = self.decoder.mid.attn_1(h)
        h = self.decoder.mid.block_2(h, temb)
        # upsampling
        for i_level in reversed(range(self.decoder.num_resolutions)):
            for i_block in range(self.decoder.num_res_blocks+1):
                h = self.decoder.up[i_level].block[i_block](h, temb)
                if len(self.decoder.up[i_level].attn) > 0:
                    h = self.decoder.up[i_level].attn[i_block](h)

            f_size = str(h.shape[-1])
            if str(f_size) in self.connect_list and w>0:
                h = self.fuse_convs_dict[f_size](enc_feat_dict[f_size].detach(), h, temb=None, w=w)

            if i_level != 0:
                h = self.decoder.up[i_level].upsample(h)

        BF,DF,CF,HF,WF = h.shape
        h = h.view(BF*DF,CF,HF,WF)

        h = self.decoder.norm_out(h)
        h = nonlinearity(h)
        h = self.decoder.conv_out(h)
        
        out = h
        # logits doesn't need softmax before cross_entropy loss
        return out, logits, lq_feat.permute(0,2,3,1)
    
    

from collections import OrderedDict
def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper



if __name__ == '__main__':
    import yaml
    with open('test.yml', mode='r') as f:
        opt = yaml.load(f, Loader=ordered_yaml()[0])
    ooo = opt['network_g']
    model = PGTFormer(**ooo).cuda()
    tensor = torch.ones((3,3,512,512)).cuda()
    import time
    with torch.no_grad():
        for i in range(21):
            if i==1:
                st = time.time()
            out = model(tensor,w=1)
            print(out[0].shape)
        print((time.time()-st)/60.0)

