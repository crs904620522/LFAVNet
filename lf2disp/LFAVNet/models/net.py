import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from torchsummary import summary
from torchstat import stat
import os
import numpy as np
import math
import torch.nn.functional as F
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

class CResnetBlockConv3d(nn.Module):
    ''' Conditional batch normalization-based Resnet block class.'''

    def __init__(self, img_dim, cv_dim,output_dim, kernel_size=3,padding=1):
        super().__init__()
        # Submodules
        self.bn_0 = CBatchNorm3d(img_dim, cv_dim)
        self.bn_1 = CBatchNorm3d(img_dim, output_dim)

        self.conv_0 = nn.Conv3d(cv_dim, output_dim, kernel_size=kernel_size,padding=padding)
        self.conv_1 = nn.Conv3d(output_dim, output_dim, kernel_size=kernel_size,padding=padding)
        self.actvn = nn.LeakyReLU(0.1,inplace=True)
        self.shortcut = nn.Conv3d(cv_dim, output_dim, 1, bias=False)
        # Initialization
        nn.init.zeros_(self.conv_1.weight)
    def forward(self, x, c):
        net = self.conv_0(self.actvn(self.bn_0(x, c)))
        dx = self.conv_1(self.actvn(self.bn_1(net, c)))

        x_s = self.shortcut(x)

        return x_s + dx


class CBatchNorm3d(nn.Module):
    ''' Conditional batch normalization layer class.

    Args:
        c_dim (int): dimension of latent conditioned code c
        f_dim (int): feature dimension
        norm_method (str): normalization method
    '''

    def __init__(self, c_dim, f_dim):
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim
        # Submodules
        self.conv_gamma = nn.Conv2d(c_dim, f_dim, kernel_size=1)
        self.conv_beta = nn.Conv2d(c_dim, f_dim, kernel_size=1)
        self.bn = nn.BatchNorm3d(f_dim, affine=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.conv_gamma.weight)
        nn.init.zeros_(self.conv_beta.weight)
        nn.init.ones_(self.conv_gamma.bias)
        nn.init.zeros_(self.conv_beta.bias)

    def forward(self, cv, img):
        B, C, H, W, N = cv.shape
        B, C, H, W = img.shape
        # Affine mapping
        gamma = self.conv_gamma(img).reshape(B, -1, H, W, 1)
        beta = self.conv_beta(img).reshape(B, -1, H, W, 1)
        # Batchnorm
        net = self.bn(cv)
        out = gamma * net + beta
        return out



class CBatchNorm2d(nn.Module):
    ''' Conditional batch normalization layer class.

    Args:
        c_dim (int): dimension of latent conditioned code c
        f_dim (int): feature dimension
        norm_method (str): normalization method
    '''

    def __init__(self, c_dim, f_dim):
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim
        # Submodules
        self.conv_gamma = nn.Conv2d(c_dim, f_dim, kernel_size=1)
        self.conv_beta = nn.Conv2d(c_dim, f_dim, kernel_size=1)
        self.bn = nn.BatchNorm2d(f_dim, affine=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.conv_gamma.weight)
        nn.init.zeros_(self.conv_beta.weight)
        nn.init.ones_(self.conv_gamma.bias)
        nn.init.zeros_(self.conv_beta.bias)

    def forward(self, lfs, img):
        B, C, H, W = lfs.shape
        B, C, H, W = img.shape
        # Affine mapping
        gamma = self.conv_gamma(img)
        beta = self.conv_beta(img)
        # Batchnorm
        net = self.bn(lfs)
        out = gamma * net + beta
        return out



def convbn(in_channels, out_channels, kernel_size, stride, padding, dilation):
    convbn = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm2d(out_channels),
    )
    return convbn


def convbn_3d(in_channels, out_channels, kernel_size, stride, padding):
    convbn_3d = nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.BatchNorm3d(out_channels),
    )
    return convbn_3d


class BasicBlock(nn.Module):

    def __init__(self, input_dim, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_dim, out_channels, kernel_size=3, stride=stride, padding=1, dilation=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(out_channels),
        )

        if stride != 1 or out_channels != 4:
            self.downsample = nn.Sequential(
                nn.Conv2d(input_dim, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = nn.Sequential()

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.downsample(x)
        out = x1 + x2
        return out
# -------------------------------------- layers ----------------------#
class feature_extraction(nn.Module):
    def __init__(self, input_dim, device=None):
        super(feature_extraction, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(input_dim, 4, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(4),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.layers = list()
        input_dim = 4
        numblock = [2, 8, 2, 2]
        output_dim = [4, 8, 16, 16]
        for i in range(0, 4):
            temp = self._make_layer(input_dim, output_dim[i], numblock[i], 1)
            self.layers.append(temp)
            input_dim = output_dim[i]
        self.layers = nn.Sequential(*self.layers)
        # SPP Module
        self.branchs = list()
        output_dim = [4, 4, 4, 4]
        size = [2, 4, 8, 16]
        for i in range(0, 4):
            temp = nn.Sequential(
                nn.AvgPool2d((size[i], size[i]), (size[i], size[i])),
                nn.Conv2d(input_dim, output_dim[i], kernel_size=1, stride=1, dilation=1),
                nn.BatchNorm2d(output_dim[i]),
                nn.LeakyReLU(0.1, inplace=True),
            )
            self.branchs.append(temp)
        self.branchs = nn.Sequential(*self.branchs)
        input_dim = np.array(output_dim).sum() + 8 + 16
        self.last = nn.Sequential(
            nn.Conv2d(input_dim, 32, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1, bias=False)
        )

    def _make_layer(self, input_dim, out_channels, blocks, stride):
        layers = list()
        layers.append(BasicBlock(input_dim, out_channels, stride))
        for i in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels, 1))
        layers = nn.Sequential(*layers)
        return layers

    def forward(self, x):
        x = self.conv1(x)
        layers_out = [x]
        for i in range(len(self.layers)):
            layers_out.append(self.layers[i](layers_out[-1]))
        layer4_size = layers_out[-1].shape
        branchs_out = []
        for i in range(len(self.branchs)):
            temp = self.branchs[i](layers_out[-1])
            temp = nn.UpsamplingBilinear2d(size=(int(layer4_size[-2]), int(layer4_size[-1])))(temp)
            branchs_out.append(temp)
        cat_f = [layers_out[2], layers_out[4]] + branchs_out
        feature = torch.cat([i for i in cat_f], dim=1)
        out = self.last(feature)
        return out


class Coordinate_guided_Aggregation(nn.Module):
    def __init__(self, input_dims=160, position_dim=16, hidden_dims=160, n_views=9, device=None):
        super(Coordinate_guided_Aggregation, self).__init__()
        self.views = n_views
        self.conv1 = CResnetBlockConv3d(img_dim=position_dim, cv_dim=input_dims, output_dim=hidden_dims)
        self.conv2 = CResnetBlockConv3d(img_dim=position_dim, cv_dim=hidden_dims, output_dim=hidden_dims)
        self.conv3 = CResnetBlockConv3d(img_dim=position_dim, cv_dim=hidden_dims, output_dim=hidden_dims)
        self.conv4 = CResnetBlockConv3d(img_dim=position_dim, cv_dim=hidden_dims, output_dim=hidden_dims)
        self.cls = nn.Sequential(nn.Conv3d(hidden_dims, 1, kernel_size=1, stride=1, padding=0, bias=False))

    def forward(self, cv, img):
        cv = self.conv1(cv, img)
        cv = self.conv2(cv, img)
        cv = self.conv3(cv, img)
        cv = self.conv4(cv, img)
        x = self.cls(cv)
        return x


class QueryKeyValue(nn.Module):
    def __init__(self, input_dims=16, output_dims=160, n_views=9, d_nums=9, n_dims=16,c_dim=64, device=None):
        super(QueryKeyValue, self).__init__()

        self.device = device
        self.output_dims = output_dims
        self.d_nums = d_nums
        self.n_views = n_views
        len_u, len_v = n_views, n_views
        # pixel wise local
        self.pixel_query = nn.Conv2d(input_dims, n_dims, kernel_size=1)
        self.pixel_key = nn.Conv3d(input_dims, n_dims, kernel_size=1)
        # view wise global
        self.view_query = nn.Conv2d(input_dims, n_dims,kernel_size=1)
        self.view_key = nn.Conv2d(input_dims, n_dims, kernel_size=1)
        # value
        self.value = nn.Conv3d(input_dims, n_dims, kernel_size=1)
        # position
        self.position_embeding = nn.Conv2d(in_channels=2,out_channels=c_dim,kernel_size=1)
        self.cbn = CBatchNorm2d(c_dim=c_dim, f_dim=n_dims * n_views ** 2)  # 位置信息
        # output
        self.last = nn.Conv2d(n_dims * n_views ** 2, self.output_dims, kernel_size=1)

    def forward(self, x, guide_index):
        B, C, MM, H, W = x.shape
        M = int(math.sqrt(MM))
        guide_image = x[:, :, guide_index, :, :].reshape(B, -1, H, W)
        (guide_v, guide_u) = divmod(guide_index, M)
        # pixel
        q = self.pixel_query(guide_image).view(B, -1, 1, H, W)
        k = self.pixel_key(x).view(B, -1, M * M, H, W)
        pixel_energy = (q * k).reshape(B, -1, M * M, H, W)
        # view
        guide_mean = nn.AdaptiveAvgPool2d(1)(guide_image)
        x_mean = nn.AdaptiveAvgPool2d(1)(x.reshape(B, -1, H, W)).reshape(B, -1, M * M, 1)
        q = self.view_query(guide_mean).view(B, -1, 1)
        k = self.view_key(x_mean).view(B, -1, M * M)
        view_energy = (q * k).reshape(B, -1, M * M, 1, 1)
        # attention
        energy = (pixel_energy + view_energy).reshape(B, -1, MM, H, W)
        attention = nn.Sigmoid()(energy)
        value = self.value(x).view(B, -1, MM, H, W)
        out = (value * attention).view(B, -1, H, W)
        # position
        position = torch.ones((B, 2, H, W)).to(self.device)
        position[:,0] = position[:,0] * guide_v / M
        position[:,1] = position[:,1] * guide_u / M
        # fusion
        position = self.position_embeding(position)
        out = self.cbn(out, position)
        out = self.last(out).view(B, -1, H, W)
        return out, position


class Net(nn.Module):
    def __init__(self, input_dim=1, n_views=9, device=None):
        super(Net, self).__init__()
        self.device = device
        self.n_views = n_views

        self.FE = feature_extraction(input_dim)
        self.QKV = QueryKeyValue(input_dims=16, output_dims=160, n_views=9, n_dims=16, c_dim=64, device=self.device)
        self.CGA = Coordinate_guided_Aggregation(input_dims=160, position_dim=64, hidden_dims=150, n_views=9, device=self.device)
        self.Regression = nn.Softmax(dim=4)

    def forward(self, input, guide_index=40):
        # input
        B, H, W, C, M, M = input.shape  #
        # Feature Extraction
        x = input.permute(0, 4, 5, 3, 1, 2).reshape(B * M * M, C, H, W)
        x = self.FE(x)
        # Image_guide_Construction
        _, C, _, _ = x.shape
        x = x.reshape(B, M * M, C, H, W).permute(1, 0, 2, 3, 4)
        view_list = list()
        for i in range(0, M * M):
            view_list.append(x[i, :])
        (guide_v, guide_u) = divmod(guide_index, M)
        disparity_costs = list()
        for d in range(-4, 5):
            if d == 0:
                tmp_list = list()
                for i in range(len(view_list)):
                    temp = view_list[i]
                    tmp_list.append(temp.unsqueeze(dim=2))  # B,C,1,H,W
            else:
                tmp_list = list()
                for i in range(len(view_list)):
                    (v, u) = divmod(i, M)
                    rate = [2 * d * (u - guide_u) / W, 2 * d * (v - guide_v) / H]  # 这个只和对齐操作有关，没关系的
                    theta = torch.tensor([[1, 0, rate[0]], [0, 1, rate[1]]], dtype=float).to(self.device)
                    grid = F.affine_grid(theta.unsqueeze(0).repeat(B, 1, 1), view_list[i].size()).type_as(view_list[i])
                    temp = F.grid_sample(view_list[i], grid)
                    tmp_list.append(temp.unsqueeze(dim=2))  # B,C,1,H,W
            cost = torch.cat([i for i in tmp_list], dim=2)  # B,C,MM,H,W
            cost, guide_position = self.QKV(cost, guide_index)  # B,C,H,W
            disparity_costs.append(cost.unsqueeze(dim=-1))  # B,C,H,W,1
        cv = torch.cat([i for i in disparity_costs], dim=-1)  # B,C,H,W,N
        # Coordinate-guided Aggregation
        cv = cv.reshape(B, -1, H, W, 9)
        cv = self.CGA(cv, guide_position)  # B,1,H,W,9
        # Regression
        attention = self.Regression(cv)
        disparity_values = torch.linspace(-4, 4, 9).to(self.device)
        disparity_values = disparity_values.reshape(1, 1, 1, 9)
        out = (attention * disparity_values).sum(dim=-1)
        out = out.reshape(B, H, W, 1)
        return out


if __name__ == "__main__":
    from thop import profile
    input = torch.ones(1, 512, 512, 1, 9, 9).to('cuda:0')
    encoder = Net(input_dim=1, n_views=9, device='cuda:0').to('cuda:0')
    total_num = sum(p.numel() for p in encoder.parameters())
    trainable_num = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print('Total', total_num, 'Trainable', trainable_num)

    import time
    start = time.time()
    with torch.no_grad():
        for i in range(1):
            input = torch.ones(1, 512, 512, 1, 9, 9).cuda()
            encoder(input)
            torch.cuda.empty_cache()
    print((time.time()-start)/10)
