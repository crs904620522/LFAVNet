# -*- coding: utf-8 -*-
import numpy as np
from einops import rearrange
import torch
import torch.nn.functional as F
def read_pfm(fpath, expected_identifier="Pf",print_limit=30):
    # PFM format definition: http://netpbm.sourceforge.net/doc/pfm.html

    def _get_next_line(f):
        next_line = f.readline().decode('utf-8').rstrip()
        # ignore comments
        while next_line.startswith('#'):
            next_line = f.readline().rstrip()
        return next_line

    with open(fpath, 'rb') as f:
        #  header
        identifier = _get_next_line(f)
        if identifier != expected_identifier:
            raise Exception('Unknown identifier. Expected: "%s", got: "%s".' % (expected_identifier, identifier))

        try:
            line_dimensions = _get_next_line(f)
            dimensions = line_dimensions.split(' ')
            width = int(dimensions[0].strip())
            height = int(dimensions[1].strip())
        except:
            raise Exception('Could not parse dimensions: "%s". '
                            'Expected "width height", e.g. "512 512".' % line_dimensions)

        try:
            line_scale = _get_next_line(f)
            scale = float(line_scale)
            assert scale != 0
            if scale < 0:
                endianness = "<"
            else:
                endianness = ">"
        except:
            raise Exception('Could not parse max value / endianess information: "%s". '
                            'Should be a non-zero number.' % line_scale)

        try:
            data = np.fromfile(f, "%sf" % endianness)
            data = np.reshape(data, (height, width))
            data = np.flipud(data)
            with np.errstate(invalid="ignore"):
                data *= abs(scale)
        except:
            raise Exception('Invalid binary values. Could not create %dx%d array from input.' % (height, width))

        return data

import sys
def write_pfm(data, fpath, scale=1, file_identifier=b'Pf', dtype="float32"):
    # PFM format definition: http://netpbm.sourceforge.net/doc/pfm.html

    data = np.flipud(data)
    height, width = np.shape(data)[:2]
    values = np.ndarray.flatten(np.asarray(data, dtype=dtype))
    endianess = data.dtype.byteorder
    print(endianess)

    if endianess == '<' or (endianess == '=' and sys.byteorder == 'little'):
        scale *= -1

    with open(fpath, 'wb') as file:
        file.write((file_identifier))
        file.write(('\n%d %d\n' % (width, height)).encode())
        file.write(('%d\n' % scale).encode())
        file.write(values)
    file.close()

def cal_mse(label, pre):  # 计算mse 输入为numpy数组

    mae = np.abs(label - pre)
    mean_mse = 100 * np.average(np.square(mae))
    return mean_mse


def cal_mae(label, pre):  # 计算mae 输入为numpy数组
    mae = np.abs(label - pre)
    mean_mae = 100 * np.average(mae)
    return mean_mae


def cal_bp(label, pre):
    mae = np.abs(label - pre)
    bp1 = 100 * np.average((mae >= 0.01))
    bp3 = 100 * np.average((mae >= 0.03))
    bp7 = 100 * np.average((mae >= 0.07))
    return bp1, bp3, bp7


def depth_metric(label, pre):
    metric = {}
    metric['mae'] = cal_mae(label, pre)
    metric['mse'] = cal_mse(label, pre)
    metric['bp1'], metric['bp3'], metric['bp7'] = cal_bp(label, pre)
    return metric



def ImageExtend(Im, bdr):
    [_, _, h, w] = Im.size()
    Im_lr = torch.flip(Im, dims=[-1])
    Im_ud = torch.flip(Im, dims=[-2])
    Im_diag = torch.flip(Im, dims=[-1, -2])

    Im_up = torch.cat((Im_diag, Im_ud, Im_diag), dim=-1)
    Im_mid = torch.cat((Im_lr, Im, Im_lr), dim=-1)
    Im_down = torch.cat((Im_diag, Im_ud, Im_diag), dim=-1)
    Im_Ext = torch.cat((Im_up, Im_mid, Im_down), dim=-2)
    Im_out = Im_Ext[:, :, h - bdr[0]: 2 * h + bdr[1], w - bdr[2]: 2 * w + bdr[3]]

    return Im_out

def LFdivide(lf, patch_size, stride):
    U, V, C, H, W = lf.shape
    data = rearrange(lf, 'u v c h w -> (u v) c h w')

    bdr = (patch_size - stride) // 2
    numU = (H + bdr * 2 - 1) // stride
    numV = (W + bdr * 2 - 1) // stride
    data_pad = ImageExtend(data, [bdr, bdr + stride - 1, bdr, bdr + stride - 1])
    subLF = F.unfold(data_pad, kernel_size=patch_size, stride=stride)
    subLF = rearrange(subLF, '(u v) (c h w) (n1 n2) -> n1 n2 u v c h w',
                      n1=numU, n2=numV, u=U, v=V, h=patch_size, w=patch_size)

    return subLF


def LFintegrate(subLFs, patch_size, stride):
    bdr = (patch_size - stride) // 2
    out = subLFs[:, :, :, bdr:bdr+stride, bdr:bdr+stride]
    out = rearrange(out, 'n1 n2 c h w -> (n1 h) (n2 w) c')

    return out.squeeze()


def SAIs2MacPI(SAIs):
    # B,H,W,C,V,U
    B,H,W,C,V,U = SAIs.shape
    SAIs = SAIs.permute(0,4,5,1,2,3).reshape(B, V, U, H, W, C)
    temp1 = list()
    for i in range(0, H):
        temp2 = list()
        for j in range(0, W):
            MM = SAIs[:, :, :, i, j]
            temp2.append(MM)
        temp1.append(torch.cat([mm for mm in temp2], dim=2))
    MacPI = torch.cat([mm for mm in temp1], dim=1)
    # B, HV,WU,C
    return MacPI

def MacPI2SAIs(MacPI, n_views=9):
    B, H, W, C =MacPI.shape
    SAIs = list()
    for u in range(0, n_views):  # 取中心9x9的视角
        for v in range(0, n_views):
            temp = MacPI[:,u::n_views, v::n_views, :]
            B, H, W, C = temp.shape
            SAIs.append(temp.unsqueeze(1))
    SAIs = torch.cat([i for i in SAIs], axis=1)
    SAIs = SAIs.reshape(B, n_views, n_views, H, W, C).permute(0,3,4,5,1,2) # B,H,W,C,M,M
    return SAIs