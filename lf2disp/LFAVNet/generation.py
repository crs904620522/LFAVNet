import torch
import torch.optim as optim
from torch import autograd
import numpy as np
from tqdm import trange
import os
import cv2
import scipy.io as scio
from lf2disp.utils.utils import depth_metric, write_pfm, LFdivide, LFintegrate
from einops import rearrange

class GeneratorDepth(object):

    def __init__(self, model, cfg=None, device=None):
        self.model = model.to(device)
        self.device = device
        self.generate_dir = cfg['generation']['generation_dir']
        self.name = cfg['generation']['name']
        self.dataset = cfg["data"]["dataset"]
        self.guide_index = cfg['generation']['guide_view']
        self.mode = cfg['generation']['mode']
        
        if not os.path.exists(self.generate_dir):
            os.makedirs(self.generate_dir)

    def generate_depth(self, data, id=0):
        ''' Generates the output depthmap '''
        device = self.device
        self.model.eval()
        
        image = data.get('image').to(device)
        labelMxM = data.get('label').to(device)
        B1, B2, H, W, C, M, M = image.shape
        mat_out = np.ones((H, W, M * M))

        for guide_index in self.guide_index:
            if len(labelMxM.shape) == 7:
                label = labelMxM.reshape(B1 * B2, H, W, -1)[:, :, :, guide_index]
            else:
                label = labelMxM

            with torch.no_grad():
                torch.cuda.empty_cache()
                
                if H > 512 or W > 512:
                    # crop
                    patchsize = 128
                    stride = patchsize // 2
                    data = image.reshape(H, W, C, M, M)
                    sub_lfs = LFdivide(data.permute(3, 4, 2, 0, 1), patchsize, stride)
                    n1, n2, u, v, c, h, w = sub_lfs.shape
                    sub_lfs = rearrange(sub_lfs, 'n1 n2 u v c h w -> (n1 n2) h w c u v')     
                    mini_batch = 16
                    num_inference = (n1 * n2) // mini_batch  
                    
                    out_disp = []
                    for idx_inference in range(num_inference):
                        current_lfs = sub_lfs[idx_inference * mini_batch: (idx_inference + 1) * mini_batch, :, :, :, :, :]
                        temp = self.model(current_lfs.unsqueeze(0), guide_index)['pred']
                        temp = temp.reshape(-1, 1, patchsize, patchsize)
                        out_disp.append(temp)

                    if (n1 * n2) % mini_batch:
                        current_lfs = sub_lfs[(idx_inference + 1) * mini_batch:, :, :, :, :, :]
                        temp = self.model(current_lfs.unsqueeze(0), guide_index)['pred']
                        temp = temp.reshape(-1, 1, patchsize, patchsize)
                        out_disp.append(temp)

                    out_disps = torch.cat(out_disp, dim=0)
                    out_disps = rearrange(out_disps, '(n1 n2) c h w -> n1 n2 c h w', n1=n1, n2=n2)
                    disp = LFintegrate(out_disps, patchsize, patchsize // 2)
                    pred = disp[0: H, 0: W]
                else:
                    temp = self.model(image, guide_index)['pred']
                    pred = temp.reshape(B1 * B2, H, W)

            depthmap = pred.cpu().numpy().reshape(B1 * B2, H, W, 1)[0]
            label = label.cpu().numpy().reshape(B1 * B2, H, W, 1)[0]
            metric = depth_metric(label[15:-15, 15:-15], depthmap[15:-15, 15:-15])
            metric['id'] = id
            
            print("-------------------------------------------------------------\n")
            print('                            ' + str(guide_index) + '                         \n ')
            print('result:', metric)
            
            mat_out[:, :, guide_index] = depthmap.reshape(H, W)
            
            if guide_index == 40:
                depthpath = os.path.join(self.generate_dir, self.name[id] + '_' + str(guide_index) + '.png')
                depth_fix = depthmap.reshape(H, W, 1)
                pfm_path = os.path.join(self.generate_dir, self.name[id] + '_' + str(guide_index) + '.pfm')
                write_pfm(depth_fix, pfm_path, scale=1.0)
                depth_img = 255 * (depth_fix - depth_fix.min()) / (depth_fix.max() - depth_fix.min())
                cv2.imwrite(depthpath, np.uint8(depth_img))

        mat_out = mat_out.reshape(H, W, M, M)
        save_mat_path = os.path.join(self.generate_dir, self.name[id] + '.mat')
        scio.savemat(save_mat_path, {'D': mat_out})

        return metric