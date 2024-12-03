import os
import torch
import numpy as np
import cv2
import math
from tqdm import trange
from torch import nn
from torch.nn import functional as F
from lf2disp.training import BaseTrainer
from lf2disp.utils.utils import depth_metric

class Trainer(BaseTrainer):
    def __init__(self, model, optimizer, criterion=nn.MSELoss, device=None, cfg=None):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.criterion = criterion()
        self.vis_dir = cfg['vis']['vis_dir']
        self.test_dir = cfg['test']['test_dir']
        self.vis_guide = cfg['vis']['guide_view']
        self.test_guide = cfg['test']['guide_view']
        self.aa_arr = np.array([0, 8, 1, 1, 7, 7, 2, 2, 2, 6, 6, 6, 3, 3, 3, 3, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4])
        if not os.path.exists(self.vis_dir):
            os.makedirs(self.vis_dir)
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)
        print("use model:", self.model)
        print("use loss:", self.criterion)

    def train_step(self, data, iter=0):
        """ Performs a training step.
        Args:
            data (dict): data dictionary
        """
        self.model.train()
        self.optimizer.zero_grad()
        loss = self.compute_loss(data, iter)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def eval_step(self, data, imgid=0):
        device = self.device
        self.model.eval()
        image = data.get('image').to(device)
        label = data.get('label').to(device)
        B1, B2, H, W, C, M, M = image.shape
        label = label.reshape(B1 * B2, H, W, -1)[:, :, :, self.test_guide]
        with torch.no_grad():
            out = self.model(image, self.test_guide)
            pred = out['pred'].reshape(B1 * B2, H, W)

        depthmap = pred.cpu().numpy().reshape(B1 * B2, H, W, 1)[0][15:-15, 15:-15]
        label = label.cpu().numpy().reshape(B1 * B2, H, W, 1)[0][15:-15, 15:-15]

        metric = depth_metric(label, depthmap)
        metric['id'] = imgid
        return metric

    def visualize(self, data, id=0, vis_dir=None):
        self.model.eval()
        device = self.device
        if vis_dir is None:
            vis_dir = self.vis_dir

        image = data.get('image').to(device)
        label = data.get('label').to(device)
        B1, B2, H, W, C, M, M = image.shape
        label = label.reshape(B1 * B2, H, W, -1)[:, :, :, self.vis_guide]
        with torch.no_grad():
            out = self.model(image, self.vis_guide)
            pred = out['pred'].reshape(B1 * B2, H, W)

        depthmap = pred.cpu().numpy().reshape(B1 * B2, H, W, 1)[0][15:-15, 15:-15]
        label = label.cpu().numpy().reshape(B1 * B2, H, W, 1)[0][15:-15, 15:-15]

        depthmap = (depthmap - label.min()) / (label.max() - label.min())
        label = (label - label.min()) / (label.max() - label.min())

        path = os.path.join(vis_dir, str(id) + '_pred.png')
        labelpath = os.path.join(vis_dir, '%03d_label.png' % id)

        cv2.imwrite(path, depthmap.copy() * 255.0)
        print('save depth map in', path)
        cv2.imwrite(labelpath, label.copy() * 255.0)
        print('save label in', labelpath)

    def compute_loss(self, data, iter=0):
        device = self.device
        image = data.get('image').to(device)
        #
        B1, B2, H, W, C, M, M = image.shape
        v = np.random.choice(self.aa_arr)
        u = np.random.choice(self.aa_arr)
        guide_index = v * M + u
        #
        label = data.get('label').reshape(B1 * B2, H, W, -1)[:, :, :, guide_index].reshape(B1 * B2, H,W).to(device)
        pred = self.model(image, guide_index)['pred']
        #
        pred = pred.reshape(B1*B2, -1)
        label = label.reshape(B1*B2, -1)
        loss = self.criterion(pred, label).mean()
        return loss
